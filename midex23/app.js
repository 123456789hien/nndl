// app.js
// All UI wiring, charts, model training, scaling, prediction

// NOTE: relies on globals from data-loader.js:
// rawTrain, rawTest, monthlyAgg (combined view), monthlyTrainAgg, monthlyTestAgg

let monthly = []; // aggregated monthly rows used for UI (monthlyAgg)
let monthlyTrain = []; // aggregated for training
let monthlyTest = [];  // aggregated for test/eval
let groups = [];  // distinct group keys (room_type)
let scaler = {min:[], max:[]}; // per feature min/max [cancellation_rate, avg_room_price, lead_time_avg]

// Chart instances
let chartCancel=null, chartPrice=null, chartHeat=null, chartPredict=null;

// UI refs
const btnLoad = document.getElementById('btn-load');
const noteLoad = document.getElementById('note-load');
const mergeDiv = document.getElementById('merge-overview');
const missDiv = document.getElementById('missing-table');
const statsDiv = document.getElementById('stats-table');
const smoothingSlider = document.getElementById('smoothing-slider');
const smoothingVal = document.getElementById('smoothing-val');
const interpMethod = document.getElementById('interp-method');
const groupKeySel = document.getElementById('group-key');
const viewRangeSel = document.getElementById('view-range');

const btnTrain = document.getElementById('btn-train');
const btnDownload = document.getElementById('btn-download');
const trainStatus = document.getElementById('train-status');
const testEvalDiv = document.getElementById('test-eval');

const selGroup = document.getElementById('sel-group');
const inpYear = document.getElementById('inp-year');
const inpMonth = document.getElementById('inp-month');
const btnPredict = document.getElementById('btn-predict');
const predictResult = document.getElementById('predict-result');
const insightBox = document.getElementById('insight');

// helpers
function formatPct(x){ if (x===null || x===undefined || Number.isNaN(x)) return 'NaN'; return (x*100).toFixed(2) + '%'; }
function formatNum(x){ return (x===null || x===undefined || Number.isNaN(x)) ? 'NaN' : Number(x).toFixed(2); }
function rmse(arrTrue, arrPred){
  const n = arrTrue.length;
  if (n === 0) return null;
  let s = 0;
  for (let i=0;i<n;i++){ s += Math.pow(arrTrue[i]-arrPred[i],2); }
  return Math.sqrt(s/n);
}

// load event
btnLoad.addEventListener('click', async ()=>{
  const trainFile = document.getElementById('train-file').files[0];
  const testFile = document.getElementById('test-file').files[0];
  if (!trainFile || !testFile) { alert('Please choose both train and test CSV files (semicolon separated).'); return; }
  noteLoad.innerText = 'Status: parsing CSV ...';
  try {
    // prepareMonthly sets global monthlyAgg, monthlyTrainAgg, monthlyTestAgg
    monthly = await prepareMonthly(trainFile, testFile);
    // copy reference to train/test
    monthlyTrain = (typeof monthlyTrainAgg !== 'undefined') ? monthlyTrainAgg : [];
    monthlyTest = (typeof monthlyTestAgg !== 'undefined') ? monthlyTestAgg : [];
    groups = [...new Set(monthly.map(d=>d.group))];

    // show uploaded counts
    const trainCount = rawTrain ? rawTrain.length : 0;
    const testCount = rawTest ? rawTest.length : 0;
    noteLoad.innerText = `Loaded files — train rows: ${trainCount}, test rows: ${testCount}. Aggregated monthly rows (combined view): ${monthly.length} across ${groups.length} groups.`;

    renderTables();
    renderEDACharts();
    populateGroupDropdown();
  } catch (err){
    console.error(err);
    noteLoad.innerText = 'Error parsing files: ' + err.message;
  }
});

// render merge & missing & stats
function renderTables(){
  // show first 10 rows preview (aggregated monthly preview)
  const head = monthly.slice(0,10);
  let html = `<table><thead><tr><th>group</th><th>year</th><th>month</th><th>month_index</th><th>total_bookings</th><th>cancelled</th><th>cancellation_rate</th><th>avg_room_price</th><th>lead_time_avg</th></tr></thead><tbody>`;
  head.forEach(r=>{
    html += `<tr>
      <td>${r.group}</td>
      <td>${r.year}</td>
      <td>${r.month}</td>
      <td>${r.month_index}</td>
      <td>${r.total_bookings}</td>
      <td>${r.cancelled_bookings}</td>
      <td>${formatPct(r.cancellation_rate)}</td>
      <td>${formatNum(r.avg_room_price)}</td>
      <td>${formatNum(r.lead_time_avg)}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  mergeDiv.innerHTML = '<h3>Merge & Overview (aggregated preview — first 10 rows)</h3>' + html;

  // --- Raw Data Preview: 10 rows from train & 10 rows from test ---
  const rawPreview = [];
  if (typeof rawTrain !== 'undefined') rawPreview.push(...rawTrain.slice(0,10).map(r=>({source:'train', ...r})));
  if (typeof rawTest !== 'undefined') rawPreview.push(...rawTest.slice(0,10).map(r=>({source:'test', ...r})));
  let rawHtml = '<h3>Raw Data Preview (10 rows from each dataset)</h3>';
  if (rawPreview.length > 0) {
    rawHtml += '<table><thead><tr>';
    const keys = Object.keys(rawPreview[0]);
    keys.forEach(k => rawHtml += `<th>${k}</th>`);
    rawHtml += '</tr></thead><tbody>';
    rawPreview.forEach(r => {
      rawHtml += '<tr>' + keys.map(k => `<td>${r[k] ?? ''}</td>`).join('') + '</tr>';
    });
    rawHtml += '</tbody></table>';
  } else {
    rawHtml += '<p>No raw data loaded.</p>';
  }
  mergeDiv.innerHTML += rawHtml;

  // --- Mean per month per room type (aggregated overview) ---
  const meanByMonthRoom = {};
  monthly.forEach(r => {
    const key = `${r.group}||${r.month}`;
    if (!meanByMonthRoom[key]) meanByMonthRoom[key] = { group: r.group, month: r.month, cnt:0, sum_cancel:0, sum_price:0, sum_lead:0 };
    meanByMonthRoom[key].cnt++;
    meanByMonthRoom[key].sum_cancel += (r.cancellation_rate ?? 0);
    meanByMonthRoom[key].sum_price += (r.avg_room_price ?? 0);
    meanByMonthRoom[key].sum_lead += (r.lead_time_avg ?? 0);
  });
  const meanArr = Object.values(meanByMonthRoom).map(d => ({
    group: d.group,
    month: d.month,
    cancellation_rate_mean: d.sum_cancel/d.cnt,
    avg_room_price_mean: d.sum_price/d.cnt,
    lead_time_avg_mean: d.sum_lead/d.cnt
  }));
  meanArr.sort((a,b)=> (a.group < b.group ? -1 : a.group>b.group?1: a.month - b.month));

  let meanHtml = '<h3>Mean per Month per Room Type (aggregated overview)</h3>';
  meanHtml += '<p>This table shows the average cancellation rate, room price, and lead time for each room type by month (≈144 rows expected).</p>';
  meanHtml += '<table><thead><tr><th>Room Type</th><th>Month</th><th>Cancellation Rate (Mean)</th><th>Avg Room Price (Mean)</th><th>Lead Time (Mean)</th></tr></thead><tbody>';
  meanArr.forEach(r=>{
    meanHtml += `<tr><td>${r.group}</td><td>${r.month}</td><td>${formatPct(r.cancellation_rate_mean)}</td><td>${formatNum(r.avg_room_price_mean)}</td><td>${formatNum(r.lead_time_avg_mean)}</td></tr>`;
  });
  meanHtml += '</tbody></table>';
  mergeDiv.innerHTML += meanHtml;

  // missing
  const miss = { cancellation_rate:0, avg_room_price:0, lead_time_avg:0 };
  monthly.forEach(r=>{
    if (r.cancellation_rate === null) miss.cancellation_rate++;
    if (r.avg_room_price === null) miss.avg_room_price++;
    if (r.lead_time_avg === null) miss.lead_time_avg++;
  });
  missDiv.innerHTML = `<h3>Missing values</h3>
    <table><tr><th>variable</th><th>missing_count</th></tr>
    <tr><td>cancellation_rate</td><td>${miss.cancellation_rate}</td></tr>
    <tr><td>avg_room_price</td><td>${miss.avg_room_price}</td></tr>
    <tr><td>lead_time_avg</td><td>${miss.lead_time_avg}</td></tr></table>`;

  // stats
  const numeric = ['cancellation_rate','avg_room_price','lead_time_avg'];
  let statsHtml = '<h3>Stats</h3><table><tr><th>var</th><th>mean</th><th>min</th><th>max</th></tr>';
  numeric.forEach(k=>{
    const vals = monthly.map(d=>d[k]).filter(v=>v !== null && v !== undefined && !Number.isNaN(v));
    const mean = vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length) : 0;
    statsHtml += `<tr><td>${k}</td><td>${mean.toFixed(4)}</td><td>${(vals.length?Math.min(...vals):0)}</td><td>${(vals.length?Math.max(...vals):0)}</td></tr>`;
  });
  statsHtml += '</table>';
  statsDiv.innerHTML = statsHtml;
}

// populate group dropdown
function populateGroupDropdown(){
  selGroup.innerHTML = '';
  groups.forEach(g => {
    const opt = document.createElement('option');
    opt.value = g; opt.text = g;
    selGroup.appendChild(opt);
  });
  // default predict inputs to last available year/month from combined monthly
  const last = monthly[monthly.length-1];
  if (last) { inpYear.value = last.year; inpMonth.value = last.month; }
}

// EDA charts (cancellation line, price hist, correlation heatmap)
function renderEDACharts(){
  renderCancellationLine();
  renderPriceHistogram();
  renderCorrelationHeatmap();
}

function filterByViewRange(rows){
  const view = viewRangeSel ? viewRangeSel.value : 'all';
  if (view === '2018') return rows.filter(r=>r.year === 2018);
  return rows; // all
}

function renderCancellationLine(){
  const ctx = document.getElementById('chart-cancel').getContext('2d');
  const rows = filterByViewRange(monthly);
  // aggregate global monthly average cancellation_rate (across groups) by month_index
  const byIndex = new Map();
  rows.forEach(r => {
    const idx = r.month_index;
    if (!byIndex.has(idx)) byIndex.set(idx, []);
    if (r.cancellation_rate !== null && r.cancellation_rate !== undefined) byIndex.get(idx).push(r.cancellation_rate);
  });
  const labels = [...byIndex.keys()].sort((a,b)=>a-b);
  const data = labels.map(i => {
    const arr = byIndex.get(i);
    return arr.length ? (arr.reduce((a,b)=>a+b,0)/arr.length) : null;
  });

  if (chartCancel) chartCancel.destroy();
  chartCancel = new Chart(ctx, {
    type: 'line',
    data: { labels: labels.map(i=>i.toString()), datasets: [{ label: 'Avg Cancellation Rate (monthly)', data, borderColor:'#01497c', tension:0.2, pointRadius:4, spanGaps:true }] },
    options: {
      responsive:true,
      plugins:{
        tooltip:{
          callbacks:{
            label: function(context){
              const raw = context.raw;
              const pct = (raw === null) ? 'NaN' : (raw*100).toFixed(2) + '%';
              // derive confidence by thresholds
              let conf = 'Low';
              if (raw >= 0.5) conf = 'High';
              else if (raw >= 0.2) conf = 'Medium';
              return `Value: ${raw===null?'NaN':raw.toFixed(4)} (${pct}) — Confidence: ${conf}`;
            }
          }
        }
      }
    }
  });
}

function renderPriceHistogram(){
  const ctx = document.getElementById('chart-price').getContext('2d');
  const rows = filterByViewRange(monthly);
  // histogram over avg_room_price values (drop null)
  const prices = rows.map(d=>d.avg_room_price).filter(v=>v!==null && v!==undefined && !Number.isNaN(v));
  if (prices.length === 0){
    if (chartPrice) chartPrice.destroy();
    return;
  }
  // create bins
  const bins = 20;
  const min = Math.min(...prices), max = Math.max(...prices);
  const step = (max - min) / bins || 1;
  const counts = Array.from({length:bins}, ()=>0);
  prices.forEach(p=>{
    let idx = Math.floor((p - min) / step);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] += 1;
  });
  const labels = counts.map((_,i)=>`${(min + i*step).toFixed(0)}-${(min + (i+1)*step).toFixed(0)}`);
  if (chartPrice) chartPrice.destroy();
  chartPrice = new Chart(ctx, {
    type:'bar',
    data:{ labels, datasets:[{ label:'Count', data: counts, backgroundColor:'#0288d1' }] },
    options:{
      responsive:true,
      plugins:{
        tooltip:{
          callbacks:{
            label: function(context){
              const val = context.raw;
              return `Count: ${val}`;
            }
          }
        }
      }
    }
  });
}

function renderCorrelationHeatmap(){
  const ctx = document.getElementById('chart-heatmap').getContext('2d');
  const rows = filterByViewRange(monthly);
  const vars = ['cancellation_rate','avg_room_price','lead_time_avg'];
  // compute corr matrix
  const series = vars.map(v => rows.map(d => (d[v]===null ? 0 : d[v])));
  function corr(a,b){
    const n = a.length;
    if (n === 0) return 0;
    const ma = a.reduce((x,y)=>x+y,0)/n; const mb = b.reduce((x,y)=>x+y,0)/n;
    let num = 0, s1 = 0, s2 = 0;
    for(let i=0;i<n;i++){ num += (a[i]-ma)*(b[i]-mb); s1 += (a[i]-ma)*(a[i]-ma); s2 += (b[i]-mb)*(b[i]-mb); }
    const den = Math.sqrt(s1*s2);
    return den === 0 ? 0 : num/den;
  }
  const data = [];
  for(let i=0;i<vars.length;i++){
    for(let j=0;j<vars.length;j++){
      const v = corr(series[i], series[j]);
      data.push({ x: i, y: j, v });
    }
  }
  if (chartHeat) chartHeat.destroy();
  chartHeat = new Chart(ctx, {
    type: 'matrix',
    data: { datasets: [{ label:'corr', data, backgroundColor: ctx=>{
      const val = ctx.dataset.data[ctx.dataIndex].v;
      const alpha = Math.min(1, Math.abs(val));
      return val >= 0 ? `rgba(1,73,124,${alpha})` : `rgba(210,50,45,${alpha})`;
    } }]},
    options: {
      responsive:true,
      scales: {
        x: { type:'linear', min: -0.5, max: vars.length - 0.5, ticks:{ callback: i => vars[i] } },
        y: { type:'linear', min: -0.5, max: vars.length - 0.5, ticks:{ callback: i => vars[i] } }
      },
      plugins:{ tooltip:{ callbacks:{ label: function(ctx){ const raw = ctx.raw; return `${vars[raw.x]} vs ${vars[raw.y]}: ${raw.v.toFixed(3)}` } } } }
    }
  });
}

// smoothing slider applied to displayed sequences — note this modifies a view copy (not original monthly array)
smoothingSlider.addEventListener('input', () => {
  smoothingVal.innerText = smoothingSlider.value;
  // re-render cancellation line with smoothing
  const windowSize = Number(smoothingSlider.value);
  // apply smoothing per group for cancellation_rate view only
  const groupsLocal = [...new Set(monthly.map(d=>d.group))];
  const smoothed = [];
  groupsLocal.forEach(g => {
    const seq = monthly.filter(x=>x.group===g).sort((a,b)=>a.month_index - b.month_index).map(r=>({ ...r }));
    for (let i=0;i<seq.length;i++){
      const start = Math.max(0, i-windowSize+1);
      const window = seq.slice(start, i+1).map(x => (x.cancellation_rate === null ? 0 : x.cancellation_rate));
      seq[i].cancellation_rate_smoothed = window.reduce((a,b)=>a+b,0)/window.length;
    }
    smoothed.push(...seq);
  });
  const byIndex = new Map();
  smoothed.forEach(r=>{
    if (!byIndex.has(r.month_index)) byIndex.set(r.month_index, []);
    if (r.cancellation_rate_smoothed !== null && r.cancellation_rate_smoothed !== undefined) byIndex.get(r.month_index).push(r.cancellation_rate_smoothed);
  });
  const labels = [...byIndex.keys()].sort((a,b)=>a-b);
  const data = labels.map(i => {
    const arr = byIndex.get(i);
    return arr.length ? (arr.reduce((a,b)=>a+b,0)/arr.length) : null;
  });
  if (chartCancel) {
    chartCancel.data.labels = labels.map(l => l.toString());
    chartCancel.data.datasets[0].data = data;
    chartCancel.update();
  }
});

// TRAIN LSTM
let model = null;
function computeScaler(featuresArray){
  const dims = featuresArray[0].length;
  const mins = new Array(dims).fill(Infinity);
  const maxs = new Array(dims).fill(-Infinity);
  featuresArray.forEach(r=>{
    r.forEach((v,i)=>{
      const val = (v === null || v === undefined) ? 0 : v;
      if (val < mins[i]) mins[i] = val;
      if (val > maxs[i]) maxs[i] = val;
    });
  });
  scaler = { min: mins, max: maxs };
}

function scaleSample(sample){
  return sample.map((v,i)=>{
    const val = (v === null || v === undefined) ? 0 : v;
    const min = scaler.min[i], max = scaler.max[i];
    if (max === min) return 0;
    return (val - min) / (max - min);
  });
}

btnTrain.addEventListener('click', async ()=>{
  // prepare sequences per group using monthlyTrain (train-only)
  if (!monthlyTrain || monthlyTrain.length === 0){ alert('No monthly train aggregated data available. Load CSV first.'); return; }

  const seqLen = 12;
  const groupsLocal = [...new Set(monthlyTrain.map(d=>d.group))];
  const X = []; const Y = [];
  groupsLocal.forEach(g=>{
    const seq = monthlyTrain.filter(x=>x.group===g).sort((a,b)=>a.month_index - b.month_index);
    // forward fill per group to avoid nulls
    let lastCancel = 0, lastPrice = 0, lastLead = 0;
    seq.forEach(s=>{
      if (s.cancellation_rate === null) s.cancellation_rate = lastCancel; else lastCancel = s.cancellation_rate;
      if (s.avg_room_price === null) s.avg_room_price = lastPrice; else lastPrice = s.avg_room_price;
      if (s.lead_time_avg === null) s.lead_time_avg = lastLead; else lastLead = s.lead_time_avg;
    });
    for (let i = seqLen; i < seq.length; i++){
      const window = seq.slice(i-seqLen, i);
      const features = window.map(w => [ w.cancellation_rate ?? 0, w.avg_room_price ?? 0, w.lead_time_avg ?? 0 ]);
      const target = seq[i].cancellation_rate ?? 0;
      X.push(features); // shape [12,3]
      Y.push([target]);
    }
  });

  if (X.length === 0){ alert('Not enough data to build sequences from train (need at least 12 months per group).'); return; }

  // compute scaler across all features (flattened per time-step)
  const featsFlat = X.flat();
  computeScaler(featsFlat);

  // scale X and Y
  const Xs = X.map(seq => seq.map(sample => scaleSample(sample)));
  const Ys = Y.map(v => v); // cancellation_rate between 0-1 already

  // build tensors
  const Xtensor = tf.tensor3d(Xs); // shape [N,12,3]
  const Ytensor = tf.tensor2d(Ys);  // shape [N,1]

  // build LSTM model: inputShape [12,3]
  model = tf.sequential();
  model.add(tf.layers.lstm({ units:50, returnSequences:true, inputShape:[12,3] }));
  model.add(tf.layers.lstm({ units:50, returnSequences:true }));
  model.add(tf.layers.lstm({ units:50 }));
  model.add(tf.layers.dense({ units:1, activation:'linear' }));
  model.compile({ optimizer: 'adam', loss:'meanSquaredError' });

  const epochs = Number(document.getElementById('input-epochs').value) || 50;
  const batch = Number(document.getElementById('input-batch').value) || 32;

  trainStatus.innerText = 'Training... (this may take a while)';
  await model.fit(Xtensor, Ytensor, { epochs, batchSize: batch, shuffle:true, callbacks: { onEpochEnd: async (epoch, logs) => {
    trainStatus.innerText = `Training epoch ${epoch+1}/${epochs} — loss ${(logs.loss||0).toFixed(5)}`;
  }}}); 
  trainStatus.innerText = 'Training done';
  btnDownload.disabled = false;

  // After training: evaluate on monthlyTest (out-of-sample) if possible
  if (monthlyTest && monthlyTest.length > 0){
    // build test sequences from monthlyTest by group
    const Xtest = []; const Ytest = [];
    const groupsTest = [...new Set(monthlyTest.map(d=>d.group))];
    groupsTest.forEach(g=>{
      const seq = monthlyTest.filter(x=>x.group===g).sort((a,b)=>a.month_index - b.month_index);
      // forward fill
      let lastCancel = 0, lastPrice = 0, lastLead = 0;
      seq.forEach(s=>{
        if (s.cancellation_rate === null) s.cancellation_rate = lastCancel; else lastCancel = s.cancellation_rate;
        if (s.avg_room_price === null) s.avg_room_price = lastPrice; else lastPrice = s.avg_room_price;
        if (s.lead_time_avg === null) s.lead_time_avg = lastLead; else lastLead = s.lead_time_avg;
      });
      for (let i = seqLen; i < seq.length; i++){
        const window = seq.slice(i-seqLen, i);
        const features = window.map(w => [ w.cancellation_rate ?? 0, w.avg_room_price ?? 0, w.lead_time_avg ?? 0 ]);
        const target = seq[i].cancellation_rate ?? 0;
        Xtest.push(features);
        Ytest.push(target);
      }
    });
    if (Xtest.length > 0){
      const XtestScaled = Xtest.map(seq => seq.map(samp => scaleSample(samp)));
      const xt = tf.tensor3d(XtestScaled);
      const preds = await model.predict(xt).array();
      const predArr = preds.map(x => x[0]);
      const rm = rmse(Ytest, predArr);
      testEvalDiv.innerText = `Test evaluation (out-of-sample): RMSE = ${rm !== null ? rm.toFixed(5) : 'N/A'} (based on ${Ytest.length} test sequences)`;
    } else {
      testEvalDiv.innerText = 'Test evaluation: Not enough aggregated monthly history in test to compute sequences.';
    }
  } else {
    testEvalDiv.innerText = 'Test evaluation: no test monthly data loaded.';
  }
});

// Download model
btnDownload.addEventListener('click', async ()=>{
  if (!model) { alert('No model to download'); return; }
  await model.save('downloads://hotel_model');
});

// upload model files and load
document.getElementById('model-json').addEventListener('change', ()=>{ /* placeholder */ });
document.getElementById('model-bin').addEventListener('change', ()=>{ /* placeholder */ });

async function loadModelFromFiles(jsonFile, binFile){
  if (!jsonFile || !binFile) { alert('Select both json and bin files'); return null; }
  const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
  return model;
}

// predict
btnPredict.addEventListener('click', async ()=>{
  // load model if user uploaded files
  let usedModel = model;
  const jsonInput = document.getElementById('model-json').files[0];
  const binInput = document.getElementById('model-bin').files[0];
  if (!usedModel && jsonInput && binInput){
    try {
      usedModel = await loadModelFromFiles(jsonInput, binInput);
      trainStatus.innerText = 'Model loaded from files';
    } catch(err){
      console.error(err); alert('Error loading model: ' + err.message); return;
    }
  }
  if (!usedModel){ alert('No model available. Train or upload model files.'); return; }

  // build last-12-months sequence for selected group
  const group = selGroup.value;
  let year = Number(inpYear.value);
  let month = Number(inpMonth.value);
  if (!group){ alert('Choose a group'); return; }

  // pick series for group from combined monthly (UI view)
  const series = monthly.filter(d=>d.group===group).sort((a,b)=>a.month_index-b.month_index);
  if (series.length < 12){ alert('Not enough history for this group (need at least 12 months).'); return; }

  // find index for requested month/year in series
  const targetIndex = series.findIndex(s => s.year===year && s.month===month);

  // If requested month exists in data -> return actual value (fix predict past)
  let isActual = false;
  let predVal = null;
  let seqWindow = null;
  if (targetIndex >= 0){
    // return actual cancellation_rate for that month
    predVal = series[targetIndex].cancellation_rate ?? 0;
    isActual = true;
    // Build window for chart historical preceding 12 months (if available)
    if (targetIndex >= 12){
      seqWindow = series.slice(targetIndex-12, targetIndex);
    } else {
      seqWindow = series.slice(0,12);
    }
  } else {
    // Not in historical data -> compute using last available 12 months
    seqWindow = series.slice(-12);
    // compute predicted month as next after last data row
    const last = series[series.length-1];
    month = last.month + 1;
    year = last.year;
    if (month > 12){ month = 1; year += 1; }
    // set UI fields so user sees correct next-month
    inpYear.value = year;
    inpMonth.value = month;
  }

  // Apply interpolation strategy to seqWindow if chosen
  const interp = interpMethod.value;
  if (interp === 'ffill'){
    let lastCancel = seqWindow[0].cancellation_rate ?? 0;
    let lastPrice = seqWindow[0].avg_room_price ?? 0;
    let lastLead = seqWindow[0].lead_time_avg ?? 0;
    seqWindow = seqWindow.map(s => {
      s = {...s};
      if (s.cancellation_rate === null) s.cancellation_rate = lastCancel; else lastCancel = s.cancellation_rate;
      if (s.avg_room_price === null) s.avg_room_price = lastPrice; else lastPrice = s.avg_room_price;
      if (s.lead_time_avg === null) s.lead_time_avg = lastLead; else lastLead = s.lead_time_avg;
      return s;
    });
  } else if (interp === 'linear'){
    ['cancellation_rate','avg_room_price','lead_time_avg'].forEach(field=>{
      const vals = seqWindow.map(s => s[field] === null ? null : s[field]);
      let left=null;
      for (let i=0;i<vals.length;i++){ if (vals[i] === null) continue; left = i; break; }
      if (left === null){
        seqWindow.forEach(s => s[field] = 0);
      } else {
        for (let i=0;i<left;i++) seqWindow[i][field] = vals[left];
        let i = left;
        while (i < vals.length){
          if (seqWindow[i][field] !== null) { i++; continue;}
          let j = i+1;
          while (j<vals.length && seqWindow[j][field] === null) j++;
          const leftVal = seqWindow[i-1][field];
          const rightVal = (j < vals.length) ? seqWindow[j][field] : leftVal;
          const span = j - (i-1);
          for (let k = i; k < j; k++){
            const t = (k - (i-1))/span;
            seqWindow[k][field] = leftVal*(1-t) + rightVal*t;
          }
          i = j;
        }
      }
    });
  } else {
    seqWindow = seqWindow.map(s => ({
      ...s,
      cancellation_rate: s.cancellation_rate ?? 0,
      avg_room_price: s.avg_room_price ?? 0,
      lead_time_avg: s.lead_time_avg ?? 0
    }));
  }

  // apply smoothing slider (moving average) to cancellation_rate in the sequence
  const windowSize = Number(smoothingSlider.value) || 1;
  const cancSeries = seqWindow.map(s => s.cancellation_rate ?? 0);
  const smoothedCanc = cancSeries.map((v,i,arr)=>{
    const start = Math.max(0, i - windowSize + 1);
    const win = arr.slice(start, i+1);
    return win.reduce((a,b)=>a+b,0)/win.length;
  });

  // If actual value not returned earlier, build features & predict
  if (!isActual){
    const features = seqWindow.map((s, idx) => [ smoothedCanc[idx], s.avg_room_price ?? 0, s.lead_time_avg ?? 0 ]);
    // ensure scaler exists
    if (!scaler.min || scaler.min.length === 0){
      const allFeats = [];
      monthlyTrain.forEach(r => allFeats.push([ r.cancellation_rate ?? 0, r.avg_room_price ?? 0, r.lead_time_avg ?? 0 ]));
      computeScaler(allFeats);
    }
    const scaled = features.map(samp => scaleSample(samp));
    const xt = tf.tensor3d([scaled]);
    const ypred = await usedModel.predict(xt).array();
    predVal = ypred[0][0];
  }

  // Display prediction / actual
  if (isActual){
    predictResult.innerText = `Actual cancellation rate for ${month}/${year}: ${(predVal*100).toFixed(2)}% (data present in dataset)`;
  } else {
    predictResult.innerText = `Predicted cancellation rate for ${month}/${year}: ${(predVal*100).toFixed(2)}%`;
  }

  // insight logic (thresholds: >=0.5 high, >=0.2 medium, else low)
  const pv = predVal;
  if (pv >= 0.5) {
    insightBox.className = 'insight high'; insightBox.innerText = 'HIGH RISK — Recommend deposit / prepayment';
  } else if (pv >= 0.2) {
    insightBox.className = 'insight medium'; insightBox.innerText = 'MEDIUM RISK — Consider flexible policy / targeted retention';
  } else {
    insightBox.className = 'insight low'; insightBox.innerText = 'LOW RISK — Consider small price increase / promotions';
  }

  // update predict chart (historic + predicted as last point)
  const ctx = document.getElementById('chart-predict').getContext('2d');
  const xLabels = seqWindow.map(s => `${s.year}-${String(s.month).padStart(2,'0')}`);
  const histVals = seqWindow.map((s, i) => smoothedCanc[i]);
  const labels = [...xLabels, `${year}-${String(month).padStart(2,'0')}`];
  const dataHist = histVals.concat([predVal]);
  if (chartPredict) chartPredict.destroy();
  chartPredict = new Chart(ctx, {
    type:'line',
    data: {
      labels,
      datasets: [
        { label: `${group} cancellation (smoothed)`, data: dataHist, borderColor:'#01497c', fill:false, pointRadius:4, pointHoverRadius:6 },
      ]
    },
    options: {
      responsive:true,
      plugins:{
        tooltip:{ 
          enabled:true,
          callbacks:{
            label: function(context){
              const value = context.raw;
              const pct = (value === null) ? 'NaN' : (value*100).toFixed(2) + '%';
              let conf = 'Low';
              if (value >= 0.5) conf = 'High';
              else if (value >= 0.2) conf = 'Medium';
              return `Value: ${value===null?'NaN':value.toFixed(4)} (${pct}) — Confidence: ${conf}`;
            }
          }
        }
      }
    }
  });
});
