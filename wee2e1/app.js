/**
 * app.js
 * Browser-only Titanic EDA + shallow classifier (TensorFlow.js)
 *
 * - Uses PapaParse to parse user-uploaded CSVs
 * - Uses tfjs for model and tfjs-vis for visuals
 *
 * Schema:
 *  - TARGET: Survived (0/1)
 *  - FEATURES: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
 *  - ID: PassengerId (excluded from features, used for submission)
 *
 * NOTE: Swap schema constants below if you want to reuse for other datasets.
 */

/* global Papa, tf, tfvis */

// ---------------------- Globals ----------------------
let trainRaw = null;
let testRaw = null;
let merged = null;

let preprocessedTrain = null; // {features: tf.Tensor2d, labels: tf.Tensor1d}
let preprocessedTest = null;  // {features: Array<Array<number>>, passengerIds: Array}

let model = null;
let valProbs = null;
let valLabels = null;
let trainingHistory = null;
let stopRequested = false;

// Schema constants (easy to swap for other datasets)
const TARGET = 'Survived';
const ID_COL = 'PassengerId';
const FEATURE_COLS = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'];

// PapaParse options
const papaOptions = { header: true, dynamicTyping: false, skipEmptyLines: true, quoteChar: '"', escapeChar: '"' };

// ---------------------- Helpers ----------------------
function setStatus(msg) {
  const el = document.getElementById('load-status');
  if (el) el.innerText = msg;
}

function toNum(v) {
  if (v === null || v === undefined || v === '') return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
}

function median(arr) {
  const s = arr.slice().sort((a,b)=>a-b);
  const mid = Math.floor(s.length/2);
  return (s.length % 2 === 1) ? s[mid] : (s[mid-1] + s[mid]) / 2;
}

function mode(arr) {
  const freq = {};
  arr.forEach(v => { freq[v] = (freq[v] || 0) + 1; });
  let best = null, bestN = -1;
  Object.keys(freq).forEach(k => { if (freq[k] > bestN) { best = k; bestN = freq[k]; } });
  return best;
}

function mean(arr) { return arr.reduce((a,b)=>a+b,0)/arr.length; }
function std(arr, m) { return Math.sqrt(arr.reduce((s,v)=>s + Math.pow(v - m, 2), 0)/arr.length); }

// download helper
function downloadBlob(text, filename) {
  const blob = new Blob([text], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ---------------------- 1) Load data ----------------------
function loadData() {
  const trainFile = document.getElementById('train-file').files[0];
  const testFile = document.getElementById('test-file').files[0];
  if (!trainFile || !testFile) {
    alert('Please select both train.csv and test.csv files.');
    return;
  }

  setStatus('Parsing CSV files...');

  Papa.parse(trainFile, {
    ...papaOptions,
    complete: (results) => {
      trainRaw = results.data.map(normalizeRow);
      setStatus(`Train loaded: ${trainRaw.length} rows.`);
      tryBuildMerged();
    },
    error: (err) => { alert('Error parsing train.csv: ' + err.message); console.error(err); }
  });

  Papa.parse(testFile, {
    ...papaOptions,
    complete: (results) => {
      testRaw = results.data.map(normalizeRow);
      setStatus(`Test loaded: ${testRaw.length} rows.`);
      tryBuildMerged();
    },
    error: (err) => { alert('Error parsing test.csv: ' + err.message); console.error(err); }
  });
}

function normalizeRow(row) {
  // Trim whitespace and convert empty strings to null
  const out = {};
  Object.keys(row).forEach(k => {
    let v = row[k];
    if (typeof v === 'string') v = v.trim();
    out[k] = (v === '' ? null : v);
  });
  return out;
}

function tryBuildMerged() {
  if (trainRaw && testRaw) {
    // Add source column
    merged = trainRaw.map(r => ({...r, source:'train'})).concat(testRaw.map(r => ({...r, source:'test'})));
    document.getElementById('run-eda-btn').disabled = false;
    document.getElementById('export-merged-btn').disabled = false;
    document.getElementById('preprocess-btn').disabled = false;
    setStatus(`Loaded train (${trainRaw.length}) and test (${testRaw.length}). Click Run EDA.`);
  }
}

// ---------------------- 2) EDA / Inspect ----------------------
function runEDA() {
  if (!merged) { alert('No data loaded.'); return; }

  renderPreview(merged.slice(0,8));
  document.getElementById('overview').innerText = `Merged shape: ${merged.length} rows × ${Object.keys(merged[0]).length} cols`;

  // Missing %
  const cols = Object.keys(merged[0]);
  const missing = {};
  cols.forEach(c => {
    const missingCount = merged.filter(r => r[c] === null || r[c] === undefined || r[c] === '').length;
    missing[c] = +(missingCount / merged.length * 100).toFixed(2);
  });
  renderMissing(missing);
  renderStatsSummary();
  renderCharts();
}

function renderPreview(rows) {
  const container = document.getElementById('head-preview');
  if (!rows || rows.length === 0) { container.innerText = 'No preview'; return; }
  const cols = Object.keys(rows[0]);
  let html = '<table><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
  rows.forEach(r => {
    html += '<tr>' + cols.map(c => `<td>${r[c] !== null && r[c] !== undefined ? r[c] : ''}</td>`).join('') + '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function renderMissing(missing) {
  const chartEl = document.getElementById('missing-chart');
  chartEl.innerHTML = '';
  const dom = document.createElement('div');
  chartEl.appendChild(dom);
  const data = Object.entries(missing).map(([k,v]) => ({index:k, value:v}));
  tfvis.render.barchart({dom}, data, {xLabel:'Column', yLabel:'Missing %', height:200});

  const tbl = document.getElementById('missing-table');
  let html = '<table><thead><tr><th>Column</th><th>Missing %</th></tr></thead><tbody>';
  Object.entries(missing).forEach(([k,v]) => { html += `<tr><td>${k}</td><td>${v}%</td></tr>`; });
  html += '</tbody></table>';
  tbl.innerHTML = html;
}

function renderStatsSummary() {
  const numericCols = ['Age','Fare','SibSp','Parch'];
  const summary = {};
  numericCols.forEach(col => {
    const vals = merged.map(r => toNum(r[col])).filter(v=>v!==null);
    if (vals.length === 0) return;
    summary[col] = {mean:+mean(vals).toFixed(2), min:Math.min(...vals), max:Math.max(...vals), count:vals.length};
  });

  // categorical
  const cats = {};
  ['Sex','Pclass','Embarked'].forEach(c => {
    cats[c] = {};
    merged.forEach(r => {
      const key = (r[c] === null || r[c] === undefined) ? 'null' : String(r[c]);
      cats[c][key] = (cats[c][key] || 0) + 1;
    });
  });
  document.getElementById('stats-summary').innerHTML = `<pre>${JSON.stringify({numeric:summary,categorical:cats}, null, 2)}</pre>`;
}

function renderCharts() {
  // Survival by Sex & Pclass if Survived exists (train rows only)
  const trainRows = merged.filter(r => r.source === 'train');
  if (trainRows.length) {
    const bySex = {};
    trainRows.forEach(r => {
      const s = r.Sex || 'null';
      bySex[s] = bySex[s] || {survived:0, total:0};
      if (r[TARGET] !== null && r[TARGET] !== undefined) {
        if (Number(r[TARGET]) === 1) bySex[s].survived++;
      }
      bySex[s].total++;
    });
    const sexData = Object.keys(bySex).map(k => ({index:k, value: +(bySex[k].survived / bySex[k].total * 100).toFixed(2)}));
    tfvis.render.barchart({name:'Survival % by Sex', tab:'Charts'}, sexData);

    const byP = {};
    trainRows.forEach(r => {
      const p = r.Pclass || 'null';
      byP[p] = byP[p] || {survived:0, total:0};
      if (r[TARGET] !== null && r[TARGET] !== undefined) {
        if (Number(r[TARGET]) === 1) byP[p].survived++;
      }
      byP[p].total++;
    });
    const pclassData = Object.keys(byP).map(k => ({index:`Class ${k}`, value: +(byP[k].survived / byP[k].total * 100).toFixed(2)}));
    tfvis.render.barchart({name:'Survival % by Pclass', tab:'Charts'}, pclassData);
  }

  // Age histograms
  const ages = merged.map(r => toNum(r.Age)).filter(v=>v!==null);
  if (ages.length) {
    const bins = makeHistogram(ages, 10);
    tfvis.render.barchart({name:'Age distribution (binned)', tab:'Charts'}, bins.map((c,i)=>({index:`b${i}`, value:c})));
  }
  // Fare Histogram (linspace bins)
  const fares = merged.map(r => toNum(r.Fare)).filter(v => v !== null);
  if (fares.length) {
    const minFare = Math.min(...fares);
    const maxFare = Math.max(...fares);
  
    // số bins muốn hiển thị (có thể chỉnh 20, 30, 50...)
    const numBins = 20;  
  
    const step = (maxFare - minFare) / numBins;
    const counts = new Array(numBins).fill(0);
  
    fares.forEach(f => {
      let idx = Math.floor((f - minFare) / step);
      if (idx >= numBins) idx = numBins - 1; // tránh out of range
      counts[idx]++;
    });
  
    const bins = counts.map((c, i) => {
      const rangeStart = (minFare + i * step).toFixed(1);
      const rangeEnd = (minFare + (i + 1) * step).toFixed(1);
      return { index: `${rangeStart}-${rangeEnd}`, value: c };
    });
  
    tfvis.render.barchart(
      { name: 'Fare distribution (binned)', tab: 'Charts' },
      bins
    );
  }
}

function makeHistogram(arr, bins=10) {
  const min = Math.min(...arr), max = Math.max(...arr);
  const width = (max-min)/bins || 1;
  const counts = new Array(bins).fill(0);
  arr.forEach(v => {
    const idx = Math.min(bins-1, Math.floor((v - min)/width));
    counts[idx]++;
  });
  return counts;
}

// ---------------------- 3) Preprocessing ----------------------
function preprocess() {
  if (!merged) { alert('Run EDA / load data first'); return; }

  // Use train rows to compute imputation/statistics (so we don't leak test labels)
  const trainRows = merged.filter(r => r.source === 'train').map(r => ({...r}));
  if (trainRows.length === 0) { alert('No train rows found'); return; }

  // Compute medians / modes using train set
  const ages = trainRows.map(r => toNum(r.Age)).filter(v=>v!==null);
  const medianAge = ages.length ? median(ages) : 28;
  const fares = trainRows.map(r => toNum(r.Fare)).filter(v=>v!==null);
  const medianFare = fares.length ? median(fares) : 14.45;
  const embarkedMode = mode(trainRows.map(r => r.Embarked).filter(v=>v!=null)) || 'S';

  // Compute mean/std for standardization
  const meanAge = ages.length ? mean(ages) : 28;
  const stdAge = ages.length ? (std(ages, meanAge) || 1) : 1;
  const meanFare = fares.length ? mean(fares) : 14.45;
  const stdFare = fares.length ? (std(fares, meanFare) || 1) : 1;

  // categories for one-hot (fixed order to ensure consistent columns)
  const pclassCats = [1,2,3];
  const sexCats = ['male','female'];
  const embarkedCats = ['C','Q','S'];

  const oneHot = (val, cats) => cats.map(c => (val === c ? 1 : 0));

  // Build train features & labels
  const X = [];
  const Y = [];
  trainRows.forEach(r => {
    // impute
    const ageVal = toNum(r.Age) !== null ? toNum(r.Age) : medianAge;
    const fareVal = toNum(r.Fare) !== null ? toNum(r.Fare) : medianFare;
    const pclass = toNum(r.Pclass) || 3;
    const sex = (r.Sex && String(r.Sex).toLowerCase()) || 'male';
    const embarked = (r.Embarked || embarkedMode);

    const feat = [];
    // standardized Age, Fare
    feat.push((ageVal - meanAge) / stdAge);
    feat.push((fareVal - meanFare) / stdFare);
    // raw SibSp, Parch
    feat.push(toNum(r.SibSp) || 0);
    feat.push(toNum(r.Parch) || 0);
    // one-hot Pclass
    feat.push(...oneHot(pclass, pclassCats));
    // one-hot Sex
    feat.push(...oneHot(sex, sexCats));
    // one-hot Embarked
    feat.push(...oneHot(embarked, embarkedCats));

    // optional family features
    if (document.getElementById('family-toggle').checked) {
      const famSize = (toNum(r.SibSp) || 0) + (toNum(r.Parch) || 0) + 1;
      const isAlone = famSize === 1 ? 1 : 0;
      feat.push(famSize, isAlone);
    }

    X.push(feat);
    const lab = (r[TARGET] !== null && r[TARGET] !== undefined) ? Number(r[TARGET]) : 0;
    Y.push(lab);
  });

  // Build test features (same transforms, using train median/means)
  const testRows = merged.filter(r => r.source === 'test').map(r => ({...r}));
  const Xtest = [];
  const passengerIds = [];
  testRows.forEach(r => {
    const ageVal = toNum(r.Age) !== null ? toNum(r.Age) : medianAge;
    const fareVal = toNum(r.Fare) !== null ? toNum(r.Fare) : medianFare;
    const pclass = toNum(r.Pclass) || 3;
    const sex = (r.Sex && String(r.Sex).toLowerCase()) || 'male';
    const embarked = (r.Embarked || embarkedMode);

    const feat = [];
    feat.push((ageVal - meanAge) / stdAge);
    feat.push((fareVal - meanFare) / stdFare);
    feat.push(toNum(r.SibSp) || 0);
    feat.push(toNum(r.Parch) || 0);
    feat.push(...oneHot(pclass, pclassCats));
    feat.push(...oneHot(sex, sexCats));
    feat.push(...oneHot(embarked, embarkedCats));
    if (document.getElementById('family-toggle').checked) {
      const famSize = (toNum(r.SibSp) || 0) + (toNum(r.Parch) || 0) + 1;
      const isAlone = famSize === 1 ? 1 : 0;
      feat.push(famSize, isAlone);
    }
    Xtest.push(feat);
    passengerIds.push(r[ID_COL]);
  });

  // Save preprocessed structures
  preprocessedTrain = {
    features: tf.tensor2d(X),
    labels: tf.tensor1d(Y, 'int32')
  };
  preprocessedTest = {
    features: Xtest,
    passengerIds
  };

  document.getElementById('preprocess-info').innerText =
    `Preprocessing done. Train: ${preprocessedTrain.features.shape[0]} rows × ${preprocessedTrain.features.shape[1]} features. Test: ${preprocessedTest.features.length} rows × ${preprocessedTest.features[0] ? preprocessedTest.features[0].length : 0} features.`;

  // Enable model creation
  document.getElementById('create-model-btn').disabled = false;
}

// ---------------------- 4) Model creation ----------------------
function createModel() {
  if (!preprocessedTrain) { alert('Run preprocessing first'); return; }

  const inputDim = preprocessedTrain.features.shape[1];
  model = tf.sequential();
  model.add(tf.layers.dense({units:16, activation:'relu', inputShape:[inputDim]}));
  model.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  model.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});

  // Print summary
  let text = `Model: input=${inputDim}\n`;
  model.layers.forEach((l,i) => text += `Layer ${i+1}: ${l.getClassName()} output ${JSON.stringify(l.outputShape)}\n`);
  text += `Total params: ${model.countParams()}\n`;
  document.getElementById('model-summary').innerText = text;

  document.getElementById('train-btn').disabled = false;
  document.getElementById('stop-btn').disabled = false;
}

// ---------------------- 5) Training ----------------------
async function trainModel() {
  if (!model || !preprocessedTrain) { alert('Create model and preprocess first'); return; }

  stopRequested = false;
  const X = preprocessedTrain.features;
  const y = preprocessedTrain.labels;

  // Stratified 80/20 split by labels
  const labelsArr = await y.array();
  const indicesByClass = {};
  labelsArr.forEach((lab, idx) => { (indicesByClass[lab] = indicesByClass[lab] || []).push(idx); });

  function shuffle(arr) { for (let i=arr.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [arr[i],arr[j]]=[arr[j],arr[i]]; } }

  const trainIdx = [], valIdx = [];
  Object.values(indicesByClass).forEach(arr => {
    shuffle(arr);
    const cut = Math.floor(arr.length * 0.8);
    trainIdx.push(...arr.slice(0,cut));
    valIdx.push(...arr.slice(cut));
  });

  // Build train/val tensors via gather
  const trainX = tf.gather(X, tf.tensor1d(trainIdx,'int32'));
  const trainY = tf.gather(y, tf.tensor1d(trainIdx,'int32'));
  const valX = tf.gather(X, tf.tensor1d(valIdx,'int32'));
  const valY = tf.gather(y, tf.tensor1d(valIdx,'int32'));

  // vis callbacks
  const fitCallbacks = tfvis.show.fitCallbacks(
    { name: 'Training Performance', tab: 'Training' },
    ['loss','val_loss','acc','val_acc'],
    { callbacks: ['onEpochEnd'], height:300 }
  );

  // Early stopping on val_loss with patience=5
  let bestVal = Number.POSITIVE_INFINITY, wait = 0, patience = 5;

  const earlyStoppingCb = {
    onEpochEnd: async (epoch, logs) => {
      const vloss = logs.val_loss !== undefined ? logs.val_loss : null;
      document.getElementById('trainingVis').innerText = `Epoch ${epoch+1} — loss: ${logs.loss?.toFixed(4)} val_loss: ${vloss?.toFixed(4) ?? 'n/a'} acc: ${logs.acc?.toFixed(4)}`;
      if (vloss !== null) {
        if (vloss < bestVal - 1e-6) { bestVal = vloss; wait = 0; }
        else { wait++; if (wait >= patience) { stopRequested = true; model.stopTraining = true; } }
      }
      if (stopRequested) model.stopTraining = true;
    }
  };

  try {
    trainingHistory = await model.fit(trainX, trainY, {
      epochs: 50,
      batchSize: 32,
      validationData: [valX, valY],
      callbacks: [fitCallbacks, earlyStoppingCb]
    });
  } catch (err) {
    console.error('Training error', err);
    alert('Training stopped or error: ' + err.message);
  }

  // Save validation probs & labels
  try {
    const valPred = model.predict(valX);
    const arr = await valPred.array();
    valProbs = arr.map(r => (Array.isArray(r) ? r[0] : r));
    valLabels = await valY.array();
    valPred.dispose();
  } catch (e) {
    console.warn('Could not compute val predictions', e);
  }

  // dispose
  trainX.dispose(); trainY.dispose(); valX.dispose(); valY.dispose();

  document.getElementById('eval-btn').disabled = false;
  document.getElementById('predict-btn').disabled = false;
  document.getElementById('download-btn').disabled = false;
}

// ---------------------- 6) Metrics (ROC/AUC + slider) ----------------------
function evaluateModel() {
  if (!valProbs || !valLabels) { alert('No validation predictions available (train first)'); return; }

  const thresholds = Array.from({length:101}, (_,i)=>i/100);
  const roc = thresholds.map(t => {
    let tp=0, fp=0, tn=0, fn=0;
    valProbs.forEach((p,i) => {
      const pred = p >= t ? 1 : 0;
      const act = valLabels[i];
      if (pred===1 && act===1) tp++;
      else if (pred===1 && act===0) fp++;
      else if (pred===0 && act===0) tn++;
      else fn++;
    });
    const tpr = tp / (tp + fn) || 0;
    const fpr = fp / (fp + tn) || 0;
    return {x: fpr, y: tpr};
  });

  // AUC (trapezoid)
  let auc = 0;
  for (let i=1;i<roc.length;i++) {
    const x0 = roc[i-1].x, y0 = roc[i-1].y;
    const x1 = roc[i].x, y1 = roc[i].y;
    auc += (x1 - x0) * (y0 + y1) / 2;
  }

  // Render ROC (into roc-container)
  const dom = document.getElementById('roc-container');
  dom.innerHTML = '';
  tfvis.render.linechart({dom}, { values: roc.map(p => ({x:p.x, y:p.y})) }, { xLabel:'FPR', yLabel:'TPR', width:400, height:300 });
  document.getElementById('perf-metrics').innerText = `AUC: ${auc.toFixed(4)}`;

  // enable slider
  document.getElementById('threshold-slider').disabled = false;
  document.getElementById('threshold-value').innerText = Number(document.getElementById('threshold-slider').value).toFixed(2);
  updateThreshold();
}

function updateThreshold() {
  const thr = parseFloat(document.getElementById('threshold-slider').value);
  document.getElementById('threshold-value').innerText = thr.toFixed(2);

  if (!valProbs || !valLabels) { document.getElementById('confusion-matrix').innerText = 'No validation predictions'; return; }

  let tp=0, fp=0, tn=0, fn=0;
  valProbs.forEach((p,i) => {
    const pred = p >= thr ? 1 : 0;
    const act = valLabels[i];
    if (pred===1 && act===1) tp++;
    else if (pred===1 && act===0) fp++;
    else if (pred===0 && act===0) tn++;
    else fn++;
  });

  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = (precision + recall) ? 2 * (precision * recall) / (precision + recall) : 0;
  const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;

  document.getElementById('confusion-matrix').innerHTML = `
    <table>
      <tr><th></th><th>Pred +</th><th>Pred -</th></tr>
      <tr><th>Actual +</th><td>${tp}</td><td>${fn}</td></tr>
      <tr><th>Actual -</th><td>${fp}</td><td>${tn}</td></tr>
    </table>
  `;
  document.getElementById('perf-metrics').innerHTML = `AUC: ${document.getElementById('perf-metrics').innerText.split('AUC: ')[1] || 'n/a'}<div>Accuracy:${(accuracy*100).toFixed(2)}% Precision:${precision.toFixed(3)} Recall:${recall.toFixed(3)} F1:${f1.toFixed(3)}</div>`;
}

// ---------------------- 7) Predict & Export ----------------------
function predictTest() {
  if (!model || !preprocessedTest) { 
    alert('Model or preprocessed test data missing'); 
    return; 
  }
  if (!preprocessedTest.features || preprocessedTest.features.length === 0) { 
    alert('Preprocessed test empty'); 
    return; 
  }

  const Xtest = tf.tensor2d(preprocessedTest.features);
  const probsTensor = model.predict(Xtest);
  probsTensor.array().then(arr => {
    const probs = arr.map(r => Array.isArray(r) ? r[0] : r);
    Xtest.dispose(); 
    probsTensor.dispose();

    const thr = parseFloat(document.getElementById('threshold-slider').value || 0.5);
    const preds = probs.map(p => p >= thr ? 1 : 0);

    // preview (first 10)
    const preview = preprocessedTest.passengerIds
      .slice(0, 10)
      .map((id,i) => ({ 
        PassengerId: id, 
        Survived: preds[i], 
        Probability: probs[i].toFixed(4) 
      }));
    renderPredictionPreview(preview);

    // store predictions for export
    preprocessedTest.predictions = preprocessedTest.passengerIds.map((id,i) => ({
      PassengerId:id, 
      Survived: preds[i], 
      Probability: probs[i]
    }));

    document.getElementById('download-btn').disabled = false;
  }).catch(err => {
    console.error(err);
    alert('Prediction error: ' + err.message);
  });
}

function renderPredictionPreview(rows) {
  const container = document.getElementById('prediction-preview');
  if (!rows || rows.length === 0) { 
    container.innerText = 'No predictions'; 
    return; 
  }
  const cols = Object.keys(rows[0]);
  let html = '<table><thead><tr>' 
    + cols.map(c => `<th>${c}</th>`).join('') 
    + '</tr></thead><tbody>';
  rows.forEach(r => { 
    html += '<tr>' + cols.map(c => `<td>${r[c]}</td>`).join('') + '</tr>'; 
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function exportPredictions() {
  if (!preprocessedTest || !preprocessedTest.predictions) { 
    alert('No predictions to export'); 
    return; 
  }

  const submissionLines = ['PassengerId,Survived'];
  const probabilitiesLines = ['PassengerId,Probability'];

  preprocessedTest.predictions.forEach(p => {
    submissionLines.push(`${p.PassengerId},${p.Survived}`);
    probabilitiesLines.push(`${p.PassengerId},${p.Probability.toFixed(6)}`);
  });

  downloadBlob(submissionLines.join('\n'), 'submission.csv');
  downloadBlob(probabilitiesLines.join('\n'), 'probabilities.csv');

  // Save model to downloads (tfjs browser save)
  model.save('downloads://titanic-tfjs-model').then(() => {
    alert('CSV files downloaded and model saved to downloads.');
  }).catch(err => {
    console.warn('Model save failed', err);
    alert('CSV files downloaded (model save failed).');
  });
}

// ---------------------- 8) Export prediction + model ----------------------

// ---------------------- Export Predictions + Model ----------------------
function exportPredictionsWithModel() {
  if (!preprocessedTest || !preprocessedTest.predictions) { 
    alert('No predictions to export'); 
    return; 
  }

  const combinedLines = ['PassengerId,Survived,Probability'];
  preprocessedTest.predictions.forEach(p => {
    combinedLines.push(`${p.PassengerId},${p.Survived},${p.Probability.toFixed(6)}`);
  });

  downloadBlob(combinedLines.join('\n'), 'predictions.csv');

  model.save('downloads://titanic-tfjs-model').then(() => {
    alert('Predictions CSV + model files downloaded.');
  }).catch(err => {
    console.warn('Model save failed', err);
    alert('Predictions CSV downloaded (model save failed).');
  });
}

// ---------------------- Utility: downloadBlob ----------------------
function downloadBlob(content, filename) {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------------------- 9) Stop training ----------------------
function stopTraining() {
  stopRequested = true;
  if (model) model.stopTraining = true;
  alert('Stop requested — training will halt after current epoch.');
}

// ---------------------- UI init ----------------------
window.addEventListener('load', () => {
  // Set initial disabled states (index.html may already set them)
  const ids = ['run-eda-btn','export-merged-btn','preprocess-btn','create-model-btn','train-btn','stop-btn','eval-btn','conf-btn','predict-btn','download-btn','threshold-slider'];
  ids.forEach(id => { const el = document.getElementById(id); if (el) el.disabled = true; });
  // load button is enabled in index.html
  // small interval to enable preprocess when merged built
  setInterval(() => {
    if (merged) {
      const el = document.getElementById('preprocess-btn');
      if (el) el.disabled = false;
    }
  }, 500);
});
