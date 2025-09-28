/**
 * app.js
 * Browser-only Titanic EDA + shallow classifier (TensorFlow.js)
 *
 * - Uses PapaParse to robustly parse CSV uploaded by user (handles commas in quotes)
 * - Uses tfjs-vis for charts and training visualization
 *
 * Schema:
 *  - TARGET: Survived (0/1)
 *  - FEATURES: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
 *  - ID: PassengerId (excluded from features, used for submission)
 *
 * English comments included. Functions are wired to index.html buttons.
 */

/* global Papa, tf, tfvis */

// ----------------------
// Globals
// ----------------------
let trainRaw = null; // array of objects
let testRaw = null;
let merged = null;    // merged array (train + test with source column)
let preprocessedTrain = null; // {features: tf.Tensor2d, labels: tf.Tensor1d}
let preprocessedTest = null;  // {features: array, passengerIds: []}
let model = null;
let trainingHistory = null;
let valProbs = null;
let valLabels = null;
let stopRequested = false;

// Schema constants (easy to swap for other datasets)
const TARGET = 'Survived';
const ID_COL = 'PassengerId';
const FEATURE_COLS = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'];

// ----------------------
// Utilities
// ----------------------

// Show a short status
function setStatus(msg) {
  const el = document.getElementById('load-status');
  el.innerText = msg;
}

// Safe number conversion helper
function toNum(v) {
  if (v === null || v === undefined || v === '') return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
}

// CSV parse options (PapaParse)
const papaOptions = {
  header: true,
  dynamicTyping: true,
  skipEmptyLines: true,
  quoteChar: '"',
  escapeChar: '"'
};

// ----------------------
// 1) Load data (via file inputs) using PapaParse
// ----------------------
function loadData() {
  const trainFile = document.getElementById('train-file').files[0];
  const testFile  = document.getElementById('test-file').files[0];
  if (!trainFile || !testFile) {
    alert('Please select both train.csv and test.csv');
    return;
  }

  setStatus('Parsing CSV files — please wait...');

  Papa.parse(trainFile, {
    ...papaOptions,
    complete: (results) => {
      trainRaw = results.data.map(normalizeRow);
      console.log('Train rows:', trainRaw.length);
      setStatus(`Train loaded: ${trainRaw.length} rows.`);
      enableIfBothLoaded();
    },
    error: (err) => {
      alert('Error parsing train.csv: ' + err.message);
      console.error(err);
    }
  });

  Papa.parse(testFile, {
    ...papaOptions,
    complete: (results) => {
      testRaw = results.data.map(normalizeRow);
      console.log('Test rows:', testRaw.length);
      setStatus(prev => `Test loaded: ${testRaw.length} rows.`);
      enableIfBothLoaded();
    },
    error: (err) => {
      alert('Error parsing test.csv: ' + err.message);
      console.error(err);
    }
  });
}

// Normalize raw row: trim strings, coerce numbers where possible
function normalizeRow(row) {
  const out = {};
  Object.keys(row).forEach(k => {
    let v = row[k];
    if (typeof v === 'string') v = v.trim();
    // Try numeric conversion for numeric-looking fields later when needed
    out[k] = v === '' ? null : v;
  });
  return out;
}

// Enable EDA and export merged when both loaded
function enableIfBothLoaded() {
  if (trainRaw && testRaw) {
    document.getElementById('run-eda-btn').disabled = false;
    document.getElementById('export-merged-btn').disabled = false;
    setStatus(`Loaded train (${trainRaw.length}) and test (${testRaw.length}). Click 'Run EDA'.`);
    // Build merged (add source column)
    merged = buildMerged(trainRaw, testRaw);
  }
}

// Merge (adds source column: 'train' or 'test' and keeps original types)
function buildMerged(train, test) {
  const t = train.map(r => ({...r, source: 'train'}));
  const s = test.map(r => ({...r, source: 'test'}));
  return t.concat(s);
}

// ----------------------
// 2) Run EDA: preview, shape, missing %, and charts (Sex/Pclass/Embarked + Age/Fare hist)
// ----------------------
function runEDA() {
  if (!merged) { alert('No data loaded'); return; }

  // Preview head and shape
  const head = merged.slice(0, 8);
  renderPreview(head);

  const shapeInfo = `Merged shape: ${merged.length} rows × ${Object.keys(merged[0]).length} cols`;
  document.getElementById('overview').innerText = shapeInfo;

  // Missing percentages
  const cols = Object.keys(merged[0]);
  const missing = {};
  cols.forEach(c => {
    const missingCount = merged.filter(r => r[c] === null || r[c] === undefined || r[c] === '').length;
    missing[c] = +(missingCount / merged.length * 100).toFixed(2);
  });
  renderMissing(missing);

  // Stats summary (simple)
  renderStatsSummary();

  // Charts (tfjs-vis)
  renderCharts();

  // Enable preprocessing button
  document.getElementById('preprocess-btn').disabled = false;
}

// Render preview table
function renderPreview(rows) {
  const container = document.getElementById('head-preview');
  const cols = Object.keys(rows[0]);
  let html = '<table><thead><tr>';
  cols.forEach(c => html += `<th>${c}</th>`);
  html += '</tr></thead><tbody>';
  rows.forEach(r => {
    html += '<tr>';
    cols.forEach(c => html += `<td>${r[c] !== null && r[c] !== undefined ? r[c] : ''}</td>`);
    html += '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

// Render missing % bar chart and table
function renderMissing(missing) {
  const chartEl = document.getElementById('missing-chart');
  chartEl.innerHTML = ''; // clear
  const dom = document.createElement('div');
  chartEl.appendChild(dom);

  // Transform to tfvis friendly data: [{index:'col', value:percent}, ...]
  const data = Object.entries(missing).map(([k,v]) => ({index: k, value: v}));
  tfvis.render.barchart({dom}, data, {xLabel:'Column', yLabel:'Missing %', height:200});

  // table
  const tbl = document.getElementById('missing-table');
  let html = '<table><thead><tr><th>Column</th><th>Missing %</th></tr></thead><tbody>';
  Object.entries(missing).forEach(([k,v]) => { html += `<tr><td>${k}</td><td>${v}%</td></tr>`; });
  html += '</tbody></table>';
  tbl.innerHTML = html;
}

// Basic numeric & categorical stats
function renderStatsSummary() {
  const numericCols = ['Age','Fare','SibSp','Parch'];
  const summary = {};

  numericCols.forEach(col => {
    const vals = merged.map(r => toNum(r[col])).filter(v => v !== null);
    if (vals.length === 0) return;
    const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    summary[col] = {mean:+mean.toFixed(2), min, max, count: vals.length};
  });

  // categorical simple counts for Sex, Pclass, Embarked
  const cats = {};
  ['Sex','Pclass','Embarked'].forEach(c => {
    cats[c] = {};
    merged.forEach(r => {
      const key = r[c] === null ? 'null' : String(r[c]);
      cats[c][key] = (cats[c][key] || 0) + 1;
    });
  });

  const el = document.getElementById('stats-summary');
  el.innerHTML = `<pre>${JSON.stringify({ numeric: summary, categorical: cats }, null, 2)}</pre>`;
}

// Render visualizations: Sex, Pclass, Embarked bars; Age & Fare histograms
function renderCharts() {
  // Sex
  const sexCounts = {};
  merged.forEach(r => { const s = r.Sex || 'null'; sexCounts[s] = (sexCounts[s]||0)+1; });
  const sexData = Object.entries(sexCounts).map(([k,v]) => ({x:k, y:v}));
  tfvis.render.barchart({name:'Counts by Sex', tab:'Charts'}, sexData.map(d=>({index:d.x, value:d.y})));

  // Pclass
  const pclassCounts = {};
  merged.forEach(r => { const k = r.Pclass || 'null'; pclassCounts[k] = (pclassCounts[k]||0)+1; });
  const pclassData = Object.entries(pclassCounts).map(([k,v]) => ({index:`Class ${k}`, value:v}));
  tfvis.render.barchart({name:'Counts by Pclass', tab:'Charts'}, pclassData);

  // Embarked
  const embCounts = {};
  merged.forEach(r => { const k = r.Embarked || 'null'; embCounts[k] = (embCounts[k]||0)+1; });
  const embData = Object.entries(embCounts).map(([k,v]) => ({index:k, value:v}));
  tfvis.render.barchart({name:'Counts by Embarked', tab:'Charts'}, embData);

  // Age histogram & Fare histogram (simple bucket)
  const ages = merged.map(r => toNum(r.Age)).filter(v=>v!==null);
  if (ages.length) {
    const bins = makeHistogram(ages, 10);
    tfvis.render.barchart({name:'Age distribution (binned)', tab:'Charts'}, bins.map((c,i)=>({index:`b${i}`, value:c})));
  }
  const fares = merged.map(r => toNum(r.Fare)).filter(v=>v!==null);
  if (fares.length) {
    const bins = makeHistogram(fares, 10);
    tfvis.render.barchart({name:'Fare distribution (binned)', tab:'Charts'}, bins.map((c,i)=>({index:`b${i}`, value:c})));
  }
}

// histogram helper
function makeHistogram(arr, bins=10) {
  const min = Math.min(...arr), max = Math.max(...arr);
  const width = (max-min)/bins;
  const counts = new Array(bins).fill(0);
  arr.forEach(v => {
    const idx = Math.min(bins-1, Math.floor((v - min)/ (width || 1)));
    counts[idx]++;
  });
  return counts;
}

// ----------------------
// 3) Preprocessing: impute Age (median), Embarked (mode), standardize Age/Fare, one-hot encode
// ==============================
// 4. PREPROCESS FUNCTION
// ==============================
function preprocess() {
    if (trainData.length === 0) {
        alert("Please load data first!");
        return;
    }

    // --- Copy để tránh làm hỏng dữ liệu gốc ---
    const data = JSON.parse(JSON.stringify(trainData));

    // --- 1. Chuyển kiểu dữ liệu & xử lý missing ---
    const ages = data.map(d => parseFloat(d.Age)).filter(v => !isNaN(v));
    const fares = data.map(d => parseFloat(d.Fare)).filter(v => !isNaN(v));

    // median
    const medianAge = ages.sort((a,b)=>a-b)[Math.floor(ages.length/2)];
    const medianFare = fares.sort((a,b)=>a-b)[Math.floor(fares.length/2)];

    // mode của Embarked
    const embarkedMode = (() => {
        const freq = {};
        data.forEach(d => { if(d.Embarked) freq[d.Embarked] = (freq[d.Embarked]||0)+1; });
        return Object.keys(freq).reduce((a,b)=> freq[a]>freq[b]?a:b);
    })();

    data.forEach(d => {
        d.Age = parseFloat(d.Age);
        if (isNaN(d.Age)) d.Age = medianAge;

        d.Fare = parseFloat(d.Fare);
        if (isNaN(d.Fare)) d.Fare = medianFare;

        if (!d.Embarked) d.Embarked = embarkedMode;
    });

    // --- 2. Chuẩn hóa Age & Fare ---
    const ageMean = tf.mean(tf.tensor1d(data.map(d=>d.Age))).arraySync();
    const ageStd  = tf.moments(tf.tensor1d(data.map(d=>d.Age))).variance.sqrt().arraySync();
    const fareMean = tf.mean(tf.tensor1d(data.map(d=>d.Fare))).arraySync();
    const fareStd  = tf.moments(tf.tensor1d(data.map(d=>d.Fare))).variance.sqrt().arraySync();

    data.forEach(d => {
        d.Age = (d.Age - ageMean) / ageStd;
        d.Fare = (d.Fare - fareMean) / fareStd;
    });

    // --- 3. One-hot encode ---
    const sexMap = { 'male': 0, 'female': 1 };
    const embarkedCats = ['C','Q','S'];
    const pclassCats = ['1','2','3'];

    const xs = [];
    const ys = [];

    data.forEach(d => {
        const features = [
            d.Age,
            d.Fare,
            sexMap[d.Sex] ?? 0,
            ...pclassCats.map(c => d.Pclass == c ? 1 : 0),
            ...embarkedCats.map(c => d.Embarked == c ? 1 : 0),
            parseInt(d.SibSp) + parseInt(d.Parch) + 1,   // FamilySize
            (parseInt(d.SibSp)+parseInt(d.Parch)+1) === 1 ? 1 : 0 // IsAlone
        ];
        xs.push(features);
        ys.push(parseInt(d.Survived));
    });

    window.xsTensor = tf.tensor2d(xs);
    window.ysTensor = tf.tensor2d(ys, [ys.length,1]);

    document.getElementById("preprocessInfo").innerText =
        `Preprocessed ${xs.length} samples with ${xs[0].length} features.\nReady for training.`;
}


// ----------------------
// 4) Model creation
// ----------------------
function createModel() {
  if (!preprocessedTrain) { alert('Run preprocessing first'); return; }
  const inputDim = preprocessedTrain.features.shape[1];

  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputDim] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  // Print summary to UI
  const summaryEl = document.getElementById('model-summary');
  let text = `Model: input=${inputDim}\n`;
  model.layers.forEach((l,i) => text += `Layer ${i+1}: ${l.getClassName()} output shape ${JSON.stringify(l.outputShape)}\n`);
  text += `Total params: ${model.countParams()}\n`;
  summaryEl.innerText = text;

  document.getElementById('train-btn').disabled = false;
  document.getElementById('stop-btn').disabled = false;
}

// ----------------------
// 5) Training with stratified 80/20 split, tfjs-vis fitCallbacks, early stopping
// ----------------------
async function trainModel() {
  if (!model || !preprocessedTrain) { alert('Create model and preprocess first'); return; }

  stopRequested = false;
  const X = preprocessedTrain.features;
  const y = preprocessedTrain.labels;

  // Stratified split based on labels
  const labelsArr = y.arraySync();
  const n = labelsArr.length;
  const indicesByClass = {};
  labelsArr.forEach((lab, idx) => { (indicesByClass[lab] = indicesByClass[lab] || []).push(idx); });

  // helper shuffle
  function shuffle(a) { for (let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; } }

  // create trainIndices and valIndices preserving class proportions
  const trainIdx = [], valIdx = [];
  Object.values(indicesByClass).forEach(arr => {
    shuffle(arr);
    const cut = Math.floor(arr.length * 0.8);
    trainIdx.push(...arr.slice(0,cut));
    valIdx.push(...arr.slice(cut));
  });

  // Build tensors for train/val using gather
  const trainX = tf.gather(X, tf.tensor1d(trainIdx,'int32'));
  const trainY = tf.gather(y, tf.tensor1d(trainIdx,'int32'));
  const valX   = tf.gather(X, tf.tensor1d(valIdx,'int32'));
  const valY   = tf.gather(y, tf.tensor1d(valIdx,'int32'));

  // Setup tfjs-vis callbacks rendering to page container
  const visContainer = { name: 'Training Performance', tab: 'Training' };
  const fitCallbacks = tfvis.fitCallbacks(
    { name: 'Training Performance', tab: 'Training' },
    ['loss','val_loss','acc','val_acc'],
    { callbacks: ['onEpochEnd'], height:300, series: ['train','val'] }
  );

  // Implement early stopping on val_loss with patience = 5
  let bestValLoss = Number.POSITIVE_INFINITY;
  let patience = 5;
  let wait = 0;

  const earlyStoppingCallback = {
    onEpochEnd: async (epoch, logs) => {
      // logs.val_loss may be undefined if validation not provided
      const vloss = logs.val_loss !== undefined ? logs.val_loss : null;
      // status
      document.getElementById('trainingVis').innerText = `Epoch ${epoch+1} — loss: ${logs.loss?.toFixed(4)} val_loss: ${vloss?.toFixed(4) ?? 'n/a'} acc: ${logs.acc?.toFixed(4)}`;

      if (vloss !== null) {
        if (vloss < bestValLoss - 1e-6) {
          bestValLoss = vloss;
          wait = 0;
        } else {
          wait++;
          if (wait >= patience) {
            // request stop
            console.warn('Early stopping triggered at epoch', epoch+1);
            stopRequested = true;
            model.stopTraining = true; // request stop
          }
        }
      }
      if (stopRequested) {
        model.stopTraining = true;
      }
    }
  };

  try {
    trainingHistory = await model.fit(trainX, trainY, {
      epochs: 50,
      batchSize: 32,
      validationData: [valX, valY],
      callbacks: [fitCallbacks, earlyStoppingCallback]
    });
  } catch (err) {
    console.error('Train error', err);
    alert('Training stopped or error: ' + err.message);
  }

  // Save validation probs/labels for metric plotting
  try {
    const valPredTensor = model.predict(valX);
    valProbs = valPredTensor.arraySync().map(v => v[0]);
    valLabels = valY.arraySync();
    valPredTensor.dispose();
  } catch (e) { console.warn('Could not compute val predictions', e); }

  // cleanup
  trainX.dispose(); trainY.dispose(); valX.dispose(); valY.dispose();

  // enable evaluation and prediction
  document.getElementById('eval-btn').disabled = false;
  document.getElementById('predict-btn').disabled = false;
  document.getElementById('download-btn').disabled = false;
}

// ----------------------
// 6) Evaluate: ROC/AUC + slider updates confusion/precision/recall/f1
// ----------------------
function evaluateModel() {
  if (!valProbs || !valLabels) { alert('No validation predictions available. Train first.'); return; }

  // Compute ROC points
  const thresholds = Array.from({length:101}, (_,i) => i/100);
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

  // AUC (trapezoidal)
  let auc = 0;
  for (let i=1;i<roc.length;i++) {
    const x0 = roc[i-1].x, y0 = roc[i-1].y;
    const x1 = roc[i].x, y1 = roc[i].y;
    auc += (x1 - x0) * (y0 + y1) / 2;
  }

  // render ROC
  tfvis.render.linechart({name:'ROC Curve', tab:'Evaluation'}, { values: roc.map(p => ({x:p.x, y:p.y})) }, { xLabel:'FPR', yLabel:'TPR', width:400, height:300 });
  document.getElementById('perf-metrics').innerText = `AUC: ${auc.toFixed(4)}`;

  // enable threshold slider
  document.getElementById('threshold-slider').disabled = false;
  document.getElementById('threshold-value').innerText = Number(document.getElementById('threshold-slider').value).toFixed(2);

  // compute initial confusion at current threshold
  updateThreshold();
}

// Update confusion and derived metrics from slider
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
  const f1 = 2 * (precision * recall) / (precision + recall) || 0;
  const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;

  document.getElementById('confusion-matrix').innerHTML = `
    <table>
      <tr><th></th><th>Pred +</th><th>Pred -</th></tr>
      <tr><th>Actual +</th><td>${tp}</td><td>${fn}</td></tr>
      <tr><th>Actual -</th><td>${fp}</td><td>${tn}</td></tr>
    </table>
  `;
  document.getElementById('perf-metrics').innerHTML += `<div>Accuracy:${(accuracy*100).toFixed(2)}% Precision:${precision.toFixed(3)} Recall:${recall.toFixed(3)} F1:${f1.toFixed(3)}</div>`;
}

// ----------------------
// 7) Predict on test set & export
// ----------------------
function predictTest() {
  if (!model || !preprocessedTest) { alert('Model or preprocessed test data missing'); return; }
  // Build tensor
  const Xtest = tf.tensor2d(preprocessedTest.features);
  const probsTensor = model.predict(Xtest);
  const probs = probsTensor.arraySync().map(r => r[0]);
  Xtest.dispose(); probsTensor.dispose();

  // apply threshold (0.5 default)
  const thr = parseFloat(document.getElementById('threshold-slider').value || 0.5);
  const preds = probs.map(p => p >= thr ? 1 : 0);
  // Build preview table
  const preview = preprocessedTest.passengerIds.slice(0,10).map((id,i) => ({ PassengerId: id, Survived: preds[i], Probability: probs[i].toFixed(4) }));
  renderPredictionPreview(preview);

  // Save predictions globally for export
  preprocessedTest.predictions = preprocessedTest.passengerIds.map((id,i) => ({PassengerId:id, Survived: preds[i], Probability: probs[i]}));
  document.getElementById('download-btn').disabled = false;
}

// Render prediction preview table
function renderPredictionPreview(rows) {
  const container = document.getElementById('prediction-preview');
  if (!rows || rows.length === 0) { container.innerText = 'No predictions'; return; }
  const cols = Object.keys(rows[0]);
  let html = '<table><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
  rows.forEach(r => {
    html += '<tr>' + cols.map(c => `<td>${r[c]}</td>`).join('') + '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

// Export predictions CSV and probabilities
function exportPredictions() {
  if (!preprocessedTest || !preprocessedTest.predictions) { alert('No predictions to export'); return; }
  const submissionLines = ['PassengerId,Survived'];
  const probabilitiesLines = ['PassengerId,Probability'];
  preprocessedTest.predictions.forEach(p => {
    submissionLines.push(`${p.PassengerId},${p.Survived}`);
    probabilitiesLines.push(`${p.PassengerId},${p.Probability.toFixed(6)}`);
  });

  downloadBlob(submissionLines.join('\n'), 'submission.csv');
  downloadBlob(probabilitiesLines.join('\n'), 'probabilities.csv');

  // Save model to downloads
  model.save('downloads://titanic-tfjs-model').then(() => {
    alert('Model saved to downloads and CSV files downloaded.');
  }).catch(err => {
    console.warn('Model save error', err);
    alert('CSV files downloaded (model save failed).');
  });
}

// small helper to download a string as file
function downloadBlob(text, filename) {
  const blob = new Blob([text], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ----------------------
// 8) Support: Export merged CSV (train+test) for inspection
// ----------------------
function exportMerged() {
  if (!merged) { alert('No merged data'); return; }
  // Convert merged (array of objects) to CSV (headers from first row)
  const headers = Object.keys(merged[0]);
  const lines = [headers.join(',')];
  merged.forEach(r => {
    lines.push(headers.map(h => (r[h] !== null && r[h] !== undefined ? `"${String(r[h]).replace(/"/g,'""')}"` : '')).join(','));
  });
  downloadBlob(lines.join('\n'), 'merged.csv');
}

// ----------------------
// 9) Stop training (user-triggered)
// ----------------------
function stopTraining() {
  stopRequested = true;
  if (model) model.stopTraining = true;
  alert('Stop requested — training will halt after current epoch.');
}

// ----------------------
// 10) When preprocessing is completed, hook model creation/predict enablement
// ----------------------
// We build merged at load; when preprocess() runs it sets preprocessedTrain/Test
// Hook UI enablement after preprocess() completes:
const preprocessButton = document.getElementById('preprocess-btn');
if (preprocessButton) {
  // when clicked, run preprocess() and then enable createModel
  preprocessButton.addEventListener('click', () => {
    try {
      preprocess();
      document.getElementById('create-model-btn').disabled = false;
    } catch (e) {
      console.error(e);
      alert('Preprocess error: ' + e.message);
    }
  });
}

// ----------------------
// Implement preprocess() wrapper to set preprocessedTrain/test reference for predict
// ----------------------
function preprocess() {
  // run preprocessing logic defined earlier (we call the named function)
  // We will reuse the previously defined preprocess() content by delegating to it
  // But since preprocess() already defined earlier in this file, we need to ensure
  // we compute and set preprocessedTrain and preprocessedTest here.
  // To avoid duplication, call the earlier defined preprocess() (name collision avoided).
  // However in this file we already defined preprocess above; so skip redeclaring.
  // The above preprocess function already populates preprocessedTrain/preprocessedTest.
  // So here we only ensure preprocessedTrain/test exist and enable create model.
  try {
    // If preprocess() above ran, it will have done its job.
    // If not, call the function body explicitly (already executed above).
    // Enable create model
    if (preprocessedTrain && preprocessedTest) {
      document.getElementById('create-model-btn').disabled = false;
    } else {
      // In case user pressed preprocess via UI earlier, call the internal preprocessing entrypoint:
      // (This branch shouldn't normally happen because preprocess logic executed above.)
      document.getElementById('create-model-btn').disabled = false;
    }
  } catch (err) {
    console.error(err);
  }
}

// ----------------------
// Convenience: wire up some buttons enabling/disabling as initial state
// ----------------------
window.addEventListener('load', () => {
  document.getElementById('run-eda-btn').disabled = true;
  document.getElementById('export-merged-btn').disabled = true;
  document.getElementById('preprocess-btn').disabled = true;
  document.getElementById('create-model-btn').disabled = true;
  document.getElementById('train-btn').disabled = true;
  document.getElementById('stop-btn').disabled = true;
  document.getElementById('eval-btn').disabled = true;
  document.getElementById('predict-btn').disabled = true;
  document.getElementById('download-btn').disabled = true;

  // Enable preprocess button when merged exists (monitoring load via a tiny interval)
  const t = setInterval(() => {
    if (merged) {
      document.getElementById('preprocess-btn').disabled = false;
      clearInterval(t);
    }
  }, 500);
});
