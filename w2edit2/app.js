/* app.js
   Titanic binary classifier (browser-only) using TensorFlow.js and tfjs-vis.
   - Fixes CSV comma/quote parsing (handles names with commas)
   - Stratified 80/20 split
   - EarlyStopping on val_loss (patience=5)
   - ROC + slider-driven confusion matrix & metrics
   - Predict & export submission/probabilities, save model to downloads
*/

/* -------------------------
   CONFIG / SCHEMA (changeable)
   -------------------------
   TARGET_FEATURE: the binary label column (0 or 1)
   ID_FEATURE: identifier column to include in exports but exclude from features
   FEATURES: features to use (numerical + categorical)
   To adapt to another dataset: update these constants.
*/
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];

// GLOBAL STATE
let trainData = null;
let testData = null;

let preprocessedTrain = null; // {features: tf.Tensor2d, labels: tf.Tensor1d}
let preprocessedTest = null;  // {featuresArray: Array, passengerIds: Array}

let model = null;
let valProbs = null;   // Array of probabilities on validation set
let valLabels = null;  // Array of labels on validation set
let testPredictionsTensor = null; // tf.Tensor of probabilities for test set

/* -------------------------
   Utils: CSV Parser (handles quoted fields and commas inside quotes)
   - RFC4180-style basic parser
   - Returns array of objects (first row = headers)
   ------------------------- */
function parseCSVWithQuotes(text) {
  // Normalize line endings
  text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

  const rows = [];
  let cur = '';
  let row = [];
  let i = 0;
  let inQuotes = false;

  while (i < text.length) {
    const ch = text[i];

    if (inQuotes) {
      if (ch === '"') {
        // Peek next char: if also a quote -> escaped quote
        if (i + 1 < text.length && text[i + 1] === '"') {
          cur += '"';
          i += 2;
          continue;
        } else {
          inQuotes = false;
          i++;
          continue;
        }
      } else {
        cur += ch;
        i++;
        continue;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
        i++;
        continue;
      }
      if (ch === ',') {
        row.push(cur);
        cur = '';
        i++;
        continue;
      }
      if (ch === '\n') {
        row.push(cur);
        rows.push(row);
        row = [];
        cur = '';
        i++;
        continue;
      }
      // normal char
      cur += ch;
      i++;
    }
  }

  // push last value if needed
  if (cur !== '' || inQuotes || row.length > 0) {
    row.push(cur);
    rows.push(row);
  }

  // Remove any trailing empty rows
  while (rows.length && rows[rows.length - 1].length === 1 && rows[rows.length - 1][0] === '') {
    rows.pop();
  }

  if (rows.length === 0) return [];

  const headers = rows[0].map(h => h.trim());
  const data = [];
  for (let r = 1; r < rows.length; r++) {
    const cols = rows[r];
    // If row has fewer columns, pad with empty strings
    while (cols.length < headers.length) cols.push('');
    const obj = {};
    for (let c = 0; c < headers.length; c++) {
      let val = cols[c];
      // Convert empty string -> null for missing
      if (val === '') val = null;
      // Convert numeric-like strings to numbers when safe
      // (avoid converting fields like 'male' or 'C' or names)
      if (val !== null && !isNaN(val) && val.trim() !== '') {
        // parseFloat is safe: will convert "5" and "5.0"
        val = parseFloat(val);
      }
      obj[headers[c]] = val;
    }
    data.push(obj);
  }

  return data;
}

/* -------------------------
   File reading helpers
*/
function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = e => reject(new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

/* -------------------------
   DOM Shortcuts / UI wires
*/
document.addEventListener('DOMContentLoaded', () => {
  // Buttons and inputs
  const loadBtn = document.getElementById('load-data-btn');
  const inspectBtn = document.getElementById('inspect-btn');
  const preprocessBtn = document.getElementById('preprocess-btn');
  const createModelBtn = document.getElementById('create-model-btn');
  const trainBtn = document.getElementById('train-btn');
  const predictBtn = document.getElementById('predict-btn');
  const exportBtn = document.getElementById('export-btn');
  const thresholdSlider = document.getElementById('threshold-slider');

  loadBtn.addEventListener('click', loadData);
  inspectBtn.addEventListener('click', inspectData);
  preprocessBtn.addEventListener('click', preprocessData);
  createModelBtn.addEventListener('click', createModel);
  trainBtn.addEventListener('click', trainModel);
  predictBtn.addEventListener('click', predict);
  exportBtn.addEventListener('click', exportResults);
  thresholdSlider.addEventListener('input', updateMetricsFromSlider);
});

/* -------------------------
   Load data (file inputs)
   - Uses parseCSVWithQuotes to avoid comma-in-name issues
*/
async function loadData() {
  const statusDiv = document.getElementById('data-status');
  statusDiv.textContent = 'Loading files...';

  const trainFile = document.getElementById('train-file').files[0];
  const testFile = document.getElementById('test-file').files[0];

  if (!trainFile || !testFile) {
    alert('Please select both train.csv and test.csv files.');
    statusDiv.textContent = 'No file(s) selected.';
    return;
  }

  try {
    const [trainText, testText] = await Promise.all([
      readFileAsText(trainFile),
      readFileAsText(testFile)
    ]);

    trainData = parseCSVWithQuotes(trainText);
    testData = parseCSVWithQuotes(testText);

    if (!trainData.length) throw new Error('Parsed training data is empty.');
    if (!testData.length) throw new Error('Parsed test data is empty.');

    statusDiv.textContent = `Loaded: ${trainData.length} train rows, ${testData.length} test rows.`;
    document.getElementById('inspect-btn').disabled = false;
    // reset downstream state
    resetStateAfterLoad();
  } catch (err) {
    console.error(err);
    statusDiv.textContent = `Error loading files: ${err.message}`;
    alert('Error reading CSVs: ' + err.message);
  }
}

function resetStateAfterLoad() {
  preprocessedTrain = null;
  preprocessedTest = null;
  model = null;
  valProbs = null;
  valLabels = null;
  testPredictionsTensor = null;

  document.getElementById('preprocess-btn').disabled = false;
  document.getElementById('create-model-btn').disabled = true;
  document.getElementById('train-btn').disabled = true;
  document.getElementById('predict-btn').disabled = true;
  document.getElementById('export-btn').disabled = true;
  document.getElementById('threshold-slider').disabled = true;
  document.getElementById('inspect-btn').disabled = false;
  document.getElementById('model-summary').innerHTML = '';
  document.getElementById('prediction-output').innerHTML = '';
  document.getElementById('preprocessing-output').innerHTML = '';
  document.getElementById('data-preview').innerHTML = '';
  document.getElementById('data-stats').innerHTML = '';
  document.getElementById('confusion-matrix').innerHTML = 'No evaluation yet.';
  document.getElementById('performance-metrics').innerHTML = 'No evaluation yet.';
}

/* -------------------------
   Inspect data: preview, stats, and charts
*/
function inspectData() {
  if (!trainData || trainData.length === 0) {
    alert('Please load data first.');
    return;
  }

  // Preview first 8 rows
  const previewDiv = document.getElementById('data-preview');
  previewDiv.innerHTML = '<h4>Data preview (first 8 rows)</h4>';
  previewDiv.appendChild(createPreviewTable(trainData.slice(0, 8)));

  // Stats: shape, missing %
  const statsDiv = document.getElementById('data-stats');
  const cols = Object.keys(trainData[0]);
  const shapeText = `Shape: ${trainData.length} rows x ${cols.length} columns`;
  // survival rate if available
  let targetText = 'Target column not found in train file.';
  if (trainData[0].hasOwnProperty(TARGET_FEATURE)) {
    const surviveCount = trainData.filter(r => r[TARGET_FEATURE] === 1).length;
    const rate = ((surviveCount / trainData.length) * 100).toFixed(2);
    targetText = `Survived: ${surviveCount}/${trainData.length} (${rate}%)`;
  }

  // missing values
  let missingHtml = '<h4>Missing % by column</h4><ul>';
  cols.forEach(c => {
    const missing = trainData.filter(r => r[c] === null || r[c] === undefined).length;
    const pct = ((missing / trainData.length) * 100).toFixed(2);
    missingHtml += `<li>${c}: ${pct}%</li>`;
  });
  missingHtml += '</ul>';

  statsDiv.innerHTML = `<p>${shapeText}</p><p>${targetText}</p>${missingHtml}`;

  // Charts: Survival by Sex and by Pclass using tfjs-vis
  createVisualizations();
}

/* Create a small HTML table for preview */
function createPreviewTable(rows) {
  const table = document.createElement('table');
  if (!rows || rows.length === 0) {
    const p = document.createElement('div');
    p.textContent = 'No rows to preview.';
    return p;
  }
  const headers = Object.keys(rows[0]);
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  headers.forEach(h => {
    const th = document.createElement('th');
    th.textContent = h;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach(r => {
    const tr = document.createElement('tr');
    headers.forEach(h => {
      const td = document.createElement('td');
      let v = r[h];
      if (v === null || v === undefined) v = 'NULL';
      td.textContent = v;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

/* Visualizations using tfjs-vis */
function createVisualizations() {
  // Survival by Sex
  const bySex = {};
  trainData.forEach(r => {
    const sex = r.Sex || 'unknown';
    if (!bySex[sex]) bySex[sex] = { survived: 0, total: 0 };
    if (r.Survived === 1) bySex[sex].survived++;
    bySex[sex].total++;
  });
  const sexData = Object.entries(bySex).map(([sex, s]) => ({ x: sex, y: (s.survived / s.total) * 100 }));
  tfvis.render.barchart({ name: 'Survival Rate by Sex', tab: 'Charts' }, sexData, { xLabel: 'Sex', yLabel: 'Survival %' });

  // Survival by Pclass
  const byP = {};
  trainData.forEach(r => {
    const p = r.Pclass || 'unknown';
    if (!byP[p]) byP[p] = { survived: 0, total: 0 };
    if (r.Survived === 1) byP[p].survived++;
    byP[p].total++;
  });
  const pclassData = Object.entries(byP).map(([p, s]) => ({ x: `Class ${p}`, y: (s.survived / s.total) * 100 }));
  tfvis.render.barchart({ name: 'Survival Rate by Pclass', tab: 'Charts' }, pclassData, { xLabel: 'Pclass', yLabel: 'Survival %' });

  // Inform the user
  const chartsDiv = document.getElementById('charts');
  chartsDiv.innerHTML = '<p class="small">Charts shown in tfjs-vis (open visor bottom-right).</p>';
}

/* -------------------------
   Preprocessing
   - Impute Age with median (train)
   - Impute Embarked with mode (train)
   - Standardize Age and Fare (train statistics)
   - One-hot encode Sex (male/female), Pclass (1/2/3), Embarked (C/Q/S)
   - Optional family features: FamilySize and IsAlone
*/
function calculateMedian(values) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a,b)=>a-b);
  const half = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) return (sorted[half-1] + sorted[half]) / 2;
  return sorted[half];
}
function calculateMode(values) {
  if (!values.length) return null;
  const freq = {};
  let maxCount = 0;
  let mode = null;
  values.forEach(v => {
    freq[v] = (freq[v] || 0) + 1;
    if (freq[v] > maxCount) { maxCount = freq[v]; mode = v; }
  });
  return mode;
}
function calculateStd(values) {
  if (!values.length) return 1;
  const mean = values.reduce((s,v)=>s+v,0)/values.length;
  const variance = values.reduce((s,v)=>s + Math.pow(v-mean,2),0)/values.length;
  return Math.sqrt(variance);
}

function oneHot(value, categories) {
  return categories.map(c => (value === c ? 1 : 0));
}

function preprocessData() {
  if (!trainData || !testData) {
    alert('Load train and test files first.');
    return;
  }

  document.getElementById('preprocessing-output').textContent = 'Preprocessing...';

  // Compute imputation stats from training data
  const ages = trainData.map(r => r.Age).filter(v => v !== null && v !== undefined && !isNaN(v));
  const fares = trainData.map(r => r.Fare).filter(v => v !== null && v !== undefined && !isNaN(v));
  const embarkedVals = trainData.map(r => r.Embarked).filter(v => v !== null && v !== undefined);

  const ageMedian = calculateMedian(ages);
  const fareMedian = calculateMedian(fares);
  const embarkedMode = calculateMode(embarkedVals);
  const ageStd = calculateStd(ages) || 1;
  const fareStd = calculateStd(fares) || 1;

  // helper to map row -> feature array
  function rowToFeatures(row) {
    // impute
    const age = (row.Age === null || row.Age === undefined || isNaN(row.Age)) ? ageMedian : row.Age;
    const fare = (row.Fare === null || row.Fare === undefined || isNaN(row.Fare)) ? fareMedian : row.Fare;
    const embarked = (row.Embarked === null || row.Embarked === undefined) ? embarkedMode : row.Embarked;
    const pclass = row.Pclass || 3;
    const sex = row.Sex || 'male';

    // standardized numerics
    const sAge = (age - ageMedian) / ageStd;
    const sFare = (fare - fareMedian) / fareStd;
    const sibsp = (row.SibSp === null || row.SibSp === undefined) ? 0 : row.SibSp;
    const parch = (row.Parch === null || row.Parch === undefined) ? 0 : row.Parch;

    let features = [sAge, sFare, sibsp, parch];
    // one-hot Pclass (1,2,3)
    features = features.concat(oneHot(pclass, [1,2,3]));
    // one-hot Sex (male,female)
    features = features.concat(oneHot(sex, ['male', 'female']));
    // one-hot Embarked (C,Q,S)
    features = features.concat(oneHot(embarked, ['C','Q','S']));

    if (document.getElementById('add-family-features').checked) {
      const familySize = sibsp + parch + 1;
      const isAlone = familySize === 1 ? 1 : 0;
      features.push(familySize, isAlone);
    }
    return features;
  }

  // Build training feature arrays and labels
  const featuresArr = [];
  const labelsArr = [];
  for (const r of trainData) {
    // skip rows missing target
    if (r[TARGET_FEATURE] === null || r[TARGET_FEATURE] === undefined) continue;
    featuresArr.push(rowToFeatures(r));
    // ensure label is 0 or 1 numeric
    labelsArr.push(Number(r[TARGET_FEATURE]));
  }

  // Build test features and passenger IDs
  const testFeaturesArr = [];
  const passengerIds = [];
  for (const r of testData) {
    testFeaturesArr.push(rowToFeatures(r));
    passengerIds.push(r[ID_FEATURE] || null);
  }

  // Convert to tensors
  const featureDim = featuresArr[0].length;
  preprocessedTrain = {
    features: tf.tensor2d(featuresArr),
    labels: tf.tensor1d(labelsArr, 'int32'),
    featureDim
  };

  preprocessedTest = {
    featuresArray: testFeaturesArr,
    passengerIds
  };

  // Report shapes
  document.getElementById('preprocessing-output').innerHTML = `
    Preprocessing complete.<br/>
    Training features shape: [${preprocessedTrain.features.shape}]<br/>
    Training labels shape: [${preprocessedTrain.labels.shape}]<br/>
    Test features shape: [${preprocessedTest.featuresArray.length}, ${featureDim}]<br/>
  `;

  // Enable model creation
  document.getElementById('create-model-btn').disabled = false;
}

/* -------------------------
   Model creation (simple architecture)
   - Dense(16, relu) -> Dense(1, sigmoid)
*/
function createModel() {
  if (!preprocessedTrain) {
    alert('Preprocess data first.');
    return;
  }

  const inputShape = [preprocessedTrain.featureDim];

  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  // Model summary (simple textual)
  const summaryDiv = document.getElementById('model-summary');
  let summaryHtml = '<h4>Model Summary</h4>';
  summaryHtml += `<p>Input shape: [${inputShape}]</p>`;
  model.layers.forEach((layer, i) => {
    summaryHtml += `<div>Layer ${i+1}: ${layer.getClassName()} — output shape: ${JSON.stringify(layer.outputShape)}</div>`;
  });
  summaryHtml += `<div>Total params: ${model.countParams()}</div>`;
  summaryDiv.innerHTML = summaryHtml;

  // enable train button
  document.getElementById('train-btn').disabled = false;
}

/* -------------------------
   Stratified split function (ensures label distribution preserved)
   returns {trainIdxs, valIdxs}
*/
function stratifiedSplit(labels, valFraction=0.2, seed=42) {
  // labels: Array of 0/1
  const idxByLabel = {};
  labels.forEach((lab, idx) => {
    idxByLabel[lab] = idxByLabel[lab] || [];
    idxByLabel[lab].push(idx);
  });

  // simple deterministic shuffle using seed
  function seededShuffle(array, seed) {
    const a = array.slice();
    let s = seed;
    for (let i = a.length - 1; i > 0; i--) {
      s = (s * 9301 + 49297) % 233280;
      const r = s / 233280;
      const j = Math.floor(r * (i + 1));
      const tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
    return a;
  }

  const trainIdxs = [];
  const valIdxs = [];

  Object.keys(idxByLabel).forEach(l => {
    const arr = seededShuffle(idxByLabel[l], seed);
    const valCount = Math.max(1, Math.floor(arr.length * valFraction));
    valIdxs.push(...arr.slice(0, valCount));
    trainIdxs.push(...arr.slice(valCount));
  });

  return { trainIdxs, valIdxs };
}

/* -------------------------
   Training:
   - 80/20 stratified split
   - 50 epochs, batch 32
   - tfjs-vis fitCallbacks for live plots
   - EarlyStopping on val_loss (patience 5)
*/
async function trainModel() {
  if (!model || !preprocessedTrain) {
    alert('Create model and preprocess data first.');
    return;
  }

  document.getElementById('training-status').textContent = 'Training...';

  const X = preprocessedTrain.features;
  const y = preprocessedTrain.labels;

  // get labels to JS array
  const labelsArray = Array.from(y.dataSync());

  const { trainIdxs, valIdxs } = stratifiedSplit(labelsArray, 0.2, 123);

  // build tensors by indexing (gather)
  const xTrain = tf.gather(X, tf.tensor1d(trainIdxs, 'int32'));
  const yTrain = tf.gather(y, tf.tensor1d(trainIdxs, 'int32'));
  const xVal = tf.gather(X, tf.tensor1d(valIdxs, 'int32'));
  const yVal = tf.gather(y, tf.tensor1d(valIdxs, 'int32'));

  // Save validation JS arrays for metrics
  valLabels = Array.from(yVal.dataSync());

  // callbacks: tfjs-vis + early stopping
  const fitCallbacks = tfvis.show.fitCallbacks(
    { name: 'Training Performance', tab: 'Training' },
    ['loss', 'acc', 'val_loss', 'val_acc'],
    { callbacks: ['onEpochEnd'] }
  );

  const earlyStopping = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 });

  try {
    const history = await model.fit(xTrain, yTrain, {
      epochs: 50,
      batchSize: 32,
      validationData: [xVal, yVal],
      callbacks: [fitCallbacks, earlyStopping, {
        onEpochEnd: (epoch, logs) => {
          document.getElementById('training-status').innerHTML = `Epoch ${epoch+1} — loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc ? logs.acc.toFixed(4) : 'n/a'}, val_loss: ${logs.val_loss ? logs.val_loss.toFixed(4) : 'n/a'}, val_acc: ${logs.val_acc ? logs.val_acc.toFixed(4) : 'n/a'}`;
        }
      }]
    });

    // compute validation probabilities and store as JS array for ROC/metrics
    const valPredsTensor = model.predict(xVal);
    valProbs = Array.from(valPredsTensor.dataSync());

    // enable the threshold slider and predict/export
    document.getElementById('threshold-slider').disabled = false;
    document.getElementById('predict-btn').disabled = false;
    document.getElementById('export-btn').disabled = false;

    // calculate initial metrics and plot ROC
    await computeAndDisplayMetrics(valLabels, valProbs);
    document.getElementById('training-status').textContent += ' — Training finished.';
  } catch (err) {
    console.error(err);
    alert('Error during training: ' + err.message);
  } finally {
    // free tensors we created
    xTrain.dispose();
    yTrain.dispose();
    xVal.dispose();
    yVal.dispose();
  }
}

/* -------------------------
   Metrics: compute confusion matrix, precision, recall, f1, accuracy
   Plot ROC and compute AUC
*/
function computeConfusion(trueArr, probsArr, threshold) {
  let tp=0,tn=0,fp=0,fn=0;
  for (let i=0;i<trueArr.length;i++) {
    const pred = probsArr[i] >= threshold ? 1 : 0;
    const actual = trueArr[i];
    if (pred===1 && actual===1) tp++;
    else if (pred===1 && actual===0) fp++;
    else if (pred===0 && actual===1) fn++;
    else if (pred===0 && actual===0) tn++;
  }
  return {tp,tn,fp,fn};
}

async function computeAndDisplayMetrics(trueArr, probsArr) {
  // plot ROC & compute AUC
  const thresholds = Array.from({length:101}, (_,i)=>i/100);
  const rocPoints = [];
  for (const t of thresholds) {
    const {tp,tn,fp,fn} = computeConfusion(trueArr, probsArr, t);
    const tpr = (tp + fn) === 0 ? 0 : tp / (tp + fn);
    const fpr = (fp + tn) === 0 ? 0 : fp / (fp + tn);
    rocPoints.push({fpr, tpr});
  }
  // AUC via trapezoidal rule (sort by fpr)
  rocPoints.sort((a,b)=>a.fpr - b.fpr);
  let auc = 0;
  for (let i=1;i<rocPoints.length;i++) {
    const x1 = rocPoints[i-1].fpr, x2 = rocPoints[i].fpr;
    const y1 = rocPoints[i-1].tpr, y2 = rocPoints[i].tpr;
    auc += (x2 - x1) * (y1 + y2) / 2;
  }

  // plot ROC in tfjs-vis
  tfvis.render.linechart(
    { name: 'ROC Curve', tab: 'Evaluation' },
    { values: rocPoints.map(p => ({ x: p.fpr, y: p.tpr })) },
    { xLabel: 'False Positive Rate', yLabel: 'True Positive Rate', width: 400, height: 400 }
  );

  // update metrics at current slider threshold
  updateMetricsFromSlider();

  // add AUC to metrics panel
  const perfDiv = document.getElementById('performance-metrics');
  perfDiv.innerHTML += `<div>AUC: ${auc.toFixed(4)}</div>`;
}

/* Update metrics UI based on slider value */
function updateMetricsFromSlider() {
  if (!valProbs || !valLabels) return;
  const thr = parseFloat(document.getElementById('threshold-slider').value);
  document.getElementById('threshold-value').textContent = thr.toFixed(2);

  const {tp,tn,fp,fn} = computeConfusion(valLabels, valProbs, thr);
  // confusion matrix HTML table
  const cmDiv = document.getElementById('confusion-matrix');
  cmDiv.innerHTML = `
    <table>
      <tr><th></th><th>Pred +</th><th>Pred -</th></tr>
      <tr><th>Actual +</th><td>${tp}</td><td>${fn}</td></tr>
      <tr><th>Actual -</th><td>${fp}</td><td>${tn}</td></tr>
    </table>
  `;

  // metrics
  const precision = (tp + fp) === 0 ? 0 : tp / (tp + fp);
  const recall = (tp + fn) === 0 ? 0 : tp / (tp + fn);
  const f1 = (precision + recall) === 0 ? 0 : 2 * (precision * recall) / (precision + recall);
  const accuracy = (tp + tn) / (tp + tn + fp + fn);

  const perfDiv = document.getElementById('performance-metrics');
  perfDiv.innerHTML = `
    <div>Accuracy: ${(accuracy*100).toFixed(2)}%</div>
    <div>Precision: ${precision.toFixed(4)}</div>
    <div>Recall: ${recall.toFixed(4)}</div>
    <div>F1 Score: ${f1.toFixed(4)}</div>
  `;
}

/* -------------------------
   Predict on test set:
   - Build tensor from preprocessedTest.featuresArray
   - Store predictions tensor and show first 10 rows
*/
async function predict() {
  if (!model || !preprocessedTest) {
    alert('Train model and preprocess data first.');
    return;
  }

  document.getElementById('prediction-output').textContent = 'Predicting...';

  try {
    const testArr = preprocessedTest.featuresArray;
    const Xtest = tf.tensor2d(testArr);
    const preds = model.predict(Xtest);
    testPredictionsTensor = preds; // keep tensor for export (will be disposed on export if needed)
    const probs = Array.from(preds.dataSync());

    // create results
    const results = preprocessedTest.passengerIds.map((id, i) => ({
      PassengerId: id,
      Survived: probs[i] >= 0.5 ? 1 : 0,
      Probability: probs[i]
    }));

    // show first 10
    const outDiv = document.getElementById('prediction-output');
    outDiv.innerHTML = '<h4>Predictions (first 10)</h4>';
    outDiv.appendChild(createPredictionTable(results.slice(0,10)));
    outDiv.insertAdjacentHTML('beforeend', `<p>Total predictions: ${results.length}</p>`);

    // enable export
    document.getElementById('export-btn').disabled = false;

    // cleanup
    Xtest.dispose();
  } catch (err) {
    console.error(err);
    alert('Error during prediction: ' + err.message);
  }
}

function createPredictionTable(rows) {
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  thead.innerHTML = '<tr><th>PassengerId</th><th>Survived</th><th>Probability</th></tr>';
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  rows.forEach(r => {
    const tr = document.createElement('tr');
    const tdId = document.createElement('td'); tdId.textContent = r.PassengerId;
    const tdSurv = document.createElement('td'); tdSurv.textContent = r.Survived;
    const tdProb = document.createElement('td'); tdProb.textContent = r.Probability.toFixed(4);
    tr.appendChild(tdId); tr.appendChild(tdSurv); tr.appendChild(tdProb);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

/* -------------------------
   Export results:
   - download submission.csv (PassengerId,Survived)
   - download probabilities.csv (PassengerId,Probability)
   - save model to downloads://
*/
async function exportResults() {
  if (!testPredictionsTensor || !preprocessedTest) {
    alert('Please run prediction first.');
    return;
  }

  const statusDiv = document.getElementById('export-status');
  statusDiv.textContent = 'Preparing export...';

  try {
    const probs = Array.from(testPredictionsTensor.dataSync());
    // build CSVs
    let submissionCSV = 'PassengerId,Survived\n';
    let probsCSV = 'PassengerId,Probability\n';
    preprocessedTest.passengerIds.forEach((id, i) => {
      const survived = probs[i] >= 0.5 ? 1 : 0;
      submissionCSV += `${id},${survived}\n`;
      probsCSV += `${id},${probs[i].toFixed(6)}\n`;
    });

    // helper to trigger download
    function downloadBlob(text, filename) {
      const blob = new Blob([text], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    downloadBlob(submissionCSV, 'submission.csv');
    downloadBlob(probsCSV, 'probabilities.csv');

    // save model (this triggers browser download of model files)
    await model.save('downloads://titanic-tfjs');

    statusDiv.innerHTML = 'Export complete. Downloaded submission.csv, probabilities.csv, and model files.';
  } catch (err) {
    console.error(err);
    statusDiv.textContent = 'Export failed: ' + err.message;
    alert('Export error: ' + err.message);
  }
}

/* -------------------------
   END of file
   Summary of logic (for human readers / LLM summarization):
   - loadData: reads train/test CSVs, robustly parsing quoted fields with commas (e.g., "Last, First")
   - inspectData: shows preview, missing percentages, and charts (tfjs-vis)
   - preprocessData: imputes Age (median) and Embarked (mode), standardizes Age/Fare, one-hot encodes categorical fields, optionally adds family features; converts to tensors
   - createModel: builds a small sequential model for binary classification (16 ReLU -> 1 sigmoid)
   - trainModel: stratified 80/20 split, trains with tfjs-vis callbacks and early stopping; stores validation probabilities for evaluation
   - computeAndDisplayMetrics: computes ROC/AUC and shows ROC plot; slider updates confusion matrix and metrics dynamically
   - predict + exportResults: runs inference on test set, shows preview, downloads submission + probabilities, and saves model to downloads
*/
