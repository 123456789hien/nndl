/* app.js
   Titanic shallow binary classifier running in the browser (TensorFlow.js).
   - Uses only browser APIs (FileReader), TensorFlow.js, and tfjs-vis.
   - Handles CSV fields with quotes/commas.
   - Workflow: load -> inspect -> preprocess -> create model -> train -> evaluate -> predict -> export
*/

/* -------------------------
   Configuration / Schema
   -------------------------
   You can swap schema for a different dataset by changing TARGET_FEATURE, FEATURES, ID_FEATURE.
*/
const TARGET_FEATURE = 'Survived';    // target column (0/1)
const ID_FEATURE = 'PassengerId';     // ID column to exclude from features
const FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']; // feature set (order used)
const NUM_EPOCHS = 50;
const BATCH_SIZE = 32;

/* -------------------------
   State variables
   ------------------------- */
let rawTrain = null;
let rawTest = null;
let preprocessedTrain = null;  // {featuresTensor, labelsTensor, featureMean, featureStd, categoryMaps}
let preprocessedTest = null;   // {featuresTensor, passengerIds}
let tfModel = null;
let trainingStopRequested = false;
let valProbs = null;           // validation probabilities (Float32Array)
let valLabelsArr = null;       // validation labels (Int32Array)
let rocData = null;            // cached roc points and auc

/* -------------------------
   Utilities: DOM helpers
   ------------------------- */
const $ = id => document.getElementById(id);
const setText = (id, txt) => { const el = $(id); if(el) el.innerHTML = txt; };

/* -------------------------
   CSV parsing (handles quotes & commas)
   Implemented as a small state-machine to avoid external libs.
   Input: csvText string
   Output: array of row objects (headers -> values). Numeric strings converted to numbers.
*/
function parseCSVWithQuotes(csvText) {
  const rows = csvText.split(/\r?\n/).filter(r => r.trim() !== '');
  if (rows.length === 0) return [];
  // parse header
  const headers = splitCSVLine(rows[0]);
  const data = [];
  for (let i = 1; i < rows.length; i++) {
    const line = rows[i];
    const values = splitCSVLine(line);
    const obj = {};
    for (let j = 0; j < headers.length; j++) {
      const h = headers[j];
      let v = values[j] !== undefined ? values[j] : null;
      if (v === '') v = null;
      // convert numeric-like to number (but keep '0' '1' as numbers)
      if (v !== null && !isNaN(v) && v !== '') {
        // Use parseFloat to allow decimals (Fare, Age)
        v = parseFloat(v);
      }
      obj[h] = v;
    }
    data.push(obj);
  }
  return data;
}

// split a single CSV line handling double quotes
function splitCSVLine(line) {
  const res = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      // Handle escaped double quotes ""
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++; // skip next quote
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === ',' && !inQuotes) {
      res.push(cur);
      cur = '';
    } else {
      cur += ch;
    }
  }
  res.push(cur);
  return res;
}

/* -------------------------
   File reading
*/
function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = e => resolve(e.target.result);
    fr.onerror = () => reject(new Error('Failed to read file'));
    fr.readAsText(file);
  });
}

/* -------------------------
   Event wiring on load
*/
window.addEventListener('load', () => {
  // Buttons
  $('load-data-btn').addEventListener('click', onLoadDataClick);
  $('inspect-btn').addEventListener('click', onInspectClick);
  $('preprocess-btn').addEventListener('click', onPreprocessClick);
  $('create-model-btn').addEventListener('click', onCreateModelClick);
  $('train-btn').addEventListener('click', onTrainClick);
  $('stop-train-btn').addEventListener('click', onStopTrainClick);
  $('eval-btn').addEventListener('click', onEvalClick);
  $('threshold-slider').addEventListener('input', onThresholdChange);
  $('predict-btn').addEventListener('click', onPredictClick);
  $('export-btn').addEventListener('click', onExportClick);
});

/* -------------------------
   1) Load Data
*/
async function onLoadDataClick() {
  try {
    setText('data-status', 'Reading files...');
    const trainFile = $('train-file').files[0];
    const testFile  = $('test-file').files[0];
    if (!trainFile || !testFile) {
      alert('Please choose both train.csv and test.csv files.');
      setText('data-status', '');
      return;
    }
    const [trainText, testText] = await Promise.all([readFileAsText(trainFile), readFileAsText(testFile)]);
    rawTrain = parseCSVWithQuotes(trainText);
    rawTest = parseCSVWithQuotes(testText);
    setText('data-status', `Loaded: train=${rawTrain.length} rows, test=${rawTest.length} rows`);
    $('inspect-btn').disabled = false;
    // reset UI sections
    $('data-preview').innerHTML = '';
    $('data-stats').innerHTML = '';
    $('charts').innerHTML = '';
    $('preprocess-btn').disabled = true;
    $('create-model-btn').disabled = true;
    $('train-btn').disabled = true;
    $('predict-btn').disabled = true;
    $('export-btn').disabled = true;
    $('eval-btn').disabled = true;
    $('threshold-slider').disabled = true;
  } catch (err) {
    console.error(err);
    alert('Error loading CSV files: ' + err.message);
    setText('data-status', 'Error loading files');
  }
}

/* -------------------------
   2) Inspect data (preview, shape, missing %, charts)
*/
function onInspectClick() {
  if (!rawTrain) {
    alert('Load data first.');
    return;
  }
  // Preview first 10 rows
  renderPreviewTable(rawTrain.slice(0, 10), 'data-preview');

  // Shape + target stats
  const nRows = rawTrain.length;
  const nCols = Object.keys(rawTrain[0] || {}).length;
  const survivalCount = rawTrain.filter(r => r[TARGET_FEATURE] === 1).length;
  const survivalRate = ((survivalCount / nRows) * 100).toFixed(2);
  let statsHtml = `<p>Shape: ${nRows} rows x ${nCols} columns</p>`;
  statsHtml += `<p>${TARGET_FEATURE} positive: ${survivalCount} (${survivalRate}%)</p>`;

  // Missing percentages per column
  const headers = Object.keys(rawTrain[0] || {});
  statsHtml += '<h4>Missing % per column</h4><ul>';
  headers.forEach(h => {
    const missing = rawTrain.filter(r => r[h] === null || r[h] === undefined || r[h] === '').length;
    statsHtml += `<li>${h}: ${(missing / nRows * 100).toFixed(2)}%</li>`;
  });
  statsHtml += '</ul>';
  setText('data-stats', statsHtml);

  // Charts: Survival by Sex and by Pclass
  createVisualizations();

  // Enable preprocessing button
  $('preprocess-btn').disabled = false;
}

/* helper: render HTML preview table */
function renderPreviewTable(rows, containerId) {
  const container = $(containerId);
  if (!rows || rows.length === 0) {
    container.innerHTML = '<p>No data to preview</p>';
    return;
  }
  const table = document.createElement('table');
  const headerRow = document.createElement('tr');
  Object.keys(rows[0]).forEach(h => {
    const th = document.createElement('th'); th.textContent = h; headerRow.appendChild(th);
  });
  table.appendChild(headerRow);
  rows.forEach(r => {
    const tr = document.createElement('tr');
    Object.keys(r).forEach(h => {
      const td = document.createElement('td');
      td.textContent = (r[h] === null || r[h] === undefined) ? 'NULL' : String(r[h]);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  container.innerHTML = '';
  container.appendChild(table);
}

/* Create visualization charts rendered directly in page */
function createVisualizations() {
  const chartsDiv = $('charts');
  chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';

  // Survival by Sex
  const sexAgg = {};
  rawTrain.forEach(r => {
    const sex = r.Sex || 'Unknown';
    if (r[TARGET_FEATURE] !== null && r[TARGET_FEATURE] !== undefined) {
      sexAgg[sex] = sexAgg[sex] || {survived:0, total:0};
      sexAgg[sex].total++;
      if (r[TARGET_FEATURE] === 1) sexAgg[sex].survived++;
    }
  });
  const sexData = Object.keys(sexAgg).map(k => ({x: k, y: (sexAgg[k].survived / sexAgg[k].total) * 100}));
  const sexDiv = document.createElement('div'); sexDiv.style.marginTop='8px';
  chartsDiv.appendChild(sexDiv);
  tfvis.render.barchart(sexDiv, sexData, {xLabel: 'Sex', yLabel: 'Survival Rate (%)', width: 380, height: 240});

  // Survival by Pclass
  const classAgg = {};
  rawTrain.forEach(r => {
    const pc = (r.Pclass !== null && r.Pclass !== undefined) ? String(r.Pclass) : 'Unknown';
    if (r[TARGET_FEATURE] !== null && r[TARGET_FEATURE] !== undefined) {
      classAgg[pc] = classAgg[pc] || {survived:0, total:0};
      classAgg[pc].total++;
      if (r[TARGET_FEATURE] === 1) classAgg[pc].survived++;
    }
  });
  const classData = Object.keys(classAgg).map(k => ({x: `Class ${k}`, y: (classAgg[k].survived / classAgg[k].total) * 100}));
  const classDiv = document.createElement('div'); classDiv.style.marginTop='8px';
  chartsDiv.appendChild(classDiv);
  tfvis.render.barchart(classDiv, classData, {xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)', width: 380, height: 240});
}

/* -------------------------
   3) Preprocessing
   - Impute Age median, Embarked mode
   - Standardize Age and Fare using training mean/std
   - One-hot encode Sex, Pclass, Embarked
   - Optionally add FamilySize, IsAlone
*/
function onPreprocessClick() {
  if (!rawTrain || !rawTest) { alert('Load data first'); return; }
  try {
    // Compute imputation values on training set
    const ageVals = rawTrain.map(r => r.Age).filter(v => v !== null && v !== undefined && !isNaN(v));
    const fareVals = rawTrain.map(r => r.Fare).filter(v => v !== null && v !== undefined && !isNaN(v));
    const ageMedian = median(ageVals);
    const fareMedian = median(fareVals);
    const embarkedVals = rawTrain.map(r => r.Embarked).filter(v => v !== null && v !== undefined);
    const embarkedMode = mode(embarkedVals) || 'S';

    // Determine categories from training data (Pclass typically 1,2,3; Sex male/female; Embarked unique)
    const pclassCats = Array.from(new Set(rawTrain.map(r => r.Pclass))).filter(x => x !== null && x !== undefined).sort();
    const sexCats = Array.from(new Set(rawTrain.map(r => r.Sex))).filter(x => x !== null && x !== undefined);
    const embarkedCats = Array.from(new Set(rawTrain.map(r => r.Embarked))).filter(x => x !== null && x !== undefined);

    // Feature builder for a row, using training statistics (so preprocessing is reproducible)
    function buildFeatures(row) {
      // Impute
      const age = (row.Age !== null && row.Age !== undefined && !isNaN(row.Age)) ? row.Age : ageMedian;
      const fare = (row.Fare !== null && row.Fare !== undefined && !isNaN(row.Fare)) ? row.Fare : fareMedian;
      const sibsp = (row.SibSp !== null && row.SibSp !== undefined && !isNaN(row.SibSp)) ? row.SibSp : 0;
      const parch = (row.Parch !== null && row.Parch !== undefined && !isNaN(row.Parch)) ? row.Parch : 0;
      const pclass = row.Pclass !== null && row.Pclass !== undefined ? row.Pclass : pclassCats[0];
      const sex = row.Sex || sexCats[0];
      const embarked = row.Embarked || embarkedMode;

      // Numeric base features: standardized age and fare will be applied later
      const base = { age, fare, sibsp, parch };

      // One-hot encodings
      const pclassOH = pclassCats.map(c => (c === pclass ? 1 : 0));
      const sexOH = sexCats.map(s => (s === sex ? 1 : 0));
      const embarkedOH = embarkedCats.map(e => (e === embarked ? 1 : 0));

      // Family features optional
      const addFamily = $('add-family-features').checked;
      const familySize = addFamily ? (sibsp + parch + 1) : undefined;
      const isAlone = addFamily ? (familySize === 1 ? 1 : 0) : undefined;

      return {
        base,
        pclassOH,
        sexOH,
        embarkedOH,
        familySize,
        isAlone
      };
    }

    // Build matrix for train
    const trainFeatureRows = [];
    const trainLabels = [];
    rawTrain.forEach(r => {
      const f = buildFeatures(r);
      // store raw numeric parts in array to standardize after
      let rowArr = [f.base.age, f.base.fare, f.base.sibsp, f.base.parch]
        .concat(f.pclassOH).concat(f.sexOH).concat(f.embarkedOH);
      if ($('add-family-features').checked) rowArr = rowArr.concat([f.familySize, f.isAlone]);
      trainFeatureRows.push(rowArr);
      trainLabels.push((r[TARGET_FEATURE] !== null && r[TARGET_FEATURE] !== undefined) ? r[TARGET_FEATURE] : 0);
    });

    // Build matrix for test
    const testFeatureRows = [];
    const testIds = [];
    rawTest.forEach(r => {
      const f = buildFeatures(r);
      let rowArr = [f.base.age, f.base.fare, f.base.sibsp, f.base.parch]
        .concat(f.pclassOH).concat(f.sexOH).concat(f.embarkedOH);
      if ($('add-family-features').checked) rowArr = rowArr.concat([f.familySize, f.isAlone]);
      testFeatureRows.push(rowArr);
      testIds.push(r[ID_FEATURE]);
    });

    // Convert to tensors and standardize Age & Fare columns based on train stats
    const Xtrain = tf.tensor2d(trainFeatureRows);
    const ytrain = tf.tensor1d(trainLabels, 'int32');

    // Determine which indices correspond to Age and Fare in feature vector
    // Current order: [Age, Fare, SibSp, Parch] + pclassOH + sexOH + embarkedOH (+ family features)
    const ageIndex = 0;
    const fareIndex = 1;

    // Compute mean/std from training features
    const ageValsTensor = Xtrain.gather([ageIndex], 1).reshape([-1]);
    const fareValsTensor = Xtrain.gather([fareIndex], 1).reshape([-1]);
    const ageMean = ageValsTensor.mean().arraySync();
    const ageStd  = ageValsTensor.sub(ageMean).pow(2).mean().sqrt().arraySync() || 1;
    const fareMean = fareValsTensor.mean().arraySync();
    const fareStd  = fareValsTensor.sub(fareMean).pow(2).mean().sqrt().arraySync() || 1;

    // Standardize Age and Fare in Xtrain
    const XtrainArr = Xtrain.arraySync().map(row => {
      const newRow = row.slice();
      newRow[ageIndex] = (newRow[ageIndex] - ageMean) / (ageStd || 1);
      newRow[fareIndex] = (newRow[fareIndex] - fareMean) / (fareStd || 1);
      return newRow;
    });
    const XtrainStd = tf.tensor2d(XtrainArr);

    // Standardize test features with same train mean/std
    const Xtest = tf.tensor2d(testFeatureRows);
    const XtestArr = Xtest.arraySync().map(row => {
      const newRow = row.slice();
      newRow[ageIndex] = (newRow[ageIndex] - ageMean) / (ageStd || 1);
      newRow[fareIndex] = (newRow[fareIndex] - fareMean) / (fareStd || 1);
      return newRow;
    });
    const XtestStd = tf.tensor2d(XtestArr);

    // Save preprocessed objects for model use later
    preprocessedTrain = {
      featuresTensor: XtrainStd,
      labelsTensor: ytrain,
      ageMean, ageStd, fareMean, fareStd,
      pclassCats, sexCats, embarkedCats
    };
    preprocessedTest = {
      featuresTensor: XtestStd,
      passengerIds: testIds
    };

    // Update UI
    setText('preprocessing-output',
      `Preprocessing done. Train shape: [${XtrainStd.shape}], Labels: [${ytrain.shape}]. Test shape: [${XtestStd.shape}].`
    );

    $('create-model-btn').disabled = false;
  } catch (err) {
    console.error(err);
    alert('Preprocessing error: ' + err.message);
  }
}

// simple numeric helpers
function median(arr) {
  const a = arr.slice().sort((x,y) => x-y);
  const mid = Math.floor(a.length / 2);
  return (a.length % 2 === 0) ? (a[mid-1] + a[mid]) / 2 : a[mid];
}
function mode(arr) {
  const counts = {};
  arr.forEach(x => counts[x] = (counts[x]||0) + 1);
  let best = null, bestC = 0;
  for (const k in counts) if (counts[k] > bestC) { best = k; bestC = counts[k]; }
  return best;
}

/* -------------------------
   4) Model creation
*/
function onCreateModelClick() {
  if (!preprocessedTrain) { alert('Preprocess data first'); return; }
  // input shape = number of columns in feature tensor
  const inputShape = preprocessedTrain.featuresTensor.shape[1];
  tfModel = tf.sequential();
  tfModel.add(tf.layers.dense({units:16, activation:'relu', inputShape:[inputShape]}));
  tfModel.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  tfModel.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});

  // Display a simple summary in the UI
  let sumHtml = '<h3>Model Summary</h3><ul>';
  tfModel.layers.forEach((L,i) => sumHtml += `<li>Layer ${i+1}: ${L.getClassName()} - outputShape: ${JSON.stringify(L.outputShape)}</li>`);
  sumHtml += `</ul><p>Total params: ${tfModel.countParams()}</p>`;
  setText('model-summary', sumHtml);

  $('train-btn').disabled = false;
}

/* -------------------------
   Helper: Stratified split (train/val)
   returns {trainX, trainY, valX, valY}
*/
function stratifiedSplit(featuresTensor, labelsTensor, testFraction=0.2, seed=42) {
  // Convert to arrays for easy manipulation
  const X = featuresTensor.arraySync();
  const y = labelsTensor.arraySync();

  // Group indices by class label
  const groups = {};
  y.forEach((lbl, idx) => {
    groups[lbl] = groups[lbl] || [];
    groups[lbl].push(idx);
  });

  const trainIndices = [];
  const valIndices = [];

  // simple pseudo-random shuffle
  function shuffle(arr) {
    for (let i = arr.length-1; i>0; i--) {
      const j = Math.floor(Math.abs(Math.sin((i+seed)) * 10000) % (i+1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  for (const lbl in groups) {
    const idxs = groups[lbl].slice();
    shuffle(idxs);
    const nVal = Math.max(1, Math.round(idxs.length * testFraction));
    valIndices.push(...idxs.slice(0, nVal));
    trainIndices.push(...idxs.slice(nVal));
  }

  // build tensors
  const trainX = tf.tensor2d(trainIndices.map(i => X[i]));
  const trainY = tf.tensor1d(trainIndices.map(i => y[i]), 'int32');
  const valX = tf.tensor2d(valIndices.map(i => X[i]));
  const valY = tf.tensor1d(valIndices.map(i => y[i]), 'int32');
  return {trainX, trainY, valX, valY};
}

/* -------------------------
   5) Training
   - 80/20 stratified
   - 50 epochs, batch 32
   - tfjs-vis live plots
   - early stopping on val_loss (patience 5)
*/
async function onTrainClick() {
  if (!tfModel || !preprocessedTrain) { alert('Model & data required'); return; }
  $('train-btn').disabled = true;
  $('stop-train-btn').disabled = false;
  trainingStopRequested = false;
  setText('training-status', 'Preparing training...');

  const F = preprocessedTrain.featuresTensor;
  const Y = preprocessedTrain.labelsTensor;
  const {trainX, trainY, valX, valY} = stratifiedSplit(F, Y, 0.2);
  // store validation arrays for later metric calculation
  validationComputePrepare(valX, valY);

  // set up callbacks: tfvis + early stop + custom onEpochEnd to show status & allow stop
  const visCallbacks = tfvis.show.fitCallbacks(
    {name: 'Training Performance', tab: 'Training'},
    ['loss','acc','val_loss','val_acc'],
    { callbacks: ['onEpochEnd'] }
  );

  const earlyStop = tf.callbacks.earlyStopping({monitor:'val_loss', patience:5});

  const customCB = {
    onEpochEnd: async (epoch, logs) => {
      setText('training-status', `Epoch ${epoch+1}/${NUM_EPOCHS} — loss:${logs.loss.toFixed(4)}, acc:${(logs.acc||logs.acc).toFixed(4)}, val_loss:${(logs.val_loss||0).toFixed(4)}, val_acc:${(logs.val_acc||0).toFixed(4)}`);
      // If user requested stop, throw to end training
      if (trainingStopRequested) {
        throw new Error('Training stopped by user');
      }
    }
  };

  try {
    const history = await tfModel.fit(trainX, trainY, {
      epochs: NUM_EPOCHS,
      batchSize: BATCH_SIZE,
      validationData: [valX, valY],
      callbacks: [visCallbacks, earlyStop, customCB]
    });
    setText('training-status', 'Training completed.');
    $('stop-train-btn').disabled = true;
    // after training, compute validation probabilities and metrics
    const preds = tfModel.predict(valX);
    valProbs = await preds.data(); // Float32Array
    valLabelsArr = valY.dataSync();
    // compute ROC/AUC and draw initial ROC
    computeROCandAUC(valLabelsArr, Array.from(valProbs));
    // enable evaluation & threshold UI
    $('eval-btn').disabled = false;
    $('threshold-slider').disabled = false;
    $('predict-btn').disabled = false;
  } catch (err) {
    // training aborted or error
    console.error(err);
    setText('training-status', 'Training stopped or error: ' + err.message);
    $('train-btn').disabled = false;
    $('stop-train-btn').disabled = true;
  }
}

// called when user clicks stop
function onStopTrainClick() {
  trainingStopRequested = true;
  setText('training-status', 'Stop requested; finishing epoch...');
}

/* store validation tensors for computing confusion etc. */
let _valXtensor = null;
let _valYtensor = null;
function validationComputePrepare(valX, valY) {
  if (_valXtensor) _valXtensor.dispose();
  if (_valYtensor) _valYtensor.dispose();
  _valXtensor = valX;
  _valYtensor = valY;
}

/* -------------------------
   6) Evaluation: ROC/AUC, threshold slider, confusion matrix, metrics
*/
function onEvalClick() {
  if (!valProbs || !valLabelsArr) { alert('No validation predictions available. Train first.'); return; }
  // Already computed after training, so just update UI
  drawConfusionAndMetrics(parseFloat($('threshold-slider').value));
}

function onThresholdChange(e) {
  const t = parseFloat(e.target.value);
  $('threshold-value').textContent = t.toFixed(2);
  drawConfusionAndMetrics(t);
}

/* Compute ROC points & AUC from arrays of true labels and predicted probabilities */
function computeROCandAUC(yTrueArr, yScoreArr) {
  // thresholds from 0..1
  const thresholds = Array.from({length:101}, (_,i) => i/100);
  const rocPoints = [];
  for (const thr of thresholds) {
    let tp=0, fp=0, tn=0, fn=0;
    for (let i=0;i<yTrueArr.length;i++) {
      const actual = yTrueArr[i];
      const pred = (yScoreArr[i] >= thr) ? 1 : 0;
      if (actual === 1 && pred === 1) tp++;
      if (actual === 0 && pred === 1) fp++;
      if (actual === 0 && pred === 0) tn++;
      if (actual === 1 && pred === 0) fn++;
    }
    const tpr = (tp) / (tp + fn + 1e-12);
    const fpr = (fp) / (fp + tn + 1e-12);
    rocPoints.push({threshold: thr, tpr, fpr});
  }
  // Sort by fpr asc
  rocPoints.sort((a,b) => a.fpr - b.fpr);
  // compute AUC with trapezoidal rule
  let auc = 0.0;
  for (let i=1;i<rocPoints.length;i++) {
    const x1 = rocPoints[i-1].fpr, y1 = rocPoints[i-1].tpr;
    const x2 = rocPoints[i].fpr, y2 = rocPoints[i].tpr;
    auc += (x2 - x1) * (y1 + y2) / 2.0;
  }
  rocData = {points: rocPoints, auc};
  // draw ROC chart
  drawROCChart();
}

/* Draw ROC into #roc-chart card */
function drawROCChart() {
  if (!rocData) return;
  const points = rocData.points.map(p => ({x: p.fpr, y: p.tpr}));
  const container = $('roc-chart');
  container.innerHTML = `<h4>ROC Curve (AUC=${rocData.auc.toFixed(4)})</h4>`;
  const chartDiv = document.createElement('div');
  container.appendChild(chartDiv);
  // tfvis expects {values: [{x,y}, ...]}
  tfvis.render.linechart(chartDiv, {values: [points]}, {
    xLabel: 'False Positive Rate', yLabel: 'True Positive Rate', width: 360, height: 300
  });
}

/* Compute confusion matrix & metrics for chosen threshold and render */
function drawConfusionAndMetrics(threshold=0.5) {
  if (!valProbs || !valLabelsArr) return;
  let tp=0, tn=0, fp=0, fn=0;
  for (let i=0;i<valLabelsArr.length;i++) {
    const actual = valLabelsArr[i];
    const pred = (valProbs[i] >= threshold) ? 1 : 0;
    if (actual === 1 && pred === 1) tp++;
    if (actual === 0 && pred === 0) tn++;
    if (actual === 0 && pred === 1) fp++;
    if (actual === 1 && pred === 0) fn++;
  }

  // confusion matrix HTML
  const cmDiv = $('confusion-matrix');
  cmDiv.innerHTML = `<h4>Confusion Matrix (threshold=${threshold.toFixed(2)})</h4>
    <table><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
    <tr><th>True 0</th><td>${tn}</td><td>${fp}</td></tr>
    <tr><th>True 1</th><td>${fn}</td><td>${tp}</td></tr></table>`;

  // metrics
  const accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12);
  const precision = tp / (tp + fp + 1e-12);
  const recall = tp / (tp + fn + 1e-12);
  const f1 = 2 * precision * recall / (precision + recall + 1e-12);

  const perfDiv = $('performance-metrics');
  perfDiv.innerHTML = `<h4>Metrics</h4>
    <p>Accuracy: ${accuracy.toFixed(4)}</p>
    <p>Precision: ${precision.toFixed(4)}</p>
    <p>Recall: ${recall.toFixed(4)}</p>
    <p>F1 Score: ${f1.toFixed(4)}</p>`;
}

/* -------------------------
   7) Prediction on test.csv & export
   - Predict probabilities on preprocessedTest.featuresTensor
   - Apply threshold to produce Survived 0/1, create CSVs and download
*/
async function onPredictClick() {
  if (!tfModel || !preprocessedTest) { alert('Model and preprocessed test data required'); return; }
  try {
    setText('prediction-output', 'Predicting...');
    const predsTensor = tfModel.predict(preprocessedTest.featuresTensor);
    const probs = await predsTensor.data();
    // store last predictions
    preprocessedTest.lastProbabilities = Array.from(probs);
    // show some preview in UI
    const previewHtml = `<h4>Prediction probabilities (first 20)</h4><div class="small">${preprocessedTest.lastProbabilities.slice(0,20).map(p => p.toFixed(4)).join(', ')}</div>`;
    setText('prediction-output', previewHtml);
    $('export-btn').disabled = false;
  } catch (err) {
    console.error(err);
    alert('Prediction error: ' + err.message);
  }
}

/* Export results: submission.csv (PassengerId,Survived) and probabilities.csv; also save model */
function onExportClick() {
  if (!preprocessedTest || !preprocessedTest.lastProbabilities) { alert('Run prediction first'); return; }
  const ids = preprocessedTest.passengerIds || rawTest.map(r => r[ID_FEATURE]);
  const probs = preprocessedTest.lastProbabilities;
  // create submission CSV using threshold slider value
  const thr = parseFloat($('threshold-slider').value);
  let subCsv = 'PassengerId,Survived\n';
  let probCsv = 'PassengerId,Probability\n';
  for (let i=0;i<ids.length;i++) {
    const survived = (probs[i] >= thr) ? 1 : 0;
    subCsv += `${ids[i]},${survived}\n`;
    probCsv += `${ids[i]},${probs[i].toFixed(6)}\n`;
  }
  downloadBlob(subCsv, 'submission.csv');
  downloadBlob(probCsv, 'probabilities.csv');
  // save model to downloads (tfjs)
  tfModel.save('downloads://titanic-tfjs').then(() => {
    alert('Model saved to downloads (tfjs). CSVs downloaded.');
  }).catch(err => {
    console.error(err);
    alert('Model save error: ' + err.message);
  });
}

function downloadBlob(text, filename) {
  const blob = new Blob([text], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* -------------------------
   Error handling and small helpers
*/
function safeNumber(x, fallback=0) {
  return (x !== null && x !== undefined && !isNaN(x)) ? x : fallback;
}

/* -------------------------
   End of file
   -------------------------
   Notes:
   - This code runs entirely in browser, ready for GitHub Pages.
   - Schema swap: change TARGET_FEATURE, FEATURES, ID_FEATURE at top to adapt to other datasets.
   - For larger datasets, consider streaming parsing or server-side preprocessing.
*/
