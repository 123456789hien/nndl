/****************************************************
 * Titanic Classifier - TensorFlow.js (Browser only)
 * Make sure index.html links to this file at bottom.
 ****************************************************/

// ---------------------------
// Global state
// ---------------------------
let trainRaw = null;
let testRaw = null;
let merged = null;
let preprocessedTrain = null;
let preprocessedTest = null;
let model = null;
let trainingHistory = null;
let valProbs = null;
let valLabels = null;
let stopRequested = false;

const TARGET = 'Survived';
const ID_COL = 'PassengerId';
const FEATURE_COLS = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'];

// ----------------------
// Utilities
// ----------------------
function setStatus(msg) {
  const el = document.getElementById('load-status');
  el.innerText = msg;
}

function toNum(v) {
  if (v === null || v === undefined || v === '') return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
}

const papaOptions = {
  header: true,
  dynamicTyping: true,
  skipEmptyLines: true,
  quoteChar: '"',
  escapeChar: '"'
};

// ----------------------
// 1) Load Data
// ----------------------
function loadData() {
  const trainFile = document.getElementById('train-file').files[0];
  const testFile = document.getElementById('test-file').files[0];
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
    }
  });

  Papa.parse(testFile, {
    ...papaOptions,
    complete: (results) => {
      testRaw = results.data.map(normalizeRow);
      console.log('Test rows:', testRaw.length);
      setStatus(`Test loaded: ${testRaw.length} rows.`);
      enableIfBothLoaded();
    },
    error: (err) => {
      alert('Error parsing test.csv: ' + err.message);
    }
  });
}

function normalizeRow(row) {
  const out = {};
  Object.keys(row).forEach(k => {
    let v = row[k];
    if (typeof v === 'string') v = v.trim();
    out[k] = v === '' ? null : v;
  });
  return out;
}

function enableIfBothLoaded() {
  if (trainRaw && testRaw) {
    document.getElementById('run-eda-btn').disabled = false;
    document.getElementById('export-merged-btn').disabled = false;
    setStatus(`Loaded train (${trainRaw.length}) and test (${testRaw.length}). Click 'Run EDA'.`);
    merged = buildMerged(trainRaw, testRaw);
  }
}

function buildMerged(train, test) {
  const t = train.map(r => ({ ...r, source: 'train' }));
  const s = test.map(r => ({ ...r, source: 'test' }));
  return t.concat(s);
}

// ----------------------
// 2) Run EDA
// ----------------------
function runEDA() {
  if (!merged) { alert('No data loaded'); return; }

  renderPreview(merged.slice(0, 8));

  const shapeInfo = `Merged shape: ${merged.length} rows × ${Object.keys(merged[0]).length} cols`;
  document.getElementById('overview').innerText = shapeInfo;

  const cols = Object.keys(merged[0]);
  const missing = {};
  cols.forEach(c => {
    const count = merged.filter(r => r[c] === null || r[c] === undefined || r[c] === '').length;
    missing[c] = +(count / merged.length * 100).toFixed(2);
  });
  renderMissing(missing);
  renderStatsSummary();
  renderCharts();

  document.getElementById('preprocess-btn').disabled = false;
}

function renderPreview(rows) {
  const container = document.getElementById('head-preview');
  const cols = Object.keys(rows[0]);
  let html = '<table><thead><tr>';
  cols.forEach(c => html += `<th>${c}</th>`);
  html += '</tr></thead><tbody>';
  rows.forEach(r => {
    html += '<tr>';
    cols.forEach(c => html += `<td>${r[c] ?? ''}</td>`);
    html += '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function renderMissing(missing) {
  const chartEl = document.getElementById('missing-chart');
  chartEl.innerHTML = '';
  const dom = document.createElement('div');
  chartEl.appendChild(dom);

  const data = Object.entries(missing).map(([k, v]) => ({ index: k, value: v }));
  tfvis.render.barchart({ dom }, data, { xLabel: 'Column', yLabel: 'Missing %', height: 200 });

  const tbl = document.getElementById('missing-table');
  let html = '<table><thead><tr><th>Column</th><th>Missing %</th></tr></thead><tbody>';
  Object.entries(missing).forEach(([k, v]) => {
    html += `<tr><td>${k}</td><td>${v}%</td></tr>`;
  });
  html += '</tbody></table>';
  tbl.innerHTML = html;
}

function renderStatsSummary() {
  const numericCols = ['Age', 'Fare', 'SibSp', 'Parch'];
  const summary = {};

  numericCols.forEach(col => {
    const vals = merged.map(r => toNum(r[col])).filter(v => v !== null);
    if (vals.length === 0) return;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    summary[col] = { mean: +mean.toFixed(2), min, max, count: vals.length };
  });

  const cats = {};
  ['Sex', 'Pclass', 'Embarked'].forEach(c => {
    cats[c] = {};
    merged.forEach(r => {
      const key = r[c] == null ? 'null' : String(r[c]);
      cats[c][key] = (cats[c][key] || 0) + 1;
    });
  });

  document.getElementById('stats-summary').innerHTML =
    `<pre>${JSON.stringify({ numeric: summary, categorical: cats }, null, 2)}</pre>`;
}

function renderCharts() {
  const sexCounts = {};
  merged.forEach(r => { const s = r.Sex || 'null'; sexCounts[s] = (sexCounts[s] || 0) + 1; });
  tfvis.render.barchart({ name: 'Counts by Sex', tab: 'Charts' },
    Object.entries(sexCounts).map(([x, y]) => ({ index: x, value: y })));

  const pclassCounts = {};
  merged.forEach(r => { const k = r.Pclass || 'null'; pclassCounts[k] = (pclassCounts[k] || 0) + 1; });
  tfvis.render.barchart({ name: 'Counts by Pclass', tab: 'Charts' },
    Object.entries(pclassCounts).map(([x, y]) => ({ index: `Class ${x}`, value: y })));

  const embCounts = {};
  merged.forEach(r => { const k = r.Embarked || 'null'; embCounts[k] = (embCounts[k] || 0) + 1; });
  tfvis.render.barchart({ name: 'Counts by Embarked', tab: 'Charts' },
    Object.entries(embCounts).map(([x, y]) => ({ index: x, value: y })));
}

// ----------------------
// ---------------------------
// 3. Preprocess
// ---------------------------
function preprocess() {
  if (!rawTrainData || !rawTestData) {
    alert('Load data first!');
    return;
  }

  // Columns to use
  const feats = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'];
  const labelCol = 'Survived';

  // ---------- Helper: process a dataset ----------
  function processDataset(data, isTrain=true) {
    // Deep copy
    data = data.map(row => ({...row}));

    // Convert to numeric and clean
    data.forEach(r=>{
      // Handle missing Age (median)
      if (!r.Age) r.Age = NaN;
      else r.Age = parseFloat(r.Age);
      if (!r.Fare) r.Fare = NaN;
      else r.Fare = parseFloat(r.Fare);

      r.SibSp = parseInt(r.SibSp || 0);
      r.Parch = parseInt(r.Parch || 0);
      r.Pclass = parseInt(r.Pclass || 0);
      // Encode Sex
      r.Sex = (r.Sex === 'male') ? 1 : 0;
      // Embarked
      r.Embarked = (r.Embarked==='S'?0 : (r.Embarked==='C'?1 : (r.Embarked==='Q'?2:NaN)));
      if (isTrain) r[labelCol] = parseInt(r[labelCol]);
    });

    // Impute Age/Fare/Embarked
    const ageVals = data.map(d=>d.Age).filter(v=>!isNaN(v));
    const fareVals = data.map(d=>d.Fare).filter(v=>!isNaN(v));
    const embVals = data.map(d=>d.Embarked).filter(v=>!isNaN(v));

    const median = arr => {
      const s = [...arr].sort((a,b)=>a-b);
      const m = Math.floor(s.length/2);
      return s.length%2 ? s[m] : (s[m-1]+s[m])/2;
    };
    const mode = arr => {
      const cnt = {};
      arr.forEach(v=>cnt[v]=(cnt[v]||0)+1);
      return parseInt(Object.keys(cnt).reduce((a,b)=> cnt[a]>cnt[b]?a:b));
    };

    const ageMed = median(ageVals);
    const fareMed = median(fareVals);
    const embMode = mode(embVals);

    data.forEach(r=>{
      if (isNaN(r.Age)) r.Age = ageMed;
      if (isNaN(r.Fare)) r.Fare = fareMed;
      if (isNaN(r.Embarked)) r.Embarked = embMode;
    });

    // Standardize Age & Fare
    const std = (arr) => {
      const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
      const stddev = Math.sqrt(arr.map(x=>(x-mean)**2).reduce((a,b)=>a+b,0)/arr.length);
      return {mean,stddev};
    };
    const ageStats = std(data.map(d=>d.Age));
    const fareStats = std(data.map(d=>d.Fare));

    data.forEach(r=>{
      r.Age = (r.Age - ageStats.mean) / ageStats.stddev;
      r.Fare = (r.Fare - fareStats.mean) / fareStats.stddev;
    });

    // Features matrix and label
    const X = data.map(r=>[
      r.Pclass, r.Sex, r.Age, r.SibSp, r.Parch, r.Fare,
      // One-hot Embarked: 3 cols
      r.Embarked===0?1:0,
      r.Embarked===1?1:0,
      r.Embarked===2?1:0
    ]);
    const y = isTrain ? data.map(r=>r[labelCol]) : null;

    return {X, y};
  }

  processedTrain = processDataset(rawTrainData,true);
  processedTest = processDataset(rawTestData,false);

  document.getElementById('preprocessInfo').innerText =
    `✅ Preprocessed!
     Train shape: ${processedTrain.X.length} x ${processedTrain.X[0].length}
     Test shape: ${processedTest.X.length} x ${processedTest.X[0].length}`;
}

// ---------------------------
// 4. Model creation
// ---------------------------
function createModel() {
  if (!processedTrain) { alert('Preprocess first!'); return; }

  model = tf.sequential();
  model.add(tf.layers.dense({units:16, activation:'relu', inputShape:[processedTrain.X[0].length]}));
  model.add(tf.layers.dense({units:1, activation:'sigmoid'}));

  model.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});

  const summaryLines = [];
  model.summary(l => summaryLines.push(l));
  document.getElementById('modelSummary').textContent = summaryLines.join('\n');

  alert('✅ Model created!');
}

// ---------------------------
// 5. Train
// ---------------------------
async function trainModel() {
  if (!model || !processedTrain) { alert('Need preprocess + createModel first!'); return; }

  stopFlag = false;

  const X = tf.tensor2d(processedTrain.X);
  const y = tf.tensor2d(processedTrain.y, [processedTrain.y.length,1]);

  // Shuffle & split
  const total = X.shape[0];
  const idx = tf.util.createShuffledIndices(total);
  const trainCount = Math.floor(total*0.8);

  const Xtrain = X.gather(idx.slice(0,trainCount));
  const ytrain = y.gather(idx.slice(0,trainCount));
  const Xval = X.gather(idx.slice(trainCount));
  const yval = y.gather(idx.slice(trainCount));

  const surface = { name: 'Training', tab: 'Charts' };
  const fitCallbacks = tfvis.show.fitCallbacks(surface, ['loss','acc'], {callbacks:['onEpochEnd']});

  await model.fit(Xtrain,ytrain,{
    epochs:50,
    batchSize:32,
    validationData:[Xval,yval],
    callbacks:{
      ...fitCallbacks,
      onEpochEnd:(epoch,logs)=>{
        if(stopFlag) return false;
      }
    }
  });

  X.dispose(); y.dispose();
  Xtrain.dispose(); ytrain.dispose(); Xval.dispose(); yval.dispose();

  alert('✅ Training done');
}

function stopTraining(){ stopFlag = true; alert('⏹️ Stop requested'); }

// ---------------------------
// 6. Placeholder for metrics, predict, export
// ---------------------------
function computeMetrics(){ alert('Metrics computation not implemented in this demo'); }

function predictTest(){ alert('Predict not implemented in this demo'); }

function exportResults(){ alert('Export not implemented in this demo'); }

// Expose to global
window.loadData = loadData;
window.inspectData = inspectData;
window.preprocess = preprocess;
window.createModel = createModel;
window.trainModel = trainModel;
window.stopTraining = stopTraining;
window.computeMetrics = computeMetrics;
window.predictTest = predictTest;
window.exportResults = exportResults;
