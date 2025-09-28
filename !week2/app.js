/****************************************************
 * Titanic Classifier - TensorFlow.js (Browser only)
 * Make sure index.html links to this file at bottom.
 ****************************************************/

// ---------------------------
// Global state
// ---------------------------
let rawTrainData = null;
let rawTestData = null;
let processedTrain = null;
let processedTest = null;

let model = null;
let stopFlag = false;

// ---------------------------
// 1. Load CSV from file input
// ---------------------------
async function loadData() {
  try {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];
    if (!trainFile || !testFile) {
      alert('Please upload both train.csv and test.csv');
      return;
    }

    rawTrainData = await readCSV(trainFile);
    rawTestData = await readCSV(testFile);

    document.getElementById('dataPreview').innerHTML =
      `<p>✅ Train loaded: ${rawTrainData.length} rows<br>
         ✅ Test loaded: ${rawTestData.length} rows</p>`;
  } catch (err) {
    console.error(err);
    alert('Error loading CSV files. Check console.');
  }
}

// Utility: read local CSV file
function readCSV(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => {
      const text = e.target.result.trim();
      const rows = text.split(/\r?\n/);
      const header = rows[0].split(',');
      const data = rows.slice(1).map(r => {
        const vals = r.split(',');
        const obj = {};
        header.forEach((h, i) => obj[h] = vals[i]);
        return obj;
      });
      resolve(data);
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

// ---------------------------
// 2. Inspect basic info
// ---------------------------
function inspectData() {
  if (!rawTrainData) {
    alert('Load data first!');
    return;
  }
  // Preview first 5 rows
  const cols = Object.keys(rawTrainData[0]);
  let html = `<table><thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>`;
  rawTrainData.slice(0,5).forEach(r=>{
    html += `<tr>${cols.map(c=>`<td>${r[c]}</td>`).join('')}</tr>`;
  });
  html += `</tbody></table>`;
  document.getElementById('dataPreview').innerHTML = html;
}

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
