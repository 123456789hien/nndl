// =======================
// Titanic Classifier JS
// =======================

// Global variables
let trainData = [];
let testData = [];
let processed = {};
let model;
let stopFlag = false;
let valProbs = [];
let valLabels = [];

// ---------- Utility ----------
function alertError(msg) { alert(msg); console.error(msg); }

function parseCSVFile(file, callback) {
  const reader = new FileReader();
  reader.onload = e => {
    // FIX: handle commas inside quotes
    const text = e.target.result;
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    const headers = lines[0].split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/).map(h => h.replace(/"/g, ''));
    const data = lines.slice(1).map(line => {
      const cols = line.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/).map(c => c.replace(/^"|"$/g, ''));
      const obj = {};
      headers.forEach((h, i) => obj[h] = cols[i]);
      return obj;
    });
    callback(data);
  };
  reader.onerror = () => alertError("Error reading CSV file.");
  reader.readAsText(file);
}

// ---------- 1. Load & Inspect ----------
function loadData() {
  const trainFile = document.getElementById('trainFile').files[0];
  const testFile = document.getElementById('testFile').files[0];
  if (!trainFile || !testFile) { alertError("Please upload both train and test CSV files!"); return; }

  parseCSVFile(trainFile, data => { trainData = data; console.log("Train loaded:", trainData.length); });
  parseCSVFile(testFile, data => { testData = data; console.log("Test loaded:", testData.length); });
  alert("Data loaded! Now click 'Inspect Data'.");
}

function inspectData() {
  if (trainData.length === 0) { alertError("Load data first!"); return; }

  // Preview first 5 rows
  const preview = document.getElementById('dataPreview');
  preview.innerHTML = tableHTML(trainData.slice(0,5));

  // Charts
  const bySex = { male:0, female:0 };
  const byPclass = { '1':0, '2':0, '3':0 };
  trainData.forEach(r=>{
    if (r.Survived==='1') {
      if (r.Sex) bySex[r.Sex]++;
      if (r.Pclass) byPclass[r.Pclass]++;
    }
  });

  const surface = { name: 'Survival Charts', tab: 'Charts' };
  tfvis.render.barchart(surface, [
    {name:'Male', value:bySex.male},
    {name:'Female', value:bySex.female}
  ], { width: 300, height: 200, xLabel:'Sex', yLabel:'Survived' });

  tfvis.render.barchart({ name:'By Pclass', tab:'Charts' }, 
    Object.keys(byPclass).map(k=>({name:k, value:byPclass[k]})),
    { width: 300, height: 200, xLabel:'Pclass', yLabel:'Survived' });
}

function tableHTML(rows) {
  if(rows.length===0) return "<p>No data</p>";
  const headers = Object.keys(rows[0]);
  const th = headers.map(h=>`<th>${h}</th>`).join('');
  const tr = rows.map(r=>`<tr>${headers.map(h=>`<td>${r[h]}</td>`).join('')}</tr>`).join('');
  return `<table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`;
}

// ---------- 2. Preprocessing ----------
function preprocess() {
  if (trainData.length===0) { alertError("Load data first!"); return; }

  const clean = d => {
    const obj = {...d};
    obj.Age = parseFloat(obj.Age) || NaN;
    obj.Fare = parseFloat(obj.Fare) || NaN;
    obj.Survived = parseInt(obj.Survived);
    return obj;
  };
  const data = trainData.map(clean);

  // Impute
  const median = arr => { const s=arr.filter(v=>!isNaN(v)).sort((a,b)=>a-b); const m=Math.floor(s.length/2); return s.length%2?s[m]:(s[m-1]+s[m])/2; };
  const ageMed = median(data.map(r=>r.Age));
  const fareMed = median(data.map(r=>r.Fare));

  data.forEach(r=>{
    if (isNaN(r.Age)) r.Age = ageMed;
    if (isNaN(r.Fare)) r.Fare = fareMed;
  });

  // Standardize
  const mean = arr=>arr.reduce((a,b)=>a+b,0)/arr.length;
  const std = arr=>Math.sqrt(arr.reduce((a,b)=>a+Math.pow(b-mean(arr),2),0)/arr.length);
  const ageMean = mean(data.map(r=>r.Age)), ageStd = std(data.map(r=>r.Age));
  const fareMean = mean(data.map(r=>r.Fare)), fareStd = std(data.map(r=>r.Fare));

  data.forEach(r=>{
    r.Age = (r.Age - ageMean) / ageStd;
    r.Fare = (r.Fare - fareMean) / fareStd;
  });

  processed.X = data.map(r=>[
    parseInt(r.Pclass),
    r.Sex==='female'?1:0,
    r.Age,
    parseInt(r.SibSp),
    parseInt(r.Parch),
    r.Fare,
    r.Embarked==='C'?1:r.Embarked==='Q'?2:0
  ]);
  processed.y = data.map(r=>r.Survived);

  document.getElementById('preprocessInfo').innerText = `Processed ${processed.X.length} rows, ${processed.X[0].length} features.`;
}

// ---------- 3. Model ----------
function createModel() {
  model = tf.sequential();
  model.add(tf.layers.dense({inputShape:[7], units:16, activation:'relu'}));
  model.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  model.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});
  const summary = [];
  model.summary( line => summary.push(line) );
  document.getElementById('modelSummary').innerText = summary.join('\n');
}

async function trainModel() {
  if (!model || !processed.X) { alertError("Preprocess & Create Model first!"); return; }
  stopFlag = false;

  const X = tf.tensor2d(processed.X);
  const y = tf.tensor2d(processed.y, [processed.y.length,1]);

  const N = X.shape[0];
  const idx = Math.floor(0.8*N);
  const Xtrain = X.slice([0,0],[idx,7]);
  const ytrain = y.slice([0,0],[idx,1]);
  const Xval = X.slice([idx,0],[N-idx,7]);
  const yval = y.slice([idx,0],[N-idx,1]);

  const surface = { name:'Training Performance', tab:'Training' };
  await model.fit(Xtrain, ytrain, {
    epochs:50,
    batchSize:32,
    validationData:[Xval, yval],
    callbacks: tfvis.show.fitCallbacks(surface, ['loss','val_loss','acc','val_acc'], { callbacks:['onEpochEnd'] })
  }).catch(err=>alertError("Training stopped or error: "+err));

  // Save validation set for ROC
  valProbs = model.predict(Xval).arraySync().map(v=>v[0]);
  valLabels = yval.arraySync().map(v=>v[0]);

  X.dispose(); y.dispose();
}

function stopTraining() {
  stopFlag = true;
  if (model) model.stopTraining = true;
  alert("Training stopped by user.");
}

// ---------- 4. Metrics ----------
function computeMetrics() {
  if (valProbs.length===0) { alertError("Train model first!"); return; }

  // Simple ROC chart
  const points = [];
  for (let t=0; t<=1; t+=0.1) {
    const cm = confusion(valProbs, valLabels, t);
    const TPR = cm.TP / (cm.TP + cm.FN);
    const FPR = cm.FP / (cm.FP + cm.TN);
    points.push({x:FPR, y:TPR});
  }
  const surface = { name:'ROC Curve', tab:'Evaluation' };
  tfvis.render.scatterplot(surface, {values:points}, {xLabel:'FPR', yLabel:'TPR'});

  updateThreshold(0.5);
}

function confusion(probs, labels, th) {
  let TP=0,FP=0,TN=0,FN=0;
  probs.forEach((p,i)=>{
    const pred = p>=th?1:0;
    const act = labels[i];
    if(pred===1 && act===1) TP++;
    else if(pred===1 && act===0) FP++;
    else if(pred===0 && act===0) TN++;
    else FN++;
  });
  return {TP,FP,TN,FN};
}

function updateThreshold(val) {
  document.getElementById('thresholdVal').innerText = val;
  const cm = confusion(valProbs, valLabels, parseFloat(val));
  document.getElementById('confMatrix').innerHTML = `
    <p>TP:${cm.TP} FP:${cm.FP} TN:${cm.TN} FN:${cm.FN}</p>`;
}

// ---------- 5. Prediction & Export ----------
function predictTest() {
  if (!model || testData.length===0) { alertError("Need model and test data."); return; }

  // Preprocess test
  const testX = testData.map(r=>[
    parseInt(r.Pclass),
    r.Sex==='female'?1:0,
    (parseFloat(r.Age)||0),
    parseInt(r.SibSp),
    parseInt(r.Parch),
    (parseFloat(r.Fare)||0),
    r.Embarked==='C'?1:r.Embarked==='Q'?2:0
  ]);

  const probs = model.predict(tf.tensor2d(testX)).arraySync().map(v=>v[0]);
  testData.forEach((r,i)=> r.Survived = probs[i]>=0.5?1:0 );

  document.getElementById('predictionPreview').innerHTML = tableHTML(testData.slice(0,5));
}

function exportResults() {
  if (testData.length===0) { alertError("No predictions."); return; }

  let csv = "PassengerId,Survived\n";
  testData.forEach(r=> csv += `${r.PassengerId},${r.Survived}\n`);
  const blob = new Blob([csv], {type:'text/csv'});
  const url = URL.createObjectURL(blob);
  const
