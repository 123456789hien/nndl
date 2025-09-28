// Titanic TensorFlow.js App
// Author: ChatGPT
// This script runs entirely in the browser with tfjs and tfjs-vis

let trainData = [];
let testData = [];
let model;
let preprocessed = {};
let valPredProbs, valLabels;

// ====================== LOAD CSV ======================
function parseCSVRow(row) {
  return {
    PassengerId: row.PassengerId ? +row.PassengerId : null,
    Survived: row.Survived !== undefined ? +row.Survived : null,
    Pclass: row.Pclass ? +row.Pclass : null,
    Sex: row.Sex ? row.Sex.trim().toLowerCase() : null,
    Age: row.Age ? +row.Age : null,
    SibSp: row.SibSp ? +row.SibSp : 0,
    Parch: row.Parch ? +row.Parch : 0,
    Fare: row.Fare ? +row.Fare : null,
    Embarked: row.Embarked ? row.Embarked.trim().toUpperCase() : null
  };
}

async function loadCSV(file, isTrain = true) {
  const text = await file.text();
  const rows = Papa.parse(text, { header: true }).data;
  const parsed = rows.map(parseCSVRow).filter(r => r.PassengerId != null);
  if (isTrain) trainData = parsed;
  else testData = parsed;
  alert(`${isTrain ? "Train" : "Test"} data loaded: ${parsed.length} rows`);
  showPreview(parsed, isTrain ? 'train-preview' : 'test-preview');
  if (isTrain) showCharts();
}

function showPreview(data, divId) {
  const div = document.getElementById(divId);
  div.innerHTML = '<h3>Preview</h3>';
  const table = document.createElement('table');
  table.border = "1";
  const keys = Object.keys(data[0]);
  const header = document.createElement('tr');
  keys.forEach(k => {
    const th = document.createElement('th');
    th.textContent = k;
    header.appendChild(th);
  });
  table.appendChild(header);
  data.slice(0, 10).forEach(row => {
    const tr = document.createElement('tr');
    keys.forEach(k => {
      const td = document.createElement('td');
      td.textContent = row[k];
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  div.appendChild(table);
}

// ====================== PREPROCESS ======================
function preprocess(data) {
  // Impute Age
  const ages = data.map(d => d.Age).filter(v => v != null);
  const medianAge = ages.sort((a,b)=>a-b)[Math.floor(ages.length/2)];
  data.forEach(d => { if (d.Age == null) d.Age = medianAge; });

  // Impute Fare
  const fares = data.map(d => d.Fare).filter(v => v != null);
  const medianFare = fares.sort((a,b)=>a-b)[Math.floor(fares.length/2)];
  data.forEach(d => { if (d.Fare == null) d.Fare = medianFare; });

  // Impute Embarked
  const embarkCounts = {};
  data.forEach(d => { if (d.Embarked) embarkCounts[d.Embarked] = (embarkCounts[d.Embarked]||0)+1; });
  const modeEmb = Object.entries(embarkCounts).sort((a,b)=>b[1]-a[1])[0][0];
  data.forEach(d => { if (!d.Embarked) d.Embarked = modeEmb; });

  // Feature engineering
  data.forEach(d => {
    d.FamilySize = d.SibSp + d.Parch + 1;
    d.IsAlone = d.FamilySize === 1 ? 1 : 0;
  });
  return data;
}

function standardize(arr) {
  const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
  const std = Math.sqrt(arr.map(x=>(x-mean)**2).reduce((a,b)=>a+b,0)/arr.length);
  return arr.map(x => std === 0 ? 0 : (x-mean)/std);
}

function encodeOneHot(values) {
  const uniq = [...new Set(values)];
  return values.map(v => uniq.map(u => v === u ? 1 : 0));
}

function prepareTensors(train, test) {
  train = preprocess(train);
  test = preprocess(test);

  const ageTrain = standardize(train.map(d=>d.Age));
  const fareTrain = standardize(train.map(d=>d.Fare));
  const sexTrain = encodeOneHot(train.map(d=>d.Sex));
  const pclassTrain = encodeOneHot(train.map(d=>d.Pclass));
  const embTrain = encodeOneHot(train.map(d=>d.Embarked));
  const famTrain = train.map(d=>d.FamilySize);
  const aloneTrain = train.map(d=>d.IsAlone);

  const xsArr = train.map((d,i) => [
    ageTrain[i], fareTrain[i],
    ...sexTrain[i], ...pclassTrain[i], ...embTrain[i],
    famTrain[i], aloneTrain[i]
  ]);
  const ysArr = train.map(d=>d.Survived);

  const xs = tf.tensor2d(xsArr);
  const ys = tf.tensor2d(ysArr, [ysArr.length,1]);

  // Test set (for inference)
  const ageTest = standardize(test.map(d=>d.Age));
  const fareTest = standardize(test.map(d=>d.Fare));
  const sexTest = encodeOneHot(test.map(d=>d.Sex));
  const pclassTest = encodeOneHot(test.map(d=>d.Pclass));
  const embTest = encodeOneHot(test.map(d=>d.Embarked));
  const famTest = test.map(d=>d.FamilySize);
  const aloneTest = test.map(d=>d.IsAlone);

  const testArr = test.map((d,i) => [
    ageTest[i], fareTest[i],
    ...sexTest[i], ...pclassTest[i], ...embTest[i],
    famTest[i], aloneTest[i]
  ]);

  return {xs, ys, testArr};
}

// ====================== CHARTS ======================
function showCharts() {
  const div = document.getElementById('charts');
  div.innerHTML = '<h3>Charts</h3>';

  // Survival by Sex
  const groupsSex = {};
  trainData.forEach(d=>{
    if (d.Sex && d.Survived!=null){
      if (!groupsSex[d.Sex]) groupsSex[d.Sex]={surv:0,total:0};
      groupsSex[d.Sex].total++;
      if (d.Survived===1) groupsSex[d.Sex].surv++;
    }
  });
  const sexChart = Object.keys(groupsSex).map(k=>({
    x:k, y:groupsSex[k].surv/groupsSex[k].total
  }));
  tfvis.render.barchart(div.appendChild(document.createElement('div')), sexChart, {
    xLabel:'Sex', yLabel:'Survival Rate', width:300, height:250
  });

  // Survival by Pclass
  const groupsClass = {};
  trainData.forEach(d=>{
    if (d.Pclass && d.Survived!=null){
      if (!groupsClass[d.Pclass]) groupsClass[d.Pclass]={surv:0,total:0};
      groupsClass[d.Pclass].total++;
      if (d.Survived===1) groupsClass[d.Pclass].surv++;
    }
  });
  const classChart = Object.keys(groupsClass).map(k=>({
    x:`Class ${k}`, y:groupsClass[k].surv/groupsClass[k].total
  }));
  tfvis.render.barchart(div.appendChild(document.createElement('div')), classChart, {
    xLabel:'Pclass', yLabel:'Survival Rate', width:300, height:250
  });
}

// ====================== MODEL ======================
function createModel(inputDim) {
  const m = tf.sequential();
  m.add(tf.layers.dense({units:16, activation:'relu', inputShape:[inputDim]}));
  m.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  m.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});
  m.summary();
  return m;
}

// ====================== TRAIN ======================
async function train() {
  preprocessed = prepareTensors(trainData, testData);
  const {xs, ys} = preprocessed;
  const inputDim = xs.shape[1];
  model = createModel(inputDim);

  // Split train/val
  const idx = Math.floor(xs.shape[0]*0.8);
  const xTrain = xs.slice([0,0],[idx,inputDim]);
  const yTrain = ys.slice([0,0],[idx,1]);
  const xVal = xs.slice([idx,0],[xs.shape[0]-idx,inputDim]);
  const yVal = ys.slice([idx,0],[ys.shape[0]-idx,1]);

  const history = await model.fit(xTrain, yTrain, {
    epochs:50, batchSize:32, validationData:[xVal,yVal],
    callbacks: tfvis.show.fitCallbacks(
      {name:'Training Performance', tab:'Training'},
      ['loss','val_loss','acc','val_acc'],
      {callbacks:['onEpochEnd']}
    )
  });

  valPredProbs = model.predict(xVal);
  valLabels = yVal;
  alert("Training finished");
}

// ====================== METRICS ======================
function evaluate(threshold=0.5) {
  if (!valPredProbs) { alert("Train model first"); return; }
  const probs = valPredProbs.dataSync();
  const labels = valLabels.dataSync();

  let tp=0, tn=0, fp=0, fn=0;
  for (let i=0;i<probs.length;i++){
    const pred = probs[i]>threshold?1:0;
    if (pred===1 && labels[i]===1) tp++;
    else if (pred===0 && labels[i]===0) tn++;
    else if (pred===1 && labels[i]===0) fp++;
    else if (pred===0 && labels[i]===1) fn++;
  }
  const precision = tp/(tp+fp+1e-7);
  const recall = tp/(tp+fn+1e-7);
  const f1 = 2*precision*recall/(precision+recall+1e-7);

  const div = document.getElementById('metrics');
  div.innerHTML = `
    <h3>Metrics (Threshold=${threshold})</h3>
    <p>TP=${tp}, TN=${tn}, FP=${fp}, FN=${fn}</p>
    <p>Precision=${precision.toFixed(3)}, Recall=${recall.toFixed(3)}, F1=${f1.toFixed(3)}</p>
  `;

  // ROC Curve
  const rocPoints=[];
  for (let t=0;t<=1;t+=0.05){
    let TP=0, FP=0, TN=0, FN=0;
    for (let i=0;i<probs.length;i++){
      const p = probs[i]>t?1:0;
      if (p===1 && labels[i]===1) TP++;
      if (p===0 && labels[i]===0) TN++;
      if (p===1 && labels[i]===0) FP++;
      if (p===0 && labels[i]===1) FN++;
    }
    const tpr = TP/(TP+FN+1e-7);
    const fpr = FP/(FP+TN+1e-7);
    rocPoints.push({x:fpr, y:tpr});
  }
  div.appendChild(document.createElement('div'));
  tfvis.render.scatterplot(div.lastChild,{values:rocPoints},{xLabel:'FPR', yLabel:'TPR'});
}

// ====================== PREDICT & EXPORT ======================
function predictAndExport(threshold=0.5) {
  const arr = preprocessed.testArr;
  const xs = tf.tensor2d(arr);
  const probs = model.predict(xs).dataSync();
  const results = testData.map((d,i)=>({
    PassengerId:d.PassengerId,
    Survived: probs[i]>threshold?1:0,
    Probability: probs[i]
  }));

  // Submission CSV
  let sub="PassengerId,Survived\n";
  results.forEach(r=> sub+=`${r.PassengerId},${r.Survived}\n`);
  downloadCSV(sub,"submission.csv");

  // Probabilities CSV
  let prob="PassengerId,Probability\n";
  results.forEach(r=> prob+=`${r.PassengerId},${r.Probability}\n`);
  downloadCSV(prob,"probabilities.csv");

  // Save model
  model.save('downloads://titanic-tfjs');
}

function downloadCSV(content, filename){
  const blob = new Blob([content],{type:"text/csv"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href=url; a.download=filename; a.click();
  URL.revokeObjectURL(url);
}
