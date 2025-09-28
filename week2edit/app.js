let trainData = [];
let testData = [];
let processedTrain = [];
let processedTest = [];
let model = null;
let threshold = 0.5;

// ================== Load Data ==================
function loadData() {
  const trainFile = document.getElementById('trainFile').files[0];
  const testFile = document.getElementById('testFile').files[0];
  if (!trainFile || !testFile) {
    alert("Select both train and test CSV!");
    return;
  }

  Papa.parse(trainFile, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    quoteChar: '"',       // ✅ fix comma-escape
    escapeChar: '"',
    complete: res => {
      trainData = res.data;
      document.getElementById('loadStatus').innerText = `Train loaded: ${trainData.length} rows.`;
      checkLoaded();
    }
  });

  Papa.parse(testFile, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    quoteChar: '"',
    escapeChar: '"',
    complete: res => {
      testData = res.data;
      document.getElementById('loadStatus').innerText += ` Test loaded: ${testData.length} rows.`;
      checkLoaded();
    }
  });
}

function checkLoaded() {
  if (trainData.length && testData.length) {
    document.getElementById('inspectBtn').disabled = false;
  }
}

// ================== Inspect Data ==================
function inspectData() {
  const head = trainData.slice(0, 5);
  let html = "<table><tr>";
  Object.keys(head[0]).forEach(k => html += `<th>${k}</th>`);
  html += "</tr>";
  head.forEach(r => {
    html += "<tr>";
    Object.values(r).forEach(v => html += `<td>${v}</td>`);
    html += "</tr>";
  });
  html += "</table>";
  document.getElementById('preview').innerHTML = html;

  // Chart survival by Sex
  const maleSurvived = trainData.filter(d => d.Sex === 'male' && d.Survived === 1).length;
  const maleDied = trainData.filter(d => d.Sex === 'male' && d.Survived === 0).length;
  const femaleSurvived = trainData.filter(d => d.Sex === 'female' && d.Survived === 1).length;
  const femaleDied = trainData.filter(d => d.Sex === 'female' && d.Survived === 0).length;

  const chartData = [
    { index: 'Male Survived', value: maleSurvived },
    { index: 'Male Died', value: maleDied },
    { index: 'Female Survived', value: femaleSurvived },
    { index: 'Female Died', value: femaleDied }
  ];

  tfvis.render.barchart({ name: 'Survival by Sex', tab: 'Charts' }, chartData);

  document.getElementById('preprocessBtn').disabled = false;
}

// ================== Preprocess ==================
function preprocessData() {
  const addFamily = document.getElementById('addFamily').checked;

  // Compute median/mean/std
  const ages = trainData.map(r => r.Age).filter(a => !isNaN(a));
  const fares = trainData.map(r => r.Fare).filter(f => !isNaN(f));
  const median = arr => { const s = arr.slice().sort((a,b)=>a-b); return s[Math.floor(s.length/2)]; };
  const mean = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
  const std = (arr,m) => Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/arr.length);

  const mAge = median(ages), mFare = median(fares);
  const meanAge = mean(ages), stdAge = std(ages,meanAge);
  const meanFare = mean(fares), stdFare = std(fares,meanFare);

  function transform(row,isTrain){
    const age = isNaN(row.Age) ? mAge : row.Age;
    const fare = isNaN(row.Fare) ? mFare : row.Fare;
    const feat = [
      row.Pclass,
      row.Sex === 'female' ? 1 : 0,
      (age-meanAge)/stdAge,
      row.SibSp,
      row.Parch,
      (fare-meanFare)/stdFare,
      row.Embarked==='C'?1:row.Embarked==='Q'?2:0
    ];
    if(addFamily){
      const familySize = row.SibSp + row.Parch + 1;
      const isAlone = familySize===1?1:0;
      feat.push(familySize);
      feat.push(isAlone);
    }
    return {x:feat, y: isTrain?row.Survived:null};
  }

  processedTrain = trainData.map(r=>transform(r,true));
  processedTest = testData.map(r=>transform(r,false));

  document.getElementById('prepStatus').innerText =
    `Preprocessed. Feature length: ${processedTrain[0].x.length}`;
  document.getElementById('createModelBtn').disabled = false;
}

// ================== Create Model ==================
function createModel(){
  const inputDim = processedTrain[0].x.length;
  model = tf.sequential();
  model.add(tf.layers.dense({units:16,activation:'relu',inputShape:[inputDim]}));
  model.add(tf.layers.dense({units:1,activation:'sigmoid'}));
  model.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});

  tfvis.show.modelSummary({name:'Model Summary',tab:'Model'},model);
  document.getElementById('trainBtn').disabled = false;
}

// ================== Train Model ==================
async function trainModel(){
  const xs = tf.tensor2d(processedTrain.map(r=>r.x));
  const ys = tf.tensor2d(processedTrain.map(r=>[r.y]));

  const callbacks = tfvis.fitCallbacks(
    {name:'Training',tab:'Model'},
    ['loss','val_loss','acc','val_acc'],
    {callbacks:['onEpochEnd']}   // ✅ fix error
  );

  await model.fit(xs,ys,{
    epochs:50,
    batchSize:32,
    validationSplit:0.2,
    callbacks
  });

  document.getElementById('trainStatus').innerText = "Training complete!";
  document.getElementById('threshold').disabled = false;
  document.getElementById('predictBtn').disabled = true; // enable after threshold adjust
  document.getElementById('predictBtn').disabled = false;
}

// ================== Threshold ==================
function updateThreshold(){
  threshold = parseFloat(document.getElementById('threshold').value);
  document.getElementById('thresholdVal').innerText = threshold.toFixed(2);
}

// ================== Predict ==================
function predict(){
  const xs = tf.tensor2d(processedTest.map(r=>r.x));
  const probs = model.predict(xs).dataSync();
  testData.forEach((r,i)=>{ r.Survived = probs[i] > threshold ? 1:0; });

  document.getElementById('predictStatus').innerText = "Prediction done.";
  document.getElementById('exportBtn').disabled = false;
}

// ================== Export ==================
function exportResults(){
  let csv = "PassengerId,Survived\n";
  testData.forEach(r => csv += `${r.PassengerId},${r.Survived}\n`);
  const blob = new Blob([csv],{type:'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'submission.csv';
  a.click();
}
