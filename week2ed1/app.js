// =========================
// Global variables
// =========================
let trainData = [];
let testData = [];
let model;
let stopFlag = false;
let predictions = [];

// =========================
// CSV Loader (FileReader)
// =========================
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map(line => {
    const cols = line.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/); 
    const row = {};
    headers.forEach((h, i) => {
      row[h.trim()] = cols[i] ? cols[i].replace(/"/g, "").trim() : "";
    });
    return row;
  });
}

function loadData() {
  const trainFile = document.getElementById("trainFile").files[0];
  const testFile = document.getElementById("testFile").files[0];

  if (!trainFile || !testFile) {
    alert("Please upload both train.csv and test.csv files!");
    return;
  }

  const readerTrain = new FileReader();
  readerTrain.onload = e => {
    trainData = parseCSV(e.target.result).map(cleanRow);
    console.log("Train loaded:", trainData.length);
    showPreview(trainData, "dataPreview", "Train Data (first 5 rows)");
  };
  readerTrain.readAsText(trainFile);

  const readerTest = new FileReader();
  readerTest.onload = e => {
    testData = parseCSV(e.target.result).map(cleanRow);
    console.log("Test loaded:", testData.length);
    showPreview(testData, "dataPreview", "Test Data (first 5 rows)");
  };
  readerTest.readAsText(testFile);
}

// =========================
// Helpers
// =========================
function showPreview(data, elementId, title) {
  const preview = document.getElementById(elementId);
  preview.innerHTML = `<h3 class="font-bold text-pink-600">${title}</h3>`;

  let html = "<table><thead><tr>";
  Object.keys(data[0]).forEach(k => html += `<th>${k}</th>`);
  html += "</tr></thead><tbody>";
  data.slice(0, 5).forEach(r => {
    html += "<tr>" + Object.values(r).map(v => `<td>${v}</td>`).join("") + "</tr>";
  });
  html += "</tbody></table>";
  preview.innerHTML += html;
}

function cleanRow(row) {
  return {
    PassengerId: +row.PassengerId,
    Survived: row.Survived !== "" ? +row.Survived : null,
    Pclass: row.Pclass !== "" ? +row.Pclass : null,
    Sex: row.Sex ? row.Sex.trim().toLowerCase() : null,
    Age: row.Age !== "" ? +row.Age : null,
    SibSp: row.SibSp !== "" ? +row.SibSp : 0,
    Parch: row.Parch !== "" ? +row.Parch : 0,
    Fare: row.Fare !== "" ? +row.Fare : null,
    Embarked: row.Embarked ? row.Embarked.trim().toUpperCase() : null
  };
}

// =========================
// Preprocessing
// =========================
function encodeFeatures(data) {
  return data.map(r => [
    r.Pclass ?? 0,
    r.Sex === "male" ? 1 : 0,
    r.Age ?? 0,
    r.SibSp ?? 0,
    r.Parch ?? 0,
    r.Fare ?? 0
  ]);
}

function preprocess() {
  if (trainData.length === 0) {
    alert("Load data first!");
    return;
  }

  const ages = trainData.map(d => d.Age).filter(v => v != null);
  const medianAge = ages.sort((a,b)=>a-b)[Math.floor(ages.length/2)];
  trainData.forEach(d => { if (d.Age == null) d.Age = medianAge; });

  const fares = trainData.map(d => d.Fare).filter(v => v != null);
  const medianFare = fares.sort((a,b)=>a-b)[Math.floor(fares.length/2)];
  trainData.forEach(d => { if (d.Fare == null) d.Fare = medianFare; });

  const embarkCounts = {};
  trainData.forEach(d => { if (d.Embarked) embarkCounts[d.Embarked] = (embarkCounts[d.Embarked]||0)+1; });
  const modeEmbarked = Object.entries(embarkCounts).sort((a,b)=>b[1]-a[1])[0][0];
  trainData.forEach(d => { if (!d.Embarked) d.Embarked = modeEmbarked; });

  document.getElementById("preprocessInfo").innerText = 
    "✅ Preprocessing done!";
}

// =========================
// Model Creation
// =========================
function createModel() {
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [6] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  const summary = [];
  model.layers.forEach(l => summary.push(l.getConfig()));
  document.getElementById("modelSummary").innerText = JSON.stringify(summary, null, 2);
}

// =========================
// Training
// =========================
async function trainModel() {
  if (!model) { alert("Create model first!"); return; }

  stopFlag = false;

  const xs = tf.tensor2d(encodeFeatures(trainData));
  const ys = tf.tensor2d(trainData.map(r => [r.Survived]));

  await model.fit(xs, ys, {
    epochs: 30,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: [
      tfvis.show.fitCallbacks({ name: 'Training Performance', tab: 'Training' }, ['loss', 'acc'], { height: 200 }),
      {
        onEpochEnd: async () => { if (stopFlag) model.stopTraining = true; }
      }
    ]
  });

  xs.dispose(); ys.dispose();
}

function stopTraining() {
  stopFlag = true;
  alert("⛔ Training will stop after current epoch!");
}

// =========================
// Metrics & ROC
// =========================
function computeMetrics() {
  if (!model) { alert("Train model first!"); return; }

  const xs = tf.tensor2d(encodeFeatures(trainData));
  const ysTrue = trainData.map(r => r.Survived);

  const probs = model.predict(xs).arraySync().map(v => v[0]);
  const threshold = parseFloat(document.getElementById("threshold").value);
  const preds = probs.map(p => p > threshold ? 1 : 0);

  let TP=0, FP=0, TN=0, FN=0;
  preds.forEach((p,i) => {
    if (p===1 && ysTrue[i]===1) TP++;
    else if (p===1 && ysTrue[i]===0) FP++;
    else if (p===0 && ysTrue[i]===0) TN++;
    else FN++;
  });

  const acc = (TP+TN)/(TP+TN+FP+FN);
  const precision = TP/(TP+FP+1e-5);
  const recall = TP/(TP+FN+1e-5);
  const f1 = 2*precision*recall/(precision+recall+1e-5);

  document.getElementById("confMatrix").innerHTML =
    `<b>Confusion Matrix</b><br>TP:${TP}, FP:${FP}, TN:${TN}, FN:${FN}
     <br>Acc:${acc.toFixed(3)}, Prec:${precision.toFixed(3)}, Rec:${recall.toFixed(3)}, F1:${f1.toFixed(3)}`;

  // ROC Curve
  const rocPoints = [];
  for (let t=0; t<=1; t+=0.05) {
    let tp=0, fp=0, tn=0, fn=0;
    probs.forEach((p,i)=>{
      const pred = p > t ? 1 : 0;
      if (pred===1 && ysTrue[i]===1) tp++;
      else if (pred===1 && ysTrue[i]===0) fp++;
      else if (pred===0 && ysTrue[i]===0) tn++;
      else fn++;
    });
    const TPR = tp/(tp+fn+1e-5);
    const FPR = fp/(fp+tn+1e-5);
    rocPoints.push({x:FPR, y:TPR});
  }
  tfvis.render.linechart({ name:"ROC Curve", tab:"Metrics" }, { values:[rocPoints], series:["ROC"] }, 
    { xLabel:"False Positive Rate", yLabel:"True Positive Rate" });

  xs.dispose();
}

// =========================
// Prediction & Export
// =========================
function predictTest() {
  if (!model || testData.length === 0) {
    alert("Load model and test data first!");
    return;
  }

  const xs = tf.tensor2d(encodeFeatures(testData));
  const probs = model.predict(xs).arraySync().map(v => v[0]);
  const threshold = parseFloat(document.getElementById("threshold").value);

  predictions = testData.map((r,i)=>({
    PassengerId: r.PassengerId,
    Survived: probs[i] > threshold ? 1 : 0
  }));

  showPreview(predictions, "predictionPreview", "Prediction Results (first 5 rows)");
  xs.dispose();
}

function exportResults() {
  if (predictions.length === 0) {
    alert("No predictions to export!");
    return;
  }

  let csv = "PassengerId,Survived\n";
  predictions.forEach(r => {
    csv += `${r.PassengerId},${r.Survived}\n`;
  });

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "submission.csv";
  a.click();
  URL.revokeObjectURL(url);
}
