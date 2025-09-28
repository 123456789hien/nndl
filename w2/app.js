/**

* Titanic Survival Classifier - TensorFlow.js
* Runs fully in browser (no server). Works with GitHub Pages.
*
* Schema:
* Target: Survived (0/1)
* Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
* Identifier: PassengerId (exclude)
*
* Reuse note: swap schema here for other datasets.
  */

let trainCSV = null;
let testCSV = null;
let model = null;
let trainTensors = {};
let valLabels = null;
let valProbs = null;
let testDf = null;

// -------------------- Data Load --------------------
async function loadSampleData() {
try {
const trainResp = await fetch("[https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)");
const trainText = await trainResp.text();
trainCSV = trainText;
document.getElementById("data-status").textContent = "Sample train.csv loaded.";
showPreview(trainText, "data-preview");
} catch (err) {
alert("Error loading sample data: " + err.message);
}
}

function handleFileUpload(event, type) {
const file = event.target.files[0];
if (!file) return;
const reader = new FileReader();
reader.onload = e => {
if (type === "train") {
trainCSV = e.target.result;
document.getElementById("data-status").textContent = "Train CSV loaded.";
showPreview(trainCSV, "data-preview");
} else {
testCSV = e.target.result;
document.getElementById("data-status").textContent = "Test CSV loaded.";
}
};
reader.readAsText(file);
}

document.getElementById("train-upload").addEventListener("change", e => handleFileUpload(e, "train"));
document.getElementById("test-upload").addEventListener("change", e => handleFileUpload(e, "test"));

// Show preview table
function showPreview(csvText, divId) {
const lines = csvText.split("\n").slice(0, 6);
let html = "<table><tr>";
const headers = lines[0].split(",");
headers.forEach(h => (html += `<th>${h}</th>`));
html += "</tr>";
for (let i = 1; i < lines.length; i++) {
if (!lines[i]) continue;
html += "<tr>";
const row = lines[i].split(",");
row.forEach(c => (html += `<td>${c}</td>`));
html += "</tr>";
}
html += "</table>";
document.getElementById(divId).innerHTML = html;
}

// -------------------- Preprocessing --------------------
function preprocessData() {
if (!trainCSV) {
alert("Please load train.csv first.");
return;
}

const rows = trainCSV.split("\n").map(r => r.split(","));
const headers = rows[0];
const dataRows = rows.slice(1).filter(r => r.length === headers.length);

const colIdx = name => headers.indexOf(name);

const features = [];
const labels = [];

const ages = dataRows.map(r => parseFloat(r[colIdx("Age")]) || NaN).filter(v => !isNaN(v));
const ageMedian = ages.sort((a, b) => a - b)[Math.floor(ages.length / 2)];
const fares = dataRows.map(r => parseFloat(r[colIdx("Fare")]) || NaN).filter(v => !isNaN(v));
const fareMean = fares.reduce((a, b) => a + b, 0) / fares.length;

const embarkedValues = dataRows.map(r => r[colIdx("Embarked")] || "");
const embarkedMode = embarkedValues.sort((a, b) =>
embarkedValues.filter(v => v === a).length - embarkedValues.filter(v => v === b).length
).pop();

dataRows.forEach(r => {
const survived = parseInt(r[colIdx("Survived")]);
const pclass = parseInt(r[colIdx("Pclass")]);
const sex = r[colIdx("Sex")] === "male" ? 0 : 1;
const age = parseFloat(r[colIdx("Age")]) || ageMedian;
const sibsp = parseInt(r[colIdx("SibSp")]);
const parch = parseInt(r[colIdx("Parch")]);
const fare = parseFloat(r[colIdx("Fare")]) || fareMean;
const embarked = r[colIdx("Embarked")] || embarkedMode;

```
// Standardize age/fare
const ageStd = (age - ageMedian) / ageMedian;
const fareStd = (fare - fareMean) / fareMean;

// One-hot Pclass (1-3), Sex, Embarked (C,Q,S)
const pclassOH = [0, 0, 0]; pclassOH[pclass - 1] = 1;
const sexOH = [sex];
const embarkedOH = [embarked === "C" ? 1 : 0, embarked === "Q" ? 1 : 0, embarked === "S" ? 1 : 0];

// Extra features
const familySize = sibsp + parch + 1;
const isAlone = familySize === 1 ? 1 : 0;

const feat = [...pclassOH, ...sexOH, ageStd, sibsp, parch, fareStd, ...embarkedOH, familySize, isAlone];
features.push(feat);
labels.push(survived);
```

});

const xs = tf.tensor2d(features);
const ys = tf.tensor2d(labels, [labels.length, 1]);

// Train/val split (80/20)
const idxs = [...Array(labels.length).keys()];
tf.util.shuffle(idxs);
const split = Math.floor(0.8 * idxs.length);
const trainIdxs = idxs.slice(0, split);
const valIdxs = idxs.slice(split);

const xTrain = tf.gather(xs, tf.tensor1d(trainIdxs, "int32"));
const yTrain = tf.gather(ys, tf.tensor1d(trainIdxs, "int32"));
const xVal = tf.gather(xs, tf.tensor1d(valIdxs, "int32"));
const yVal = tf.gather(ys, tf.tensor1d(valIdxs, "int32"));

trainTensors = { xTrain, yTrain, xVal, yVal };

document.getElementById("preprocess-status").textContent =
`Preprocessed: ${features.length} rows, ${features[0].length} features.`;
}

// -------------------- Model --------------------
function createModel() {
model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [trainTensors.xTrain.shape[1]], units: 16, activation: "relu" }));
model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });
document.getElementById("model-summary").textContent = "Model created with 1 hidden layer.";
}

// -------------------- Training --------------------
async function trainModel() {
if (!model || !trainTensors.xTrain) {
alert("Preprocess and create model first.");
return;
}

const fitCallbacks = tfvis.show.fitCallbacks(
{ name: "Training Performance", tab: "Training" },
["loss", "acc", "val_loss", "val_acc"],
{ callbacks: ["onEpochEnd"] }
);

const earlyStopping = tf.callbacks.earlyStopping({ monitor: "val_loss", patience: 5 });

await model.fit(trainTensors.xTrain, trainTensors.yTrain, {
epochs: 50,
batchSize: 32,
validationData: [trainTensors.xVal, trainTensors.yVal],
callbacks: [fitCallbacks, earlyStopping]
});

// Store val preds
const preds = model.predict(trainTensors.xVal);
valProbs = Array.from(preds.dataSync());
valLabels = Array.from(trainTensors.yVal.dataSync());

document.getElementById("train-status").textContent = "Training complete.";
document.getElementById("threshold-slider").disabled = false;

plotROC();
updateMetrics(0.5);
}

// -------------------- Metrics --------------------
function plotROC() {
const thresholds = [...Array(101).keys()].map(i => i / 100);
const tpr = [], fpr = [];

thresholds.forEach(thresh => {
let tp = 0, fp = 0, tn = 0, fn = 0;
valProbs.forEach((p, i) => {
const pred = p >= thresh ? 1 : 0;
const actual = valLabels[i];
if (pred === 1 && actual === 1) tp++;
else if (pred === 1 && actual === 0) fp++;
else if (pred === 0 && actual === 0) tn++;
else if (pred === 0 && actual === 1) fn++;
});
tpr.push(tp / (tp + fn));
fpr.push(fp / (fp + tn));
});

const ctx = document.getElementById("roc-curve").getContext("2d");
ctx.clearRect(0, 0, 400, 400);
ctx.beginPath();
ctx.moveTo(0, 400);
for (let i = 0; i < tpr.length; i++) {
ctx.lineTo(fpr[i] * 400, 400 - tpr[i] * 400);
}
ctx.strokeStyle = "blue";
ctx.stroke();
}

function updateMetrics(threshold) {
let tp = 0, fp = 0, tn = 0, fn = 0;
valProbs.forEach((p, i) => {
const pred = p >= threshold ? 1 : 0;
const actual = valLabels[i];
if (pred === 1 && actual === 1) tp++;
else if (pred === 1 && actual === 0) fp++;
else if (pred === 0 && actual === 0) tn++;
else if (pred === 0 && actual === 1) fn++;
});
const precision = tp / (tp + fp + 1e-6);
const recall = tp / (tp + fn + 1e-6);
const f1 = 2 * (precision * recall) / (precision + recall + 1e-6);
document.getElementById("metrics").textContent =
`TP:${tp} FP:${fp} TN:${tn} FN:${fn} | Precision:${precision.toFixed(2)} Recall:${recall.toFixed(2)} F1:${f1.toFixed(2)}`;
}

document.getElementById("threshold-slider").addEventListener("input", e => {
const val = parseFloat(e.target.value);
document.getElementById("threshold-value").textContent = val.toFixed(2);
updateMetrics(val);
});

// -------------------- Inference & Export --------------------
function predictTest() {
if (!testCSV || !model) {
alert("Load test.csv and train model first.");
return;
}
const rows = testCSV.split("\n").map(r => r.split(","));
const headers = rows[0];
const dataRows = rows.slice(1).filter(r => r.length === headers.length);

const colIdx = name => headers.indexOf(name);
const preds = [];
const probs = [];

dataRows.forEach(r => {
if (!r[colIdx("PassengerId")]) return;
const pid = r[colIdx("PassengerId")];
const pclass = parseInt(r[colIdx("Pclass")]);
const sex = r[colIdx("Sex")] === "male" ? 0 : 1;
const age = parseFloat(r[colIdx("Age")]) || 30;
const sibsp = parseInt(r[colIdx("SibSp")]);
const parch = parseInt(r[colIdx("Parch")]);
const fare = parseFloat(r[colIdx("Fare")]) || 32.2;
const embarked = r[colIdx("Embarked")] || "S";

```
const ageStd = (age - 30) / 30;
const fareStd = (fare - 32.2) / 32.2;

const pclassOH = [0, 0, 0]; pclassOH[pclass - 1] = 1;
const sexOH = [sex];
const embarkedOH = [embarked === "C" ? 1 : 0, embarked === "Q" ? 1 : 0, embarked === "S" ? 1 : 0];
const familySize = sibsp + parch + 1;
const isAlone = familySize === 1 ? 1 : 0;

const feat = [...pclassOH, ...sexOH, ageStd, sibsp, parch, fareStd, ...embarkedOH, familySize, isAlone];
const input = tf.tensor2d([feat]);
const prob = model.predict(input).dataSync()[0];
const threshold = parseFloat(document.getElementById("threshold-slider").value);
const pred = prob >= threshold ? 1 : 0;
preds.push({ PassengerId: pid, Survived: pred });
probs.push({ PassengerId: pid, Probability: prob });
```

});

downloadCSV(preds, "submission.csv");
downloadCSV(probs, "probabilities.csv");
document.getElementById("predict-status").textContent = "Predictions saved (submission.csv, probabilities.csv).";
}

function downloadCSV(data, filename) {
const rows = [Object.keys(data[0]).join(",")];
data.forEach(obj => {
rows.push(Object.values(obj).join(","));
});
const blob = new Blob([rows.join("\n")],
