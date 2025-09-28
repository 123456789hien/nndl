let rawDataset = null;
let model = null;
let preprocessedTrain = null;
let valLabels = null;
let valProbs = null;

async function loadSampleData() {
const response = await fetch("[https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)");
const text = await response.text();
rawDataset = text;
document.getElementById("dataset-status").textContent = "Sample Titanic dataset loaded.";
}

function preprocessData() {
if (!rawDataset) {
alert("Load dataset first.");
return;
}
const lines = rawDataset.split("\n").slice(1);
const features = [];
const labels = [];
lines.forEach(line => {
if (line.trim().length === 0) return;
const parts = line.split(",");
const survived = parseInt(parts[1]);
const pclass = parseInt(parts[2]);
const sex = parts[5] === "male" ? 0 : 1;
const age = parts[6] ? parseFloat(parts[6]) : 30;
const sibsp = parseInt(parts[7]);
const parch = parseInt(parts[8]);
const fare = parts[10] ? parseFloat(parts[10]) : 32.2;
features.push([pclass, sex, age, sibsp, parch, fare]);
labels.push(survived);
});
const xs = tf.tensor2d(features);
const ys = tf.tensor2d(labels, [labels.length, 1]);
preprocessedTrain = { features: xs, labels: ys };
document.getElementById("dataset-status").textContent = "Data preprocessed. Rows: " + labels.length;
}

function createModel() {
model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [6], units: 16, activation: "relu" }));
model.add(tf.layers.dense({ units: 8, activation: "relu" }));
model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });
document.getElementById("model-summary").textContent = "Model created.";
}

async function trainModel() {
if (!model || !preprocessedTrain) {
alert("Create model and preprocess data first.");
return;
}
document.getElementById("training-status").textContent = "Training...";
const X = preprocessedTrain.features;
const y = preprocessedTrain.labels;
const fitCallbacks = tfvis.show.fitCallbacks(
{ name: "Training Performance", tab: "Training" },
["loss", "acc", "val_loss", "val_acc"],
{ callbacks: ["onEpochEnd"] }
);
await model.fit(X, y, {
epochs: 20,
batchSize: 32,
validationSplit: 0.2,
shuffle: true,
callbacks: fitCallbacks
});
document.getElementById("training-status").textContent = "Training finished.";
document.getElementById("predict-btn").disabled = false;
document.getElementById("export-btn").disabled = false;
document.getElementById("threshold-slider").disabled = false;
}

async function evaluateModel() {
if (!model || !preprocessedTrain) {
alert("Train model first.");
return;
}
const result = await model.evaluate(preprocessedTrain.features, preprocessedTrain.labels);
const loss = result[0].dataSync()[0].toFixed(4);
const acc = result[1].dataSync()[0].toFixed(4);
document.getElementById("evaluation-results").textContent = "Loss: " + loss + ", Accuracy: " + acc;
}

function predictPassenger() {
const age = parseFloat(document.getElementById("age").value);
const sex = document.getElementById("sex").value === "male" ? 0 : 1;
const pclass = parseInt(document.getElementById("pclass").value);
const sibsp = parseInt(document.getElementById("sibsp").value);
const parch = parseInt(document.getElementById("parch").value);
const fare = parseFloat(document.getElementById("fare").value);
const input = tf.tensor2d([[pclass, sex, age, sibsp, parch, fare]]);
const prob = model.predict(input).dataSync()[0];
const threshold = parseFloat(document.getElementById("threshold-slider").value);
const prediction = prob >= threshold ? "Survived" : "Did not survive";
document.getElementById("predict-result").textContent = "Probability: " + prob.toFixed(4) + " → " + prediction;
}

function exportModel() {
if (!model) return;
model.save("downloads://titanic-model");
}

document.getElementById("threshold-slider").addEventListener("input", e => {
document.getElementById("threshold-value").textContent = e.target.value;
});
