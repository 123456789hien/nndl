// app.js

let trainData, testData;
let cnnModel, autoencoder;
let classNames = [...Array(10).keys()].map(String);

const statusEl = document.getElementById("data-status");
const logsEl = document.getElementById("training-logs");
const previewRow = document.getElementById("preview-row");
const modelInfo = document.getElementById("model-info");

function logStatus(msg) {
  statusEl.textContent = msg;
}
function logTrain(msg) {
  logsEl.textContent += msg + "\n";
  logsEl.scrollTop = logsEl.scrollHeight;
}

// ---------- Model definitions ----------
function createCNN() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

function createAutoencoder() {
  const input = tf.input({ shape: [28, 28, 1] });
  let x = tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(input);
  x = tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }).apply(x);
  x = tf.layers.conv2d({ filters: 8, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);
  x = tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }).apply(x);

  x = tf.layers.conv2d({ filters: 8, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);
  x = tf.layers.upSampling2d(2).apply(x);
  x = tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);
  x = tf.layers.upSampling2d(2).apply(x);
  const decoded = tf.layers.conv2d({ filters: 1, kernelSize: 3, padding: 'same', activation: 'sigmoid' }).apply(x);

  const model = tf.model({ inputs: input, outputs: decoded });
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });
  return model;
}

// ---------- Button actions ----------
async function onLoadData() {
  logStatus("Loading data...");
  trainData = await loadCsvFromInput(document.getElementById("train-csv"));
  testData = await loadCsvFromInput(document.getElementById("test-csv"));
  logStatus(`Train: ${trainData.images.shape}, Test: ${testData.images.shape}`);
}

async function onTrainCNN() {
  if (!trainData) return alert("Please load data first!");
  cnnModel = createCNN();
  cnnModel.summary();
  modelInfo.textContent = "";
  cnnModel.summary(null, null, x => (modelInfo.textContent += x + "\n"));

  const BATCH_SIZE = 128;
  const EPOCHS = 5;

  await cnnModel.fit(trainData.images, trainData.labels, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
    validationSplit: 0.15,
    callbacks: tfvis.show.fitCallbacks(
      { name: "CNN Training", tab: "Training" },
      ["loss", "val_loss", "acc", "val_acc"],
      { callbacks: ["onEpochEnd"] }
    )
  });
  logTrain("Training CNN finished!");
}

async function onTrainDenoiser() {
  if (!trainData) return alert("Please load data first!");
  autoencoder = createAutoencoder();
  autoencoder.summary();
  modelInfo.textContent = "";
  autoencoder.summary(null, null, x => (modelInfo.textContent += x + "\n"));

  // add random noise
  const noise = tf.randomNormal(trainData.images.shape, 0, 0.5);
  const noisyTrain = trainData.images.add(noise).clipByValue(0, 1);

  await autoencoder.fit(noisyTrain, trainData.images, {
    batchSize: 128,
    epochs: 5,
    validationSplit: 0.15,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Autoencoder Training", tab: "Training" },
      ["loss", "val_loss"],
      { callbacks: ["onEpochEnd"] }
    )
  });
  logTrain("Training Autoencoder finished!");
}

async function onEvaluate() {
  if (!cnnModel || !testData) return alert("Need CNN and test data!");
  const evalOutput = cnnModel.evaluate(testData.images, testData.labels);
  const testLoss = await evalOutput[0].data();
  const testAcc = await evalOutput[1].data();

  document.getElementById("metrics").textContent =
    `Test Loss: ${testLoss[0].toFixed(4)}, Test Acc: ${testAcc[0].toFixed(4)}`;

  // Compute confusion matrix
  const preds = cnnModel.predict(testData.images).argMax(-1);
  const labels = testData.labels.argMax(-1);
  const cm = await tfvis.metrics.confusionMatrix(labels, preds);

  // Per-class accuracy
  const perClassAcc = cm.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return { label: String(i), value: row[i] / total };
  });
  tfvis.render.barChart(
    { name: "Per-class accuracy", tab: "Evaluation" },
    perClassAcc.map(x => ({ index: x.label, value: x.value }))
  );

  preds.dispose();
  labels.dispose();
}

async function onTestFive() {
  if (!cnnModel || !testData) return alert("Need CNN and test data!");
  previewRow.innerHTML = "";
  const N = testData.images.shape[0];
  for (let i = 0; i < 5; i++) {
    const idx = Math.floor(Math.random() * N);
    const x = testData.images.slice([idx, 0, 0, 0], [1, 28, 28, 1]);
    const yTrue = testData.labels.argMax(-1).arraySync()[idx];
    const yPred = cnnModel.predict(x).argMax(-1).arraySync()[0];

    const canvas = document.createElement("canvas");
    await tf.browser.toPixels(x.reshape([28, 28]), canvas);
    const div = document.createElement("div");
    div.className = "preview-item";
    const label = document.createElement("div");
    label.innerHTML =
      `Pred: <span class="${yTrue === yPred ? "correct" : "wrong"}">${yPred}</span><br/>True: ${yTrue}`;
    div.appendChild(canvas);
    div.appendChild(label);
    previewRow.appendChild(div);

    x.dispose();
  }
}

async function onSaveModel() {
  if (!cnnModel) return alert("Train model first!");
  await cnnModel.save("downloads://mnist-cnn");
}

async function onLoadModel() {
  const jsonFile = document.getElementById("upload-json").files[0];
  const weightsFile = document.getElementById("upload-weights").files[0];
  if (!jsonFile || !weightsFile) return alert("Please select both files!");

  const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
  cnnModel = model;
  model.summary();
  modelInfo.textContent = "";
  cnnModel.summary(null, null, x => (modelInfo.textContent += x + "\n"));
  logStatus("Model loaded from files.");
}

function onReset() {
  trainData = null;
  testData = null;
  cnnModel = null;
  autoencoder = null;
  logsEl.textContent = "";
  statusEl.textContent = "";
  modelInfo.textContent = "";
  previewRow.innerHTML = "";
  document.getElementById("metrics").textContent = "";
  tfvis.visor().close();
}

// ---------- Bind events ----------
document.getElementById("load-data").onclick = onLoadData;
document.getElementById("train-cnn").onclick = onTrainCNN;
document.getElementById("train-denoiser").onclick = onTrainDenoiser;
document.getElementById("evaluate").onclick = onEvaluate;
document.getElementById("test-five").onclick = onTestFive;
document.getElementById("save-model").onclick = onSaveModel;
document.getElementById("load-model").onclick = onLoadModel;
document.getElementById("reset").onclick = onReset;
document.getElementById("toggle-visor").onclick = () => tfvis.visor().toggle();
