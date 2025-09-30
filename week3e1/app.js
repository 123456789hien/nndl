// app.js
// Main logic for training, evaluating, and testing the CNN model

let trainData, testData;
let model;

const statusEl = document.getElementById('data-status');
const logsEl = document.getElementById('training-logs');
const metricsEl = document.getElementById('metrics');
const previewRow = document.getElementById('preview-row');
const modelInfoEl = document.getElementById('model-info');

// Load Data
document.getElementById('load-data').onclick = async () => {
  try {
    const trainFile = document.getElementById('train-csv').files[0];
    const testFile = document.getElementById('test-csv').files[0];
    if (!trainFile || !testFile) {
      statusEl.textContent = 'Please select both train and test CSV files.';
      return;
    }
    if (trainData) {
      trainData.xs.dispose();
      trainData.ys.dispose();
    }
    if (testData) {
      testData.xs.dispose();
      testData.ys.dispose();
    }
    trainData = await loadTrainFromFiles(trainFile);
    testData = await loadTestFromFiles(testFile);
    statusEl.textContent = `Loaded Train: ${trainData.xs.shape[0]} samples\nLoaded Test: ${testData.xs.shape[0]} samples`;
  } catch (err) {
    statusEl.textContent = 'Error loading data: ' + err.message;
  }
};

// Build CNN
function createCnnModel() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({filters:32,kernelSize:3,activation:'relu',padding:'same',inputShape:[28,28,1]}));
  m.add(tf.layers.conv2d({filters:64,kernelSize:3,activation:'relu',padding:'same'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.dropout({rate:0.25}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({units:128,activation:'relu'}));
  m.add(tf.layers.dropout({rate:0.5}));
  m.add(tf.layers.dense({units:10,activation:'softmax'}));
  m.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});
  return m;
}

// Train
document.getElementById('train-cnn').onclick = async () => {
  if (!trainData) {
    statusEl.textContent = 'Please load data first.';
    return;
  }
  if (model) model.dispose();
  model = createCnnModel();
  model.summary();
  modelInfoEl.textContent = '';
  model.summary(undefined, undefined, x => modelInfoEl.textContent += x + '\n');
  const {trainXs, trainYs, valXs, valYs} = splitTrainVal(trainData.xs, trainData.ys);
  const fitCallbacks = tfvis.show.fitCallbacks(
    {name:'Training', tab:'Training'},
    ['loss','val_loss','acc','val_acc'],
    {callbacks:['onEpochEnd']}
  );
  logsEl.textContent = 'Training started...';
  await model.fit(trainXs, trainYs, {
    epochs: 5,
    batchSize: 128,
    validationData: [valXs, valYs],
    shuffle: true,
    callbacks: fitCallbacks
  });
  logsEl.textContent = 'Training finished.';
};

// Evaluate
document.getElementById('evaluate').onclick = async () => {
  if (!model || !testData) {
    statusEl.textContent = 'Please load data and train model first.';
    return;
  }
  const preds = model.predict(testData.xs).argMax(-1);
  const labels = testData.ys.argMax(-1);
  const acc = await tf.equal(preds, labels).mean().data();
  metricsEl.textContent = `Test Accuracy: ${(acc[0]*100).toFixed(2)}%`;

  const predsArr = Array.from(await preds.data());
  const labelsArr = Array.from(await labels.data());
  const confusionMatrix = tfvis.metrics.confusionMatrix(labelsArr, predsArr);
  const classAccuracy = tfvis.metrics.perClassAccuracy(labelsArr, predsArr);

  await tfvis.render.confusionMatrix(
    {name:'Confusion Matrix', tab:'Evaluation'},
    {values:confusionMatrix}
  );
  await tfvis.render.barchart(
    {name:'Per-class Accuracy', tab:'Evaluation'},
    classAccuracy
  );
  preds.dispose(); labels.dispose();
};

// Test 5 Random
document.getElementById('test-five').onclick = async () => {
  if (!model || !testData) {
    statusEl.textContent = 'Please load data and train model first.';
    return;
  }
  previewRow.innerHTML = '';
  const {batchXs, batchYs, indices} = getRandomTestBatch(testData.xs, testData.ys, 5);
  const preds = model.predict(batchXs).argMax(-1);
  const labels = batchYs.argMax(-1);
  const predsArr = Array.from(await preds.data());
  const labelsArr = Array.from(await labels.data());
  for (let i = 0; i < 5; i++) {
    const item = document.createElement('div');
    item.className = 'preview-item';
    const canvas = document.createElement('canvas');
    canvas.width = 28*4; canvas.height = 28*4;
    draw28x28ToCanvas(batchXs.slice([i,0,0,0],[1,28,28,1]).reshape([28,28]), canvas, 4);
    item.appendChild(canvas);
    const labelEl = document.createElement('div');
    labelEl.textContent = predsArr[i];
    labelEl.className = predsArr[i]===labelsArr[i] ? 'correct' : 'wrong';
    item.appendChild(labelEl);
    previewRow.appendChild(item);
  }
  preds.dispose(); labels.dispose(); batchXs.dispose(); batchYs.dispose();
};

// Save model
document.getElementById('save-model').onclick = async () => {
  if (!model) return;
  await model.save('downloads://mnist-cnn');
};

// Load model from files
document.getElementById('load-model').onclick = async () => {
  try {
    const jsonFile = document.getElementById('upload-json').files[0];
    const weightsFile = document.getElementById('upload-weights').files[0];
    if (!jsonFile || !weightsFile) {
      statusEl.textContent = 'Please select both model.json and weights.bin files.';
      return;
    }
    if (model) model.dispose();
    model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
    modelInfoEl.textContent = '';
    model.summary(undefined, undefined, x => modelInfoEl.textContent += x + '\n');
    statusEl.textContent = 'Model loaded successfully.';
  } catch (err) {
    statusEl.textContent = 'Error loading model: ' + err.message;
  }
};

// Reset
document.getElementById('reset').onclick = () => {
  if (model) { model.dispose(); model=null; }
  if (trainData) { trainData.xs.dispose(); trainData.ys.dispose(); trainData=null; }
  if (testData) { testData.xs.dispose(); testData.ys.dispose(); testData=null; }
  statusEl.textContent = ''; logsEl.textContent = ''; metricsEl.textContent='';
  previewRow.innerHTML=''; modelInfoEl.textContent='';
};

// Toggle visor
document.getElementById('toggle-visor').onclick = () => {
  tfvis.visor().toggle();
};
