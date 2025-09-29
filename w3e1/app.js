// ==========================
// app.js  (Denoising Autoencoder Version)
// ==========================

// -----------------------------
// Global State
// -----------------------------
let trainData = null;
let testData = null;
let model = null;
let originalTestXs = null;  // Keep clean test data for comparison
let noisyTestXs = null;     // Noisy version of test data for preview

// Utility to log messages
function logMessage(id, msg) {
  const el = document.getElementById(id);
  if (el) el.textContent = msg;
}

// -----------------------------
// Model Builder for Denoising Autoencoder
// -----------------------------
function createDenoisingAutoencoder() {
  const input = tf.input({shape: [28, 28, 1]});

  // Encoder
  let x = tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(input);
  x = tf.layers.maxPooling2d({poolSize: 2, padding: 'same'}).apply(x);
  x = tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(x);
  x = tf.layers.maxPooling2d({poolSize: 2, padding: 'same'}).apply(x);

  // Decoder
  x = tf.layers.conv2dTranspose({filters: 64, kernelSize: 3, strides: 2, activation: 'relu', padding: 'same'}).apply(x);
  x = tf.layers.conv2dTranspose({filters: 32, kernelSize: 3, strides: 2, activation: 'relu', padding: 'same'}).apply(x);
  const output = tf.layers.conv2d({filters: 1, kernelSize: 3, activation: 'sigmoid', padding: 'same'}).apply(x);

  const autoencoder = tf.model({inputs: input, outputs: output});
  autoencoder.compile({
    optimizer: tf.train.adam(),
    loss: 'meanSquaredError'
  });
  return autoencoder;
}

// -----------------------------
// Add Random Noise
// -----------------------------
function addNoise(xs, noiseFactor = 0.5) {
  return tf.tidy(() => {
    const noisy = xs.add(tf.randomNormal(xs.shape, 0, noiseFactor));
    return noisy.clipByValue(0, 1);
  });
}

// -----------------------------
// UI Handlers
// -----------------------------
async function onLoadData() {
  try {
    const trainFile = document.getElementById('train-csv').files[0];
    const testFile = document.getElementById('test-csv').files[0];
    if (!trainFile || !testFile) {
      alert('Please select both train and test CSV files');
      return;
    }
    const train = await loadTrainFromFiles(trainFile);
    const test = await loadTestFromFiles(testFile);

    trainData = train;
    testData = test;
    originalTestXs = test.xs;

    // Add noise to test data
    noisyTestXs = addNoise(originalTestXs);

    logMessage('data-status', `Loaded Train: ${train.xs.shape[0]} samples, Test: ${test.xs.shape[0]} samples`);
  } catch (err) {
    console.error(err);
    alert('Error loading data: ' + err.message);
  }
}

async function onTrain() {
  try {
    if (!trainData) {
      alert('Please load data first');
      return;
    }

    // Prepare noisy input for training
    const noisyTrainXs = addNoise(trainData.xs);
    const {trainXs, trainYs, valXs, valYs} = splitTrainVal(noisyTrainXs, trainData.xs, 0.1);

    if (model) {
      model.dispose();
    }
    model = createDenoisingAutoencoder();

    // Show model summary safely
    console.log(model.summary());
    document.getElementById('model-info').textContent = model.summary().toString ? model.summary().toString() : 'Model built';

    const fitCallbacks = tfvis.show.fitCallbacks(
      {name: 'Training Performance'},
      ['loss', 'val_loss'],
      {callbacks: ['onEpochEnd']}
    );

    await model.fit(trainXs, trainYs, {
      epochs: 5,
      batchSize: 128,
      validationData: [valXs, valYs],
      shuffle: true,
      callbacks: fitCallbacks
    });

    noisyTrainXs.dispose();
    trainXs.dispose(); trainYs.dispose();
    valXs.dispose(); valYs.dispose();

    alert('Training completed!');
  } catch (err) {
    console.error(err);
    alert('Training error: ' + err.message);
  }
}

async function onTestFive() {
  try {
    if (!model || !noisyTestXs) {
      alert('Model not trained or data not loaded.');
      return;
    }

    const previewDiv = document.getElementById('preview-row');
    previewDiv.innerHTML = '';

    // Get random batch of 5 images
    const {xs, ys} = getRandomTestBatch(noisyTestXs, testData.ys, 5);
    const preds = model.predict(xs);

    const cleanBatch = getRandomTestBatch(originalTestXs, testData.ys, 5);
    const cleanXs = cleanBatch.xs;

    // Draw results
    const xsArr = await xs.array();
    const cleanArr = await cleanXs.array();
    const predsArr = await preds.array();

    for (let i = 0; i < xsArr.length; i++) {
      const container = document.createElement('div');
      container.style.display = 'inline-block';
      container.style.margin = '5px';

      const noisyCanvas = document.createElement('canvas');
      draw28x28ToCanvas(tf.tensor(xsArr[i]), noisyCanvas, 4);
      container.appendChild(noisyCanvas);

      const denoisedCanvas = document.createElement('canvas');
      draw28x28ToCanvas(tf.tensor(predsArr[i]), denoisedCanvas, 4);
      container.appendChild(denoisedCanvas);

      const label = document.createElement('div');
      label.textContent = `Original Label: ${ys.argMax(-1).arraySync()[i]}`;
      container.appendChild(label);

      previewDiv.appendChild(container);
    }

    xs.dispose();
    ys.dispose();
    preds.dispose();
    cleanXs.dispose();
    cleanBatch.ys.dispose();
  } catch (err) {
    console.error(err);
    alert('Test error: ' + err.message);
  }
}

async function onSaveDownload() {
  try {
    if (!model) {
      alert('No model to save');
      return;
    }
    await model.save('downloads://mnist-denoiser');
    alert('Model saved as mnist-denoiser.json + weights.bin');
  } catch (err) {
    console.error(err);
    alert('Save error: ' + err.message);
  }
}

async function onLoadFromFiles() {
  try {
    const jsonFile = document.getElementById('upload-json').files[0];
    const weightsFile = document.getElementById('upload-weights').files[0];
    if (!jsonFile || !weightsFile) {
      alert('Please select both model.json and weights.bin files');
      return;
    }

    if (model) model.dispose();
    model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));

    console.log(model.summary());
    document.getElementById('model-info').textContent = 'Model loaded successfully';
    alert('Model loaded!');
  } catch (err) {
    console.error(err);
    alert('Load error: ' + err.message);
  }
}

function onReset() {
  try {
    if (model) model.dispose();
    if (trainData) {
      trainData.xs.dispose();
      trainData.ys.dispose();
    }
    if (testData) {
      testData.xs.dispose();
      testData.ys.dispose();
    }
    if (noisyTestXs) noisyTestXs.dispose();

    model = null;
    trainData = null;
    testData = null;
    noisyTestXs = null;

    logMessage('data-status', '');
    document.getElementById('preview-row').innerHTML = '';
    document.getElementById('model-info').textContent = '';
    alert('Reset completed!');
  } catch (err) {
    console.error(err);
    alert('Reset error: ' + err.message);
  }
}

// -----------------------------
// Event Bindings
// -----------------------------
document.getElementById('load-data').addEventListener('click', onLoadData);
document.getElementById('train').addEventListener('click', onTrain);
document.getElementById('test-five').addEventListener('click', onTestFive);
document.getElementById('save-model').addEventListener('click', onSaveDownload);
document.getElementById('load-model').addEventListener('click', onLoadFromFiles);
document.getElementById('reset').addEventListener('click', onReset);
document.getElementById('toggle-visor').addEventListener('click', () => tfvis.visor().toggle());
