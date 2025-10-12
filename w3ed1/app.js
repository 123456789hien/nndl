// app.js
// Main app: wires UI, trains classifier & CNN denoiser, evaluates (confusion matrix + per-class accuracy),
// shows test previews (original, noisy, denoised), and supports model save/load (file-based).
//
// Keep UI/UX unchanged; robust tf usage and tfjs-vis charts.

'use strict';

// Global state
let trainXs = null, trainYs = null, testXs = null, testYs = null;
let modelCNN = null, modelDenoiser = null;

// UI elements
const statusDiv = document.getElementById('data-status');
const logsDiv = document.getElementById('training-logs');
const metricsDiv = document.getElementById('metrics');
const modelInfo = document.getElementById('model-info');
const previewRow = document.getElementById('preview-row');

// Wire UI buttons
document.getElementById('load-data').addEventListener('click', onLoadData);
document.getElementById('train-cnn').addEventListener('click', onTrainCNN);
document.getElementById('train-denoiser').addEventListener('click', onTrainDenoiser);
document.getElementById('evaluate').addEventListener('click', onEvaluate);
document.getElementById('test-five').addEventListener('click', onTestFive);
document.getElementById('save-model').addEventListener('click', onSaveModel);
document.getElementById('load-model').addEventListener('click', onLoadModel);
document.getElementById('reset').addEventListener('click', onReset);
document.getElementById('toggle-visor').addEventListener('click', () => tfvis.visor().toggle());

// Utility helpers
function safeDispose(t) { try { if (t && typeof t.dispose === 'function') t.dispose(); } catch (e) { console.warn('Dispose error', e); } }
function setStatus(txt) { statusDiv.innerText = txt; }
function log(txt) { logsDiv.innerText = txt; console.log(txt); }
function showModelSummary(m) { try { modelInfo.innerText = ''; m.summary(null, null, line => { modelInfo.innerText += line + '\n'; }); } catch (e) { console.warn(e); } }
function countParams(m) { try { return m.countParams(); } catch (e) { return 'n/a'; } }

// ---------------- Data loading ----------------
async function onLoadData() {
  try {
    const trainFile = document.getElementById('train-csv').files[0];
    const testFile = document.getElementById('test-csv').files[0];
    if (!trainFile || !testFile) throw new Error('Please select both train and test CSV files.');

    // dispose previous
    safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
    trainXs = trainYs = testXs = testYs = null;
    previewRow.innerHTML = '';
    modelInfo.innerText = '';

    setStatus('Loading CSV files...');
    await tf.nextFrame();

    const t0 = performance.now();
    const train = await window.loadTrainFromFiles(trainFile);
    const test = await window.loadTestFromFiles(testFile);
    const t1 = performance.now();

    trainXs = train.xs; trainYs = train.ys;
    testXs = test.xs; testYs = test.ys;

    // basic sanity checks
    if (trainXs.shape[1] !== 28 || trainXs.shape[2] !== 28) throw new Error('Train images not 28x28');
    if (testXs.shape[1] !== 28 || testXs.shape[2] !== 28) throw new Error('Test images not 28x28');

    setStatus(`Loaded: Train=${trainXs.shape[0]} Test=${testXs.shape[0]} (parse ${Math.round(t1 - t0)} ms)`);
    log('Data loaded successfully.');

    // preview 5 random test images
    onTestFive();
  } catch (err) {
    console.error(err);
    setStatus('Error loading data: ' + (err.message || err));
    log('Error: ' + (err.message || err));
  }
}

// ---------------- Model builders ----------------
// Build classifier CNN per spec
function buildCNN() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28, 28, 1] }));
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
  m.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  m.add(tf.layers.dropout({ rate: 0.25 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.5 }));
  m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  m.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return m;
}

// Build CNN autoencoder denoiser (convolutional autoencoder)
function buildDenoiser() {
  const input = tf.input({ shape: [28, 28, 1] });
  // Encoder
  let x = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(input); // 28x28x32
  x = tf.layers.maxPooling2d({ poolSize: [2, 2], padding: 'same' }).apply(x); // 14x14x32
  x = tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(x); // 14x14x16
  x = tf.layers.maxPooling2d({ poolSize: [2, 2], padding: 'same' }).apply(x); // 7x7x16

  // Decoder
  x = tf.layers.conv2dTranspose({ filters: 16, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(x); // 14x14x16
  x = tf.layers.conv2dTranspose({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(x); // 28x28x32
  const output = tf.layers.conv2d({ filters: 1, kernelSize: 3, padding: 'same', activation: 'sigmoid' }).apply(x); // 28x28x1

  const m = tf.model({ inputs: input, outputs: output });
  m.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  return m;
}

// ---------------- Training CNN ----------------
async function onTrainCNN() {
  try {
    if (!trainXs || !trainYs) throw new Error('Load data first.');
    // dispose old model if exists
    if (modelCNN) { modelCNN.dispose(); modelCNN = null; }

    modelCNN = buildCNN();
    showModelSummary(modelCNN);
    setStatus('Preparing training split...');

    const { trainXs: trX, trainYs: trY, valXs, valYs } = window.splitTrainVal(trainXs, trainYs, 0.1);

    setStatus('Training CNN...');
    const fitCallbacks = tfvis.show.fitCallbacks(
      { name: 'CNN Training', tab: 'Training' },
      ['loss', 'val_loss', 'accuracy', 'val_accuracy'],
      { callbacks: ['onEpochEnd'] }
    );

    await modelCNN.fit(trX, trY, {
      epochs: 6,
      batchSize: 64,
      shuffle: true,
      validationData: [valXs, valYs],
      callbacks: fitCallbacks
    });

    setStatus('CNN training done.');
    trX.dispose(); trY.dispose(); valXs.dispose(); valYs.dispose();
  } catch (err) {
    console.error(err);
    setStatus('Train error: ' + (err.message || err));
    log('Train error: ' + (err.message || err));
  }
}

// ---------------- Training Denoiser ----------------
async function onTrainDenoiser() {
  try {
    if (!trainXs) throw new Error('Load data first.');
    // dispose old denoiser
    if (modelDenoiser) { modelDenoiser.dispose(); modelDenoiser = null; }

    modelDenoiser = buildDenoiser();
    showModelSummary(modelDenoiser);

    setStatus('Preparing noisy data for denoiser (train/val)...');
    const { trainXs: trX, trainYs: trY, valXs, valYs } = window.splitTrainVal(trainXs, trainYs, 0.1);

    // create noisy inputs for train and val
    const noisyTr = window.addNoise(trX, 0.25);
    const noisyVal = window.addNoise(valXs, 0.25);

    setStatus('Training denoiser (CNN autoencoder)...');
    const fitCallbacks = tfvis.show.fitCallbacks(
      { name: 'Denoiser Training', tab: 'Training' },
      ['loss', 'val_loss'],
      { callbacks: ['onEpochEnd'] }
    );

    await modelDenoiser.fit(noisyTr, trX, {
      epochs: 8,
      batchSize: 128,
      shuffle: true,
      validationData: [noisyVal, valXs],
      callbacks: fitCallbacks
    });

    setStatus('Denoiser training done.');
    // dispose temps
    noisyTr.dispose(); noisyVal.dispose();
    trX.dispose(); trY.dispose(); valXs.dispose(); valYs.dispose();
  } catch (err) {
    console.error(err);
    setStatus('Denoiser training error: ' + (err.message || err));
    log('Denoiser error: ' + (err.message || err));
  }
}

// ---------------- Evaluation ----------------
async function onEvaluate() {
  try {
    if (!modelCNN) throw new Error('Train or load CNN first.');
    if (!testXs || !testYs) throw new Error('Load data first.');

    setStatus('Evaluating on test set...');
    log('Evaluating test set...');

    // Compute predictions in batches to avoid memory pressure
    const predsArr = [];
    const labelsArr = [];
    const BATCH = 256;
    const total = testXs.shape[0];
    for (let i = 0; i < total; i += BATCH) {
      const end = Math.min(i + BATCH, total);
      const batchX = testXs.slice([i, 0, 0, 0], [end - i, 28, 28, 1]);
      const logits = modelCNN.predict(batchX);
      const pred = logits.argMax(-1);
      const label = testYs.slice([i, 0], [end - i, 10]).argMax(-1);
      predsArr.push(...Array.from(await pred.data()));
      labelsArr.push(...Array.from(await label.data()));
      batchX.dispose(); logits.dispose(); pred.dispose(); label.dispose();
      await tf.nextFrame();
    }

    // Overall accuracy
    let correct = 0;
    for (let i = 0; i < labelsArr.length; ++i) if (labelsArr[i] === predsArr[i]) correct++;
    const overallAcc = (correct / labelsArr.length) * 100;
    metricsDiv.innerText = `Overall Test Accuracy: ${overallAcc.toFixed(2)}%`;

    // Build confusion matrix (rows = true labels, cols = predicted labels)
    const numClasses = 10;
    const conf = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
    for (let i = 0; i < labelsArr.length; ++i) {
      conf[labelsArr[i]][predsArr[i]] += 1;
    }

    // Per-class accuracy
    const perClassAcc = conf.map((row, i) => {
      const totalRow = row.reduce((a, b) => a + b, 0) || 1;
      return { label: String(i), value: row[i] / totalRow };
    });

    // Render both charts in Evaluation tab — use separate surfaces (name unique) so they both show
    tfvis.render.confusionMatrix({ name: 'Confusion Matrix', tab: 'Evaluation' }, { values: conf, tickLabels: [...Array(numClasses).keys()].map(String) });

    tfvis.render.barchart(
      { name: 'Per-class Accuracy', tab: 'Evaluation' },
      { values: perClassAcc.map(x => x.value * 100), labels: perClassAcc.map(x => x.label) }
    );

    setStatus(`Evaluation done. Accuracy ${overallAcc.toFixed(2)}%`);
    log('Evaluation complete.');
  } catch (err) {
    console.error(err);
    setStatus('Evaluate error: ' + (err.message || err));
    log('Evaluate error: ' + (err.message || err));
  }
}

// ---------------- Test 5 Random (show original | noisy | denoised | predicted) ----------------
async function onTestFive() {
  try {
    if (!testXs || !testYs) throw new Error('Load data first.');

    previewRow.innerHTML = '';
    setStatus('Preparing 5 random test images...');
    const { xs: batchXs, ys: batchYs } = window.getRandomTestBatch(testXs, testYs, 5);

    // create noisy batch and (if available) denoised output
    const noisyBatch = window.addNoise(batchXs, 0.25);
    let denoisedBatch = null;
    if (modelDenoiser) {
      // denoiser output might be larger — ensure tidy to avoid leaks
      denoisedBatch = tf.tidy(() => modelDenoiser.predict(noisyBatch));
    }

    // classifier input: if denoiser exists, classify denoised images; otherwise classify noisy images
    const classifierInput = denoisedBatch ? denoisedBatch : noisyBatch;
    const predsTensor = tf.tidy(() => modelCNN ? modelCNN.predict(classifierInput).argMax(-1) : null);
    const labelsTensor = batchYs.argMax(-1);

    const preds = predsTensor ? Array.from(await predsTensor.data()) : Array(batchXs.shape[0]).fill(null);
    const labels = Array.from(await labelsTensor.data());

    for (let i = 0; i < preds.length; ++i) {
      const container = document.createElement('div'); container.className = 'preview-item';
      // original
      const canvasOrig = document.createElement('canvas');
      window.draw28x28ToCanvas(batchXs.slice([i, 0, 0, 0], [1, 28, 28, 1]), canvasOrig, 3);
      // noisy
      const canvasNoisy = document.createElement('canvas');
      window.draw28x28ToCanvas(noisyBatch.slice([i, 0, 0, 0], [1, 28, 28, 1]), canvasNoisy, 3);
      container.appendChild(canvasOrig);
      container.appendChild(canvasNoisy);

      // denoised (if exists)
      if (denoisedBatch) {
        const canvasDen = document.createElement('canvas');
        window.draw28x28ToCanvas(denoisedBatch.slice([i, 0, 0, 0], [1, 28, 28, 1]), canvasDen, 3);
        container.appendChild(canvasDen);
      }

      // label / prediction
      const lbl = document.createElement('div');
      lbl.innerText = `Pred: ${preds[i] !== null ? preds[i] : '-'} (GT: ${labels[i]})`;
      lbl.className = (preds[i] === labels[i]) ? 'correct' : 'wrong';
      container.appendChild(lbl);

      previewRow.appendChild(container);
    }

    // dispose temps
    batchXs.dispose(); batchYs.dispose();
    noisyBatch.dispose();
    if (denoisedBatch) denoisedBatch.dispose();
    if (predsTensor) predsTensor.dispose();
    labelsTensor.dispose();

    setStatus('Random 5 preview rendered.');
    log('Test five done.');
  } catch (err) {
    console.error(err);
    setStatus('Test 5 error: ' + (err.message || err));
    log('Test 5 error: ' + (err.message || err));
  }
}

// ---------------- Save / Load models (file-based) ----------------
async function onSaveModel() {
  try {
    // prefer saving denoiser if present (per your previous behavior), otherwise CNN
    if (modelDenoiser) {
      await modelDenoiser.save('downloads://mnist-denoiser');
      setStatus('Denoiser downloaded.');
    }
    if (modelCNN) {
      await modelCNN.save('downloads://mnist-cnn');
      setStatus(prev => 'Models downloaded.');
    }
  } catch (err) {
    console.error(err);
    setStatus('Save error: ' + (err.message || err));
    log('Save error: ' + (err.message || err));
  }
}

async function onLoadModel() {
  try {
    const jsonFile = document.getElementById('upload-json').files[0];
    const binFile = document.getElementById('upload-weights').files[0];
    if (!jsonFile || !binFile) throw new Error('Select both JSON and BIN weight files.');

    setStatus('Loading model from files...');
    const m = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));

    // Heuristic: if output last dim == 10 => classifier, else denoiser
    const outShape = (m.outputs && m.outputs[0] && m.outputs[0].shape) ? m.outputs[0].shape : null;
    let isClassifier = false;
    if (outShape && outShape.length >= 2) {
      const lastDim = outShape[outShape.length - 1];
      if (lastDim === 10) isClassifier = true;
    }

    if (isClassifier) {
      if (modelCNN) modelCNN.dispose();
      modelCNN = m;
      showModelSummary(modelCNN); // update Model Info but DO NOT clear it elsewhere
      setStatus('CNN loaded from files.');
      log('CNN loaded.');
    } else {
      if (modelDenoiser) modelDenoiser.dispose();
      modelDenoiser = m;
      showModelSummary(modelDenoiser);
      setStatus('Denoiser loaded from files.');
      log('Denoiser loaded.');
    }
  } catch (err) {
    console.error(err);
    setStatus('Load error: ' + (err.message || err));
    log('Load error: ' + (err.message || err));
  }
}

// ---------------- Reset ----------------
function onReset() {
  try {
    setStatus('Resetting application...');
    safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
    safeDispose(modelCNN); safeDispose(modelDenoiser);
    trainXs = trainYs = testXs = testYs = null;
    modelCNN = modelDenoiser = null;
    previewRow.innerHTML = '';
    modelInfo.innerText = '';
    metricsDiv.innerText = '';
    logsDiv.innerText = '';
    setStatus('Reset complete.');
    log('Reset completed.');
    try { tfvis.visor().close(); } catch (e) { /* ignore */ }
  } catch (err) {
    console.error(err);
    setStatus('Reset error: ' + (err.message || err));
  }
}

// Dispose everything on unload
window.addEventListener('beforeunload', () => {
  safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
  safeDispose(modelCNN); safeDispose(modelDenoiser);
});
