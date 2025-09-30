// app.js
// Main application logic: wiring UI, training CNN, denoiser, evaluation, test preview,
// save/load models (file-based), and safe tensor/model disposal.
//
// Follow the spec: use tfjs-vis for training charts and evaluation visualizations.

/* global tf, tfvis */  // for linters/readability

'use strict';

// ----- Global state -----
let trainXs = null, trainYs = null, testXs = null, testYs = null;
let modelCNN = null;
let modelDenoiser = null;
let bestValAcc = 0;
let trainStartTime = null;

// UI elements
const statusDiv = document.getElementById('data-status');
const logsDiv = document.getElementById('training-logs');
const metricsDiv = document.getElementById('metrics');
const modelInfo = document.getElementById('model-info');
const previewRow = document.getElementById('preview-row');

document.getElementById('load-data').addEventListener('click', onLoadData);
document.getElementById('train-cnn').addEventListener('click', onTrainCNN);
document.getElementById('train-denoiser').addEventListener('click', onTrainDenoiser);
document.getElementById('evaluate').addEventListener('click', onEvaluate);
document.getElementById('test-five').addEventListener('click', onTestFive);
document.getElementById('save-model').addEventListener('click', onSaveModel);
document.getElementById('load-model').addEventListener('click', onLoadModel);
document.getElementById('reset').addEventListener('click', onReset);
document.getElementById('toggle-visor').addEventListener('click', () => tfvis.visor().toggle());

// ----- Utilities -----
function safeDispose(t) {
  try { if (t && typeof t.dispose === 'function') t.dispose(); } catch (e) { console.warn('Dispose error', e); }
}

function setStatus(txt) { statusDiv.innerText = txt; }
function log(txt) { logsDiv.innerText = txt; console.log(txt); }

/** Show model summary in the Model Info panel */
function showModelSummary(m) {
  modelInfo.innerText = '';
  m.summary(null, null, line => { modelInfo.innerText += line + '\n'; });
}

/** Count params of model (helper) */
function countParams(m) {
  try { return m.countParams(); } catch (e) { return 'n/a'; }
}

// ----- Data loading -----
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

    // basic sanity
    if (trainXs.shape[1] !== 28 || trainXs.shape[2] !== 28) throw new Error('Train images not 28x28');
    if (testXs.shape[1] !== 28 || testXs.shape[2] !== 28) throw new Error('Test images not 28x28');

    setStatus(`Loaded: Train=${trainXs.shape[0]} Test=${testXs.shape[0]} (parse ${Math.round(t1 - t0)} ms)`);
    log('Data loaded successfully.');

    // preview 5 first train images
    const previewCount = Math.min(5, trainXs.shape[0]);
    previewRow.innerHTML = '';
    for (let i = 0; i < previewCount; ++i) {
      const c = document.createElement('div'); c.className = 'preview-item';
      const canvas = document.createElement('canvas');
      window.draw28x28ToCanvas(trainXs.slice([i,0,0,0],[1,28,28,1]).reshape([28,28,1]), canvas, 3);
      const lbl = document.createElement('div'); lbl.innerText = 'sample ' + i;
      c.appendChild(canvas); c.appendChild(lbl);
      previewRow.appendChild(c);
    }
  } catch (err) {
    console.error(err);
    setStatus('Error loading data: ' + (err.message || err));
    log('Error: ' + (err.message || err));
  }
}

// ----- Model builders -----
function buildCNN() {
  // Model per spec
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28,28,1] }));
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
  m.add(tf.layers.maxPooling2d({ poolSize: [2,2] }));
  m.add(tf.layers.dropout({ rate: 0.25 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.5 }));
  m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  m.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return m;
}

function buildDenoiser() {
  const input = tf.input({ shape: [28,28,1] });
  let x = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(input);
  x = tf.layers.maxPooling2d({ poolSize: [2,2], padding: 'same' }).apply(x); // 14x14
  x = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);
  x = tf.layers.upSampling2d({ size: [2,2] }).apply(x); // 28x28
  const output = tf.layers.conv2d({ filters: 1, kernelSize: 3, padding: 'same', activation: 'sigmoid' }).apply(x);
  const m = tf.model({ inputs: input, outputs: output });
  m.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  return m;
}

// ----- Training CNN -----
async function onTrainCNN() {
  try {
    if (!trainXs || !trainYs) throw new Error('Load data first.');
    // dispose old model
    if (modelCNN) { modelCNN.dispose(); modelCNN = null; }

    modelCNN = buildCNN();
    showModelSummary(modelCNN);
    bestValAcc = 0;
    trainStartTime = performance.now();

    // split train/val
    const { trainXs:trX, trainYs:trY, valXs, valYs } = window.splitTrainVal(trainXs, trainYs, 0.1);

    // fit callbacks with tfjs-vis
    const fitCallbacks = tfvis.show.fitCallbacks(
      { name: 'CNN Training' },
      ['loss', 'val_loss', 'acc', 'val_acc'],
      { callbacks: ['onEpochEnd'] }
    );

    setStatus('Training CNN...');
    log('Starting CNN training...');

    const history = await modelCNN.fit(trX, trY, {
      epochs: 6,
      batchSize: 64,
      shuffle: true,
      validationData: [valXs, valYs],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          // track best val acc
          const v = logs.val_acc || logs.val_accuracy || 0;
          if (v > bestValAcc) bestValAcc = v;
          // update logs area
          logsDiv.innerText = `Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)} val_loss=${(logs.val_loss||0).toFixed(4)} acc=${(logs.acc||0).toFixed(4)} val_acc=${v.toFixed(4)}`;
          await tf.nextFrame();
        },
        ...fitCallbacks
      }
    });

    const duration = Math.round((performance.now() - trainStartTime) / 1000);
    setStatus(`CNN training done. Duration: ${duration}s. Best val_acc=${(bestValAcc*100).toFixed(2)}%`);
    metricsDiv.innerText = `Params: ${countParams(modelCNN)}, Best val acc: ${(bestValAcc*100).toFixed(2)}%`;

    // dispose the slices created by splitTrainVal
    trX.dispose(); trY.dispose(); valXs.dispose(); valYs.dispose();
  } catch (err) {
    console.error(err);
    setStatus('Train error: ' + (err.message || err));
    log('Train error: ' + (err.message || err));
  }
}

// ----- Training Denoiser -----
async function onTrainDenoiser() {
  try {
    if (!trainXs) throw new Error('Load data first.');
    if (modelDenoiser) { modelDenoiser.dispose(); modelDenoiser = null; }

    modelDenoiser = buildDenoiser();
    showModelSummary(modelDenoiser);

    setStatus('Preparing noisy data for denoiser...');
    // create noisy copy
    const noisy = window.addNoise(trainXs, 0.25);

    setStatus('Training denoiser...');
    await modelDenoiser.fit(noisy, trainXs, {
      epochs: 6,
      batchSize: 128,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Denoiser Training' },
        ['loss', 'val_loss'],
        { callbacks: ['onEpochEnd'] }
      )
    });

    noisy.dispose();
    setStatus('Denoiser training done.');
    log('Denoiser trained.');
  } catch (err) {
    console.error(err);
    setStatus('Denoiser training error: ' + (err.message || err));
    log('Denoiser error: ' + (err.message || err));
  }
}

// ----- Evaluate -----
async function onEvaluate() {
  try {
    if (!modelCNN) throw new Error('Train or load CNN first.');
    if (!testXs || !testYs) throw new Error('Load data first.');

    setStatus('Evaluating on test set...');
    log('Evaluating test set...');

    // Evaluate using model.evaluate to get loss & acc
    const evalOutput = await modelCNN.evaluate(testXs, testYs, { batchSize: 128 });
    // evalOutput may be tensor or array [loss, acc]
    let lossTensor = null, accTensor = null;
    if (Array.isArray(evalOutput)) {
      lossTensor = evalOutput[0];
      accTensor = evalOutput[1];
    } else {
      lossTensor = evalOutput;
      accTensor = null;
    }
    const loss = lossTensor ? (await lossTensor.data())[0] : NaN;
    const acc = accTensor ? (await accTensor.data())[0] : NaN;
    metricsDiv.innerText = `Test Accuracy: ${(acc*100).toFixed(2)}% | Loss: ${loss.toFixed(4)}`;

    // --- Confusion matrix & per-class accuracy ---
    // compute predictions and labels as arrays (batched)
    const predsArr = [];
    const labelsArr = [];
    const BATCH = 256;
    const total = testXs.shape[0];
    for (let i = 0; i < total; i += BATCH) {
      const end = Math.min(i + BATCH, total);
      const batchX = testXs.slice([i,0,0,0],[end - i, 28, 28, 1]);
      const logits = modelCNN.predict(batchX);
      const pred = logits.argMax(-1);
      const label = testYs.slice([i,0],[end - i, 10]).argMax(-1);
      predsArr.push(...Array.from(await pred.data()));
      labelsArr.push(...Array.from(await label.data()));
      batchX.dispose(); logits.dispose(); pred.dispose(); label.dispose();
      await tf.nextFrame();
    }

    // build confusion matrix 10x10 where rows = true labels, cols = predicted labels
    const numClasses = 10;
    const conf = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
    for (let i = 0; i < labelsArr.length; ++i) {
      conf[labelsArr[i]][predsArr[i]] += 1;
    }

    tfvis.render.confusionMatrix({ name: 'Confusion Matrix', tab: 'Evaluation' }, { values: conf, tickLabels: [...Array(numClasses).keys()].map(String) });

    // per-class accuracy
    const perClassAcc = conf.map((row, i) => {
      const totalRow = row.reduce((a,b)=>a+b,0) || 1;
      return { label: String(i), value: row[i] / totalRow };
    });
    tfvis.render.barChart(
      { name: 'Per-class accuracy', tab: 'Evaluation' },
      perClassAcc.map(x => ({ index: x.label, value: x.value }))
    );
}

// ----- Test 5 Random -----
async function onTestFive() {
  try {
    if (!modelCNN) throw new Error('Train or load CNN first.');
    if (!testXs || !testYs) throw new Error('Load data first.');

    previewRow.innerHTML = '';
    setStatus('Preparing 5 random test images...');
    const { xs: batchXs, ys: batchYs } = window.getRandomTestBatch(testXs, testYs, 5);
    // show noisy preview if denoiser exists
    const noisy = window.addNoise(batchXs, 0.25);
    let classifierInput = noisy;
    if (modelDenoiser) {
      const den = tf.tidy(() => modelDenoiser.predict(noisy));
      classifierInput = den;
    }

    // predict
    const predsTensor = tf.tidy(() => modelCNN.predict(classifierInput).argMax(-1));
    const labelsTensor = tf.tidy(() => batchYs.argMax(-1));
    const preds = Array.from(await predsTensor.data());
    const labels = Array.from(await labelsTensor.data());

    // render each
    for (let i = 0; i < preds.length; ++i) {
      const container = document.createElement('div'); container.className = 'preview-item';
      const canvas = document.createElement('canvas');
      // draw the (noisy) image for visibility of denoising effect
      const single = noisy.slice([i,0,0,0],[1,28,28,1]).reshape([28,28,1]);
      window.draw28x28ToCanvas(single, canvas, 3);
      single.dispose();
      const lbl = document.createElement('div');
      lbl.innerText = `Pred: ${preds[i]} (GT: ${labels[i]})`;
      lbl.className = (preds[i] === labels[i]) ? 'correct' : 'wrong';
      container.appendChild(canvas);
      container.appendChild(lbl);
      previewRow.appendChild(container);
    }

    // dispose temps
    batchXs.dispose(); batchYs.dispose(); noisy.dispose();
    predsTensor.dispose(); labelsTensor.dispose();
    if (classifierInput !== noisy) safeDispose(classifierInput); // denoiser output already disposed if tidy used, but safe
    setStatus('Random 5 preview rendered.');
    log('Test five done.');
  } catch (err) {
    console.error(err);
    setStatus('Test 5 error: ' + (err.message || err));
    log('Test 5 error: ' + (err.message || err));
  }
}

// ----- Save / Load models (file-based) -----
async function onSaveModel() {
  try {
    if (modelDenoiser) {
      await modelDenoiser.save('downloads://mnist-denoiser');
      setStatus('Denoiser downloaded.');
    } else if (modelCNN) {
      await modelCNN.save('downloads://mnist-cnn');
      setStatus('CNN downloaded.');
    } else {
      throw new Error('No model to save. Train a model first.');
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
    // heuristics: if last dim is 10 -> classifier, else denoiser
    const outShape = m.outputs[0].shape; // e.g. [null,10] or [null,28,28,1]
    if (outShape && outShape.length >= 2 && outShape[outShape.length - 1] === 10) {
      if (modelCNN) modelCNN.dispose();
      modelCNN = m;
      showModelSummary(modelCNN);
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

// ----- Reset -----
function onReset() {
  try {
    setStatus('Resetting application...');
    // dispose tensors and models
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
    // clear tfjs-vis surfaces (optional)
    try { tfvis.visor().close(); } catch (e) { /* ignore */ }
  } catch (err) {
    console.error(err);
    setStatus('Reset error: ' + (err.message || err));
  }
}

// Dispose everything when leaving page
window.addEventListener('beforeunload', () => {
  safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
  safeDispose(modelCNN); safeDispose(modelDenoiser);
});
