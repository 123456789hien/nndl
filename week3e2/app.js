// app.js
// Main UI wiring, model building, training, evaluation and save/load.
// Designed for browser-only usage with TensorFlow.js and tfjs-vis.

// ====== Globals and UI elements ======
let trainXs = null, trainYs = null, testXs = null, testYs = null;
let modelCNN = null;
let modelDenoiser = null;

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
document.getElementById('toggle-visor').addEventListener('click', ()=>tfvis.visor().toggle());

// Utility: safe dispose if exists
function safeDispose(t) { try{ if(t && typeof t.dispose === 'function') t.dispose(); }catch(e){} }

// ====== Data loading ======
async function onLoadData() {
  try {
    logsDiv.innerText = 'Loading data... please wait';
    const trainFile = document.getElementById('train-csv').files[0];
    const testFile = document.getElementById('test-csv').files[0];
    if (!trainFile || !testFile) throw new Error('Select both train and test CSV files.');

    // Dispose previous data if present
    safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
    trainXs = trainYs = testXs = testYs = null;

    // Parse files (defined in data-loader.js)
    const t0 = performance.now();
    const train = await window.loadTrainFromFiles(trainFile);
    const test = await window.loadTestFromFiles(testFile);
    const t1 = performance.now();

    trainXs = train.xs; trainYs = train.ys;
    testXs = test.xs; testYs = test.ys;

    statusDiv.innerText = `Loaded datasets\nTrain: ${trainXs.shape[0]} samples\nTest: ${testXs.shape[0]} samples\nParsing time: ${Math.round(t1 - t0)} ms`;
    logsDiv.innerText = 'Data loaded successfully. You can Train or Train Denoiser.';
    metricsDiv.innerText = '';
    modelInfo.innerText = '';
    previewRow.innerHTML = '';
    // Allow UI render
    await new Promise(r => requestAnimationFrame(r));
  } catch (err) {
    logsDiv.innerText = `Error loading files: ${err.message || err}`;
  }
}

// ====== Model builders ======
function buildCNN() {
  // Sequential classifier as specified in the assignment.
  const m = tf.sequential();
  m.add(tf.layers.conv2d({filters:32, kernelSize:3, activation:'relu', padding:'same', inputShape:[28,28,1]}));
  m.add(tf.layers.conv2d({filters:64, kernelSize:3, activation:'relu', padding:'same'}));
  m.add(tf.layers.maxPooling2d({poolSize:[2,2]}));
  m.add(tf.layers.dropout({rate:0.25}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({units:128, activation:'relu'}));
  m.add(tf.layers.dropout({rate:0.5}));
  m.add(tf.layers.dense({units:10, activation:'softmax'}));
  m.compile({optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy']});
  return m;
}

function buildDenoiser() {
  // Simple conv autoencoder for denoising (encoder-decoder)
  const input = tf.input({shape:[28,28,1]});
  let x = tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(input);
  x = tf.layers.maxPooling2d({poolSize:[2,2],padding:'same'}).apply(x); // 14x14
  x = tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(x);
  x = tf.layers.upSampling2d({size:[2,2]}).apply(x); // 28x28
  const output = tf.layers.conv2d({filters:1,kernelSize:3,padding:'same',activation:'sigmoid'}).apply(x);
  const m = tf.model({inputs: input, outputs: output});
  m.compile({optimizer:'adam', loss:'meanSquaredError'});
  return m;
}

// Utility: show model summary in modelInfo pre
function showModelSummary(m) {
  modelInfo.innerText = '';
  m.summary(null, null, line => { modelInfo.innerText += line + '\n'; });
}

// ====== Training CNN ======
async function onTrainCNN() {
  try {
    if (!trainXs || !trainYs) throw new Error('Load data first.');
    // Dispose old model if any
    if (modelCNN) { modelCNN.dispose(); modelCNN = null; }
    modelCNN = buildCNN();
    showModelSummary(modelCNN);

    // Create train/val split
    const { trainXs:trX, trainYs:trY, valXs, valYs } = window.splitTrainVal(trainXs, trainYs, 0.1);

    // Set up tfjs-vis callbacks for loss/acc charts
    const fitCallbacks = tfvis.show.fitCallbacks(
      { name: 'CNN Training (loss & acc)' },
      ['loss','val_loss','acc','val_acc'],
      { callbacks: ['onEpochEnd'] }
    );

    logsDiv.innerText = 'Training CNN...';
    const t0 = performance.now();
    await modelCNN.fit(trX, trY, {
      epochs: 6,
      batchSize: 64,
      shuffle: true,
      validationData: [valXs, valYs],
      callbacks: fitCallbacks
    });
    const t1 = performance.now();
    logsDiv.innerText = `CNN training done. Took ${(t1 - t0).toFixed(0)} ms.`;

    // Clean up slices; note top-level trainXs/trainYs remain for later use
    trX.dispose(); trY.dispose(); valXs.dispose(); valYs.dispose();

    // Show simple model params
    const params = modelCNN.countParams();
    metricsDiv.innerText = `CNN ready — params: ${params}`;
  } catch (err) {
    logsDiv.innerText = `Training error: ${err.message || err}`;
  }
}

// ====== Training Denoiser ======
async function onTrainDenoiser() {
  try {
    if (!trainXs || !trainYs) throw new Error('Load data first.');
    // Build denoiser model
    if (modelDenoiser) { modelDenoiser.dispose(); modelDenoiser = null; }
    modelDenoiser = buildDenoiser();
    showModelSummary(modelDenoiser);

    // Create noisy train images and train
    logsDiv.innerText = 'Preparing noisy training data...';
    // We'll create noisy copies in memory (careful about memory)
    const noisyTrain = window.addNoise(trainXs, 0.25);

    logsDiv.innerText = 'Training denoiser (autoencoder)...';
    await modelDenoiser.fit(noisyTrain, trainXs, {
      epochs: 6,
      batchSize: 128,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Denoiser Training (loss)' },
        ['loss', 'val_loss'],
        { callbacks: ['onEpochEnd'] }
      )
    });

    noisyTrain.dispose();
    logsDiv.innerText = 'Denoiser training completed.';
  } catch (err) {
    logsDiv.innerText = `Denoiser training error: ${err.message || err}`;
  }
}

// ====== Evaluation ======
async function onEvaluate() {
  try {
    if (!modelCNN) throw new Error('Train or load CNN first.');
    if (!testXs || !testYs) throw new Error('Load data first.');

    logsDiv.innerText = 'Evaluating model on test set...';
    // Evaluate tensors; model.evaluate returns tensor or array of tensors
    const evalResult = await modelCNN.evaluate(testXs, testYs, { batchSize: 128 });
    // Some TFJS versions return an array; normalize
    let accTensor = null;
    if (Array.isArray(evalResult)) {
      // metrics often: [loss, acc]
      accTensor = evalResult.length > 1 ? evalResult[1] : evalResult[0];
    } else {
      accTensor = evalResult;
    }
    const acc = (await accTensor.data())[0];
    metricsDiv.innerText = `Test Accuracy: ${(acc * 100).toFixed(2)}%`;
    logsDiv.innerText = 'Computing confusion matrix & per-class accuracy...';

    // Compute predictions in batches to avoid memory spikes
    await tf.nextFrame();
    const preds = tf.tidy(() => modelCNN.predict(testXs).argMax(-1));
    const labels = tf.tidy(() => testYs.argMax(-1));

    const predsArr = await preds.data();
    const labelsArr = await labels.data();

    // Build confusion matrix 10x10
    const numClasses = 10;
    const conf = Array.from({length:numClasses}, ()=>Array(numClasses).fill(0));
    for (let i = 0; i < labelsArr.length; i++) {
      conf[labelsArr[i]][predsArr[i]] += 1;
    }

    // Render confusion matrix using tfjs-vis
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(container, { values: conf, tickLabels: [...Array(10).keys()].map(String) });

    // Per-class accuracy
    const perClass = conf.map((row, i) => {
      const correct = row[i];
      const total = row.reduce((a,b)=>a+b,0) || 1;
      return { index: i, accuracy: correct / total };
    });
    const barContainer = { name: 'Per-class accuracy', tab: 'Evaluation' };
    tfvis.render.barchart(barContainer, { values: perClass.map(x => x.accuracy), labels: perClass.map(x=>String(x.index)) });

    // Clean up
    preds.dispose(); labels.dispose();
    logsDiv.innerText = 'Evaluation complete.';
  } catch (err) {
    logsDiv.innerText = `Evaluate error: ${err.message || err}`;
  }
}

// ====== Test 5 Random (with optional denoiser) ======
async function onTestFive() {
  try {
    if (!testXs || !testYs) throw new Error('Load data first.');
    if (!modelCNN) throw new Error('Train or load CNN first.');

    previewRow.innerHTML = '';
    logsDiv.innerText = 'Preparing random preview...';

    const { xs: batchXs, ys: batchYs } = window.getRandomTestBatch(testXs, testYs, 5);
    // create noisy preview — if denoiser exists we'll use it to reconstruct
    const noisyBatch = window.addNoise(batchXs, 0.25);

    // Predict using denoiser if available
    let inputForClassifier = noisyBatch;
    if (modelDenoiser) {
      // denoised is new tensor; dispose inputForClassifier after
      const denoised = tf.tidy(() => modelDenoiser.predict(noisyBatch));
      inputForClassifier = denoised;
    }

    // Prediction
    const predsTensor = tf.tidy(() => modelCNN.predict(inputForClassifier).argMax(-1));
    const labelsTensor = tf.tidy(() => batchYs.argMax(-1));

    const predsArr = Array.from(await predsTensor.data());
    const labelsArr = Array.from(await labelsTensor.data());

    // Render each image
    for (let i = 0; i < predsArr.length; i++) {
      const container = document.createElement('div');
      container.className = 'preview-item';
      const canvas = document.createElement('canvas');
      // draw the noisy image (or denoised if used). We'll draw noisyBatch for visual noise.
      const tensorToDraw = noisyBatch.slice([i,0,0,0],[1,28,28,1]).reshape([28,28,1]);
      window.draw28x28ToCanvas(tensorToDraw, canvas, 4);
      tensorToDraw.dispose();
      const lbl = document.createElement('div');
      lbl.innerText = `Pred: ${predsArr[i]} (GT: ${labelsArr[i]})`;
      lbl.className = (predsArr[i] === labelsArr[i]) ? 'correct' : 'wrong';
      container.appendChild(canvas);
      container.appendChild(lbl);
      previewRow.appendChild(container);
    }

    // dispose temporaries
    batchXs.dispose(); batchYs.dispose(); noisyBatch.dispose(); // denoiser output disposed by tidy
    predsTensor.dispose(); labelsTensor.dispose();
    logsDiv.innerText = 'Preview rendered.';
  } catch (err) {
    logsDiv.innerText = `Test preview error: ${err.message || err}`;
  }
}

// ====== Save / Load Model (file-based) ======
async function onSaveModel() {
  try {
    if (modelDenoiser) {
      await modelDenoiser.save('downloads://mnist-denoiser');
      logsDiv.innerText = 'Denoiser saved for download.';
    } else if (modelCNN) {
      await modelCNN.save('downloads://mnist-cnn');
      logsDiv.innerText = 'CNN model saved for download.';
    } else {
      throw new Error('No model available to save. Train or load a model first.');
    }
  } catch (err) {
    logsDiv.innerText = `Save error: ${err.message || err}`;
  }
}

async function onLoadModel() {
  try {
    const jsonFile = document.getElementById('upload-json').files[0];
    const binFile = document.getElementById('upload-weights').files[0];
    if (!jsonFile || !binFile) throw new Error('Select both model JSON and weights BIN files.');

    logsDiv.innerText = 'Loading model from files...';
    const m = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
    // Decide type via output shape (classifier: [null,10])
    const outShape = m.outputs[0].shape; // e.g. [ null, 10 ] or [ null, 28, 28, 1 ]
    if (outShape && outShape.length >= 2 && outShape[1] === 10) {
      // Classifier
      if (modelCNN) modelCNN.dispose();
      modelCNN = m;
      showModelSummary(modelCNN);
      logsDiv.innerText = 'CNN model loaded from files.';
    } else {
      // Denoiser / autoencoder
      if (modelDenoiser) modelDenoiser.dispose();
      modelDenoiser = m;
      showModelSummary(modelDenoiser);
      logsDiv.innerText = 'Denoiser model loaded from files.';
    }
  } catch (err) {
    logsDiv.innerText = `Load error: ${err.message || err}`;
  }
}

// ====== Reset / cleanup ======
function onReset() {
  try {
    logsDiv.innerText = 'Resetting...';
    // Dispose tensors and models safely
    safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
    safeDispose(modelCNN); safeDispose(modelDenoiser);
    trainXs = trainYs = testXs = testYs = null;
    modelCNN = modelDenoiser = null;
    previewRow.innerHTML = '';
    statusDiv.innerText = '';
    metricsDiv.innerText = '';
    modelInfo.innerText = '';
    logsDiv.innerText = 'Reset complete.';
    // clear tfjs-vis elements (optional)
    tfvis.visor().surface({ name: 'Cleared', tab: 'Evaluation' }).drawArea.innerHTML = '';
  } catch (err) {
    logsDiv.innerText = `Reset error: ${err.message || err}`;
  }
}

// Clean up when page unloads
window.addEventListener('beforeunload', () => {
  safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
  safeDispose(modelCNN); safeDispose(modelDenoiser);
});
