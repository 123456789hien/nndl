// app.js
// Main application wiring: UI, model (CNN autoencoder), training, evaluation, preview, save/load.
// This file assumes data-loader.js has exposed `window.dataLoader` utilities.

(async () => {
  // === UI elements ===
  const trainInput = document.getElementById('train-csv');
  const testInput = document.getElementById('test-csv');
  const loadDataBtn = document.getElementById('load-data-btn');
  const trainBtn = document.getElementById('train-btn');
  const evaluateBtn = document.getElementById('evaluate-btn');
  const testFiveBtn = document.getElementById('test-five-btn');
  const saveBtn = document.getElementById('save-btn');
  const resetBtn = document.getElementById('reset-btn');
  const uploadJson = document.getElementById('upload-json');
  const uploadWeights = document.getElementById('upload-weights');
  const loadModelBtn = document.getElementById('load-model-btn');
  const dataStatus = document.getElementById('data-status');
  const modelInfo = document.getElementById('model-info');
  const previewStrip = document.getElementById('preview-strip');
  const logsDiv = document.getElementById('logs');
  const visorRoot = document.getElementById('vis-root');
  const evalSummary = document.getElementById('eval-summary');
  const noiseRange = document.getElementById('noise-level');
  const noiseVal = document.getElementById('noise-val');
  const epochsInput = document.getElementById('epochs');
  const batchSizeInput = document.getElementById('batch-size');
  const toggleVisor = document.getElementById('toggle-visor');

  let trainData = null; // {xs, ys}
  let testData = null;  // {xs, ys}
  let model = null;
  let trainingInfo = null;

  // Keep visor container (tfjs-vis)
  let visorVisible = true;

  noiseRange.addEventListener('input', () => { noiseVal.textContent = noiseRange.value; });

  function appendLog(msg) {
    const p = document.createElement('div');
    p.className = 'small';
    p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logsDiv.prepend(p);
  }

  function setModelInfo(txt) {
    modelInfo.textContent = txt;
  }

  function enableButtonsForData() {
    trainBtn.disabled = false;
    evaluateBtn.disabled = false;
    testFiveBtn.disabled = false;
    saveBtn.disabled = model == null;
  }

  function disableAll() {
    trainBtn.disabled = true;
    evaluateBtn.disabled = true;
    testFiveBtn.disabled = true;
    saveBtn.disabled = true;
  }

  // Dispose everything (tensors & model)
  async function doReset() {
    appendLog('Resetting app and freeing resources...');
    try {
      if (trainData) { trainData.xs.dispose(); trainData.ys.dispose(); trainData = null; }
      if (testData) { testData.xs.dispose(); testData.ys.dispose(); testData = null; }
      if (model) { model.dispose(); model = null; }
      tf.engine().startScope(); tf.engine().endScope();
    } catch (e) {
      console.warn('Error during reset dispose', e);
    }
    previewStrip.innerHTML = '';
    logsDiv.innerHTML = '';
    evalSummary.textContent = 'No evaluation yet.';
    dataStatus.textContent = 'No data loaded.';
    setModelInfo('No model.');
    disableAll();
    appendLog('Reset complete.');
  }

  resetBtn.addEventListener('click', async () => {
    await doReset();
  });

  // LOAD DATA
  loadDataBtn.addEventListener('click', async () => {
    try {
      if (!trainInput.files[0] || !testInput.files[0]) {
        alert('Please select both train and test CSV files.');
        return;
      }
      appendLog('Loading train CSV...');
      const t0 = performance.now();
      const tdata = await window.dataLoader.loadTrainFromFiles(trainInput.files[0]);
      appendLog('Loading test CSV...');
      const tst = await window.dataLoader.loadTestFromFiles(testInput.files[0]);
      const t1 = performance.now();

      // Dispose previous if any
      if (trainData) { trainData.xs.dispose(); trainData.ys.dispose(); }
      if (testData) { testData.xs.dispose(); testData.ys.dispose(); }

      trainData = tdata;
      testData = tst;

      dataStatus.textContent = `Train: ${trainData.xs.shape[0]} images. Test: ${testData.xs.shape[0]} images. Loaded in ${Math.round(t1 - t0)} ms.`;
      appendLog(dataStatus.textContent);

      enableButtonsForData();

      // quick preview of first train image in logs (not required but handy)
      setModelInfo('No model.');
    } catch (err) {
      console.error(err);
      alert('Error loading data: ' + (err && err.message));
    }
  });

  // Build CNN autoencoder model (encoder -> decoder). Input shape [28,28,1]
  function buildAutoencoder() {
    // Using functional API to have a clean encoder-decoder architecture
    const input = tf.input({ shape: [28, 28, 1] });

    // Encoder
    let x = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(input);
    x = tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(x);
    x = tf.layers.maxPooling2d({ poolSize: [2, 2], padding: 'same' }).apply(x); // 14x14x64
    x = tf.layers.dropout({ rate: 0.25 }).apply(x);

    // Bottleneck
    x = tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(x);
    x = tf.layers.maxPooling2d({ poolSize: [2, 2], padding: 'same' }).apply(x); // 7x7x64

    // Decoder
    x = tf.layers.conv2dTranspose({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(x); // 14x14x64
    x = tf.layers.dropout({ rate: 0.25 }).apply(x);
    x = tf.layers.conv2dTranspose({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(x); // 28x28x32
    const decoded = tf.layers.conv2d({ filters: 1, kernelSize: 3, activation: 'sigmoid', padding: 'same' }).apply(x);

    const autoencoder = tf.model({ inputs: input, outputs: decoded, name: 'mnist_denoiser' });

    autoencoder.compile({
      optimizer: tf.train.adam(),
      loss: 'meanSquaredError', // common for denoising
      metrics: ['mse']
    });

    return autoencoder;
  }

  // TRAIN
  trainBtn.addEventListener('click', async () => {
    try {
      if (!trainData || !testData) { alert('Load data first'); return; }

      // Dispose previous model if exists
      if (model) { model.dispose(); model = null; setModelInfo('Replacing previous model...'); }

      appendLog('Building autoencoder model...');
      model = buildAutoencoder();
      setModelInfo('Model built. Summary:\n' + model.summaryToString());

      // Prepare training & validation split
      appendLog('Splitting train/val...');
      const { trainXs, trainYs, valXs, valYs } = window.dataLoader.splitTrainVal(trainData.xs, trainData.ys, 0.1);
      // For denoising: inputs are noisy versions of trainXs; targets are clean trainXs
      const noiseLevelDuringTraining = parseFloat(noiseRange.value) || 0.25;
      const epochs = Number(epochsInput.value) || 8;
      const batchSize = Number(batchSizeInput.value) || 128;

      // Create noisy versions via dataset pipeline (we will generate on-the-fly using map inside fit)
      // For simplicity and browser memory, use xs arrays and tf.data generator
      const trainSize = trainXs.shape[0];
      const valSize = valXs.shape[0];

      // Create tf.data.Dataset from tensors
      const trainDataset = tf.data.array(Array.from({ length: trainSize }, (_, i) => i))
        .map(idx => {
          // for each index produce {xs: noisy, ys: clean}
          return tf.tidy(() => {
            const img = trainXs.gather([idx]); // [1,28,28,1]
            const noisy = window.dataLoader.addNoiseToBatch(img, noiseLevelDuringTraining);
            return { xs: noisy.squeeze([0]), ys: img.squeeze([0]) };
          });
        })
        .batch(batchSize)
        .shuffle(1000);

      const valDataset = tf.data.array(Array.from({ length: valSize }, (_, i) => i))
        .map(idx => {
          return tf.tidy(() => {
            const img = valXs.gather([idx]);
            const noisy = window.dataLoader.addNoiseToBatch(img, noiseLevelDuringTraining);
            return { xs: noisy.squeeze([0]), ys: img.squeeze([0]) };
          });
        })
        .batch(batchSize);

      appendLog(`Starting training: epochs=${epochs}, batchSize=${batchSize}, noise=${noiseLevelDuringTraining}`);

      // tfjs-vis callbacks for live charts
      const metrics = ['loss', 'mse'];
      const container = { name: 'Training: Loss & MSE', tab: 'Training' };
      const surface = { name: 'Training History', tab: 'Training' };

      // fitDataset expects xs and ys; our dataset yields objects. Use model.fitDataset with proper config
      const fitConfig = {
        epochs,
        validationData: valDataset,
        callbacks: tfvis.show.fitCallbacks(container, metrics, { callbacks: ['onEpochEnd'] })
      };

      const t0 = performance.now();
      const history = await model.fitDataset(trainDataset, fitConfig);
      const t1 = performance.now();

      trainingInfo = { history, durationMs: t1 - t0 };
      appendLog(`Training complete in ${Math.round(trainingInfo.durationMs)} ms.`);
      // Dispose temporary tensors used for dataset mapping
      trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();

      // Re-enable Save button
      saveBtn.disabled = false;
      setModelInfo('Trained model. Summary:\n' + model.summaryToString());
      appendLog('Model ready. You can Save (download) or Test 5 Random images.');

      // Enable test/evaluate if not already
      evaluateBtn.disabled = false;
      testFiveBtn.disabled = false;
    } catch (err) {
      console.error(err);
      alert('Training error: ' + (err && err.message));
    }
  });

  // EVALUATE: compute denoising quality metrics on test dataset (MSE) and show a simple heatmap of per-class MSE
  evaluateBtn.addEventListener('click', async () => {
    try {
      if (!model || !testData) { alert('Load data and train or load a model first.'); return; }

      appendLog('Evaluating on test set with noise applied...');
      // For per-class metrics, compute MSE per class
      const testSize = testData.xs.shape[0];
      const noiseLevel = parseFloat(noiseRange.value) || 0.25;

      // We'll process in batches to avoid memory pressure
      const batchSize = 256;
      const batches = Math.ceil(testSize / batchSize);

      // Initialize arrays to accumulate per-class mse and counts
      const perClassSumMse = new Array(10).fill(0);
      const perClassCount = new Array(10).fill(0);
      let totalMse = 0;

      for (let b = 0; b < batches; ++b) {
        const start = b * batchSize;
        const end = Math.min((b + 1) * batchSize, testSize);
        const idx = Array.from({ length: end - start }, (_, i) => start + i);
        await tf.tidy(async () => {
          const orig = testData.xs.gather(idx); // [bs,28,28,1]
          const labels = testData.ys.gather(idx);
          const noisy = window.dataLoader.addNoiseToBatch(orig, noiseLevel);
          const pred = model.predict(noisy);
          // compute mse per image
          const diff = pred.sub(orig).square().mean([1, 2, 3]); // [bs]
          const diffArr = await diff.data();
          const labelArr = await labels.argMax(-1).data();
          for (let i = 0; i < diffArr.length; ++i) {
            const lab = labelArr[i];
            perClassSumMse[lab] += diffArr[i];
            perClassCount[lab] += 1;
            totalMse += diffArr[i];
          }
          orig.dispose(); labels.dispose(); noisy.dispose(); pred.dispose(); diff.dispose();
        });
        // Let browser breathe
        await new Promise(r => requestAnimationFrame(r));
      }

      const avgMse = totalMse / testSize;
      appendLog(`Overall test MSE: ${avgMse.toFixed(6)}`);
      evalSummary.textContent = `Overall test MSE: ${avgMse.toFixed(6)} (noise=${noiseLevel})`;

      // Prepare data for per-class bar chart
      const classNames = [...Array(10).keys()].map(String);
      const perClassMse = classNames.map((_, i) => perClassCount[i] ? perClassSumMse[i] / perClassCount[i] : 0);

      // Render per-class bar chart with tfjs-vis
      const barData = { values: perClassMse.map((v, i) => ({ x: String(i), y: v })) };
      tfvis.render.barchart({ name: 'Per-class MSE', tab: 'Evaluation' }, barData, { width: 600, height: 300 });

      // Also show a confusion-like heatmap? For denoiser we'll show per-class MSE heatmap (1D)
      // Render simple table summary
      const table = document.createElement('table');
      table.style.fontSize = '12px';
      table.style.borderCollapse = 'collapse';
      table.innerHTML = '<tr><th style="padding:4px;border:1px solid #ddd;">Class</th><th style="padding:4px;border:1px solid #ddd;">MSE</th><th style="padding:4px;border:1px solid #ddd;">Count</th></tr>';
      for (let i = 0; i < 10; ++i) {
        const row = document.createElement('tr');
        row.innerHTML = `<td style="padding:4px;border:1px solid #eee;">${i}</td><td style="padding:4px;border:1px solid #eee;">${(perClassMse[i]||0).toFixed(6)}</td><td style="padding:4px;border:1px solid #eee;">${perClassCount[i]}</td>`;
        table.appendChild(row);
      }
      visorRoot.appendChild(table);
    } catch (err) {
      console.error(err);
      alert('Evaluation error: ' + (err && err.message));
    }
  });

  // TEST 5 RANDOM: show original, noisy, denoised for 5 random test images
  testFiveBtn.addEventListener('click', async () => {
    try {
      if (!testData) { alert('Load test data first'); return; }
      if (!model) { alert('Train or load a model first'); return; }

      previewStrip.innerHTML = '';
      const k = 5;
      const noiseLevel = parseFloat(noiseRange.value) || 0.25;

      // get batch
      const { orig, noisy } = window.dataLoader.getRandomTestBatch(testData.xs, testData.ys, k, noiseLevel);
      // predict
      const denoised = tf.tidy(() => model.predict(noisy));

      // For each image create DOM elements
      for (let i = 0; i < k; ++i) {
        // create a container with 3 canvases stacked (original, noisy, denoised)
        const item = document.createElement('div');
        item.className = 'preview-item';

        const origCanv = document.createElement('canvas');
        origCanv.className = 'preview-canvas';
        origCanv.width = 28 * 6;
        origCanv.height = 28 * 6;

        const noisyCanv = document.createElement('canvas');
        noisyCanv.className = 'preview-canvas';
        noisyCanv.width = 28 * 6;
        noisyCanv.height = 28 * 6;

        const denoisedCanv = document.createElement('canvas');
        denoisedCanv.className = 'preview-canvas';
        denoisedCanv.width = 28 * 6;
        denoisedCanv.height = 28 * 6;

        const caption = document.createElement('div');
        caption.className = 'caption small';
        caption.innerText = 'Original / Noisy / Denoised';

        item.appendChild(origCanv);
        item.appendChild(noisyCanv);
        item.appendChild(denoisedCanv);
        item.appendChild(caption);
        previewStrip.appendChild(item);

        // draw images
        // slice individual tensors
        const oi = orig.gather([i]);
        const ni = noisy.gather([i]);
        const di = denoised.gather([i]);

        window.dataLoader.draw28x28ToCanvas(oi, origCanv, 6);
        window.dataLoader.draw28x28ToCanvas(ni, noisyCanv, 6);
        window.dataLoader.draw28x28ToCanvas(di, denoisedCanv, 6);

        oi.dispose(); ni.dispose(); di.dispose();
      }

      // dispose batch tensors
      orig.dispose(); noisy.dispose(); denoised.dispose();

      appendLog('Displayed 5 random denoising results.');
    } catch (err) {
      console.error(err);
      alert('Test 5 Random error: ' + (err && err.message));
    }
  });

  // SAVE model (download model.json + weights.bin)
  saveBtn.addEventListener('click', async () => {
    try {
      if (!model) { alert('No model to save'); return; }
      appendLog('Triggering model download...');
      await model.save('downloads://mnist-dnn-denoiser');
      appendLog('Model downloaded.');
    } catch (err) {
      console.error(err);
      alert('Save error: ' + (err && err.message));
    }
  });

  // LOAD model from user-supplied json + bin files (file inputs)
  loadModelBtn.addEventListener('click', async () => {
    try {
      const jsonFile = uploadJson.files[0];
      const binFile = uploadWeights.files[0];
      if (!jsonFile || !binFile) {
        alert('Please select both model.json and weights.bin files.');
        return;
      }
      appendLog('Loading model from selected files...');
      // Dispose previous model if any
      if (model) { model.dispose(); model = null; }

      const loaded = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
      model = loaded;
      setModelInfo('Loaded model. Summary:\n' + model.summaryToString());
      appendLog('Model loaded from files.');
      saveBtn.disabled = false;
      evaluateBtn.disabled = false;
      testFiveBtn.disabled = false;
    } catch (err) {
      console.error(err);
      alert('Load model error: ' + (err && err.message));
    }
  });

  // Toggle visor open/close
  toggleVisor.addEventListener('click', () => {
    visorVisible = !visorVisible;
    if (visorVisible) {
      tfvis.visor().open();
      appendLog('Visor opened.');
    } else {
      tfvis.visor().close();
      appendLog('Visor closed.');
    }
  });

  // Utility: model.summary as string (since model.summary prints to console by default)
  tf.Model.prototype.summaryToString = function () {
    const originalConsole = console.log;
    let out = '';
    console.log = (s) => { out += s + '\n'; };
    this.summary();
    console.log = originalConsole;
    return out;
  };

  // On page load, initialize visor
  window.addEventListener('load', () => {
    tfvis.visor().open();
    appendLog('Visor opened by default.');
  });

  // Safety: when page unload, dispose resources
  window.addEventListener('beforeunload', async () => {
    try {
      if (trainData) { trainData.xs.dispose(); trainData.ys.dispose(); }
      if (testData) { testData.xs.dispose(); testData.ys.dispose(); }
      if (model) model.dispose();
    } catch (e) { /* ignore */ }
  });

  // Initial disable
  disableAll();

})();
