// app.js
// Main application logic for MNIST denoiser (CNN autoencoder).
// - Wires UI to data-loader.js functions
// - Builds a convolutional autoencoder
// - Trains with noisy inputs / clean targets
// - Evaluates with MSE and per-class MSE
// - Shows preview of 5 random images (Original, Noisy, Denoised)
// - Saves/loads model via file download & browserFiles
// Note: this file assumes data-loader.js exported window.dataLoader and TFJS + TFJS-VIS are loaded.

(() => {
  // ---- DOM READY ----
  window.addEventListener('DOMContentLoaded', () => {
    // UI elements (IDs match index.html)
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
    const visRoot = document.getElementById('vis-root');
    const evalSummary = document.getElementById('eval-summary');
    const noiseRange = document.getElementById('noise-level');
    const noiseVal = document.getElementById('noise-val');
    const epochsInput = document.getElementById('epochs');
    const batchSizeInput = document.getElementById('batch-size');
    const toggleVisor = document.getElementById('toggle-visor');

    // App state
    let trainData = null; // {xs, ys}
    let testData = null;  // {xs, ys}
    let model = null;
    let trainingInfo = null;

    // Init visor
    tfvis.visor().open();

    // Update noise label
    noiseRange.addEventListener('input', () => { noiseVal.textContent = noiseRange.value; });

    // Logging helper
    function appendLog(msg) {
      const p = document.createElement('div');
      p.className = 'muted';
      p.style.fontSize = '13px';
      p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
      logsDiv.prepend(p);
    }

    // Safe model summary to string (capture console output)
    function modelSummaryToString(m) {
      if (!m) return 'No model.';
      const original = console.log;
      let out = '';
      console.log = (...args) => { out += args.join(' ') + '\n'; };
      try {
        m.summary();
      } catch (err) {
        out += `Error summarizing model: ${err.message}\n`;
      } finally {
        console.log = original;
      }
      return out;
    }

    // Enable/disable buttons based on state
    function setButtonsForDataLoaded(loaded = false) {
      trainBtn.disabled = !loaded;
      evaluateBtn.disabled = !loaded || !model;
      testFiveBtn.disabled = !loaded || !model;
      saveBtn.disabled = !model;
    }

    // Dispose object helper
    function safeDispose(obj) {
      try {
        if (!obj) return;
        if (Array.isArray(obj)) { obj.forEach(o => o && o.dispose && o.dispose()); return; }
        if (obj && obj.dispose) obj.dispose();
      } catch (e) {
        console.warn('Dispose error', e);
      }
    }

    // RESET
    resetBtn.addEventListener('click', async () => {
      appendLog('Resetting app — disposing tensors and model...');
      safeDispose(trainData && trainData.xs); safeDispose(trainData && trainData.ys);
      safeDispose(testData && testData.xs); safeDispose(testData && testData.ys);
      if (model) { model.dispose(); model = null; }
      trainData = null; testData = null; trainingInfo = null;
      previewStrip.innerHTML = '';
      logsDiv.innerHTML = '';
      dataStatus.textContent = 'No data loaded.';
      modelInfo.textContent = 'No model.';
      evalSummary.textContent = 'No evaluation yet.';
      setButtonsForDataLoaded(false);
      appendLog('Reset complete.');
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
        const ttrain = await window.dataLoader.loadTrainFromFiles(trainInput.files[0]);
        appendLog('Loading test CSV...');
        const ttest = await window.dataLoader.loadTestFromFiles(testInput.files[0]);
        const t1 = performance.now();

        // Dispose previous
        safeDispose(trainData && trainData.xs); safeDispose(trainData && trainData.ys);
        safeDispose(testData && testData.xs); safeDispose(testData && testData.ys);

        trainData = ttrain;
        testData = ttest;

        dataStatus.textContent = `Train: ${trainData.xs.shape[0]} images • Test: ${testData.xs.shape[0]} images (loaded ${(t1 - t0).toFixed(0)} ms)`;
        appendLog('Data loaded successfully.');
        setButtonsForDataLoaded(true);

        // Clear previous visuals
        previewStrip.innerHTML = '';
        evalSummary.textContent = 'No evaluation yet.';
        modelInfo.textContent = model ? modelSummaryToString(model) : 'No model.';
      } catch (err) {
        console.error(err);
        alert('Error loading CSVs: ' + (err && err.message));
      }
    });

    // Build autoencoder (functional API) — encoder-decoder with conv layers
    function buildAutoencoder() {
      const input = tf.input({ shape: [28, 28, 1] });

      // Encoder block
      let x = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(input);
      x = tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(x);
      x = tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }).apply(x); // 14x14x64
      x = tf.layers.dropout({ rate: 0.25 }).apply(x);

      // Bottleneck
      x = tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(x);
      x = tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }).apply(x); // 7x7x64

      // Decoder block
      x = tf.layers.conv2dTranspose({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(x); // 14x14x64
      x = tf.layers.dropout({ rate: 0.25 }).apply(x);
      x = tf.layers.conv2dTranspose({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(x); // 28x28x32
      const decoded = tf.layers.conv2d({ filters: 1, kernelSize: 3, padding: 'same', activation: 'sigmoid' }).apply(x);

      const autoencoder = tf.model({ inputs: input, outputs: decoded, name: 'mnist_denoiser' });

      autoencoder.compile({
        optimizer: tf.train.adam(),
        loss: 'meanSquaredError',
        metrics: ['mse']
      });

      return autoencoder;
    }

    // TRAIN
    trainBtn.addEventListener('click', async () => {
      try {
        if (!trainData || !testData) { alert('Please load data first'); return; }

        // Dispose previous model if exists
        if (model) { model.dispose(); model = null; }

        appendLog('Building autoencoder model...');
        model = buildAutoencoder();
        modelInfo.textContent = modelSummaryToString(model);

        // Prepare train/val split from trainData
        appendLog('Preparing training/validation split...');
        const { trainXs, trainYs, valXs, valYs } = window.dataLoader.splitTrainVal(trainData.xs, trainData.ys, 0.1);

        // We'll generate noisy inputs on-the-fly in tf.data pipeline so we don't hold two copies in memory.
        const noiseLevelDuringTraining = parseFloat(noiseRange.value) || 0.25;
        const epochs = Number(epochsInput.value) || 8;
        const batchSize = Number(batchSizeInput.value) || 128;

        // Helper to create dataset of {xs:noisyImg, ys:cleanImg}
        function createDatasetFromTensor(xsTensor) {
          const size = xsTensor.shape[0];
          // map over indices to avoid copying entire dataset
          const ds = tf.data.array([...Array(size).keys()]).map(i => {
            return tf.tidy(() => {
              const img = xsTensor.gather([i]); // [1,28,28,1]
              const noisy = window.dataLoader.addNoiseToBatch(img, noiseLevelDuringTraining);
              // Return tensors with batch dimension removed; fitDataset will collect into batches
              return { xs: noisy.squeeze([0]), ys: img.squeeze([0]) };
            });
          }).batch(batchSize).shuffle(1000);
          return ds;
        }

        const trainDs = createDatasetFromTensor(trainXs);
        const valDs = createDatasetFromTensor(valXs);

        appendLog(`Starting training: epochs=${epochs}, batchSize=${batchSize}, noise=${noiseLevelDuringTraining}`);
        const fitCallbacks = tfvis.show.fitCallbacks(
          { name: 'Training: loss & mse', tab: 'Training' },
          ['loss', 'mse', 'val_loss', 'val_mse'],
          { callbacks: ['onEpochEnd'] }
        );

        const t0 = performance.now();
        const history = await model.fitDataset(trainDs, {
          epochs,
          validationData: valDs,
          callbacks: fitCallbacks
        });
        const t1 = performance.now();
        trainingInfo = { history, durationMs: t1 - t0 };
        appendLog(`Training finished in ${Math.round(trainingInfo.durationMs)} ms`);

        // Dispose temporary tensors used for split (trainXs, trainYs, valXs, valYs)
        trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();

        // Enable buttons
        setButtonsForDataLoaded(true);
        modelInfo.textContent = modelSummaryToString(model);
        appendLog('Model trained and ready. You may Save or Test 5 Random images.');
      } catch (err) {
        console.error(err);
        alert('Training error: ' + (err && err.message));
      }
    });

    // EVALUATE: Compute overall MSE and per-class MSE on test set with noise applied
    evaluateBtn.addEventListener('click', async () => {
      try {
        if (!model || !testData) { alert('Please train or load a model and ensure test data is loaded.'); return; }

        appendLog('Evaluating model on test set (with noise)...');
        const noiseLevel = parseFloat(noiseRange.value) || 0.25;
        const testSize = testData.xs.shape[0];
        const batchSize = 256;
        const steps = Math.ceil(testSize / batchSize);

        // accumulate per-class mse
        const sumMse = Array(10).fill(0);
        const countCls = Array(10).fill(0);
        let totalMse = 0;

        for (let s = 0; s < steps; ++s) {
          const start = s * batchSize;
          const end = Math.min((s + 1) * batchSize, testSize);
          const ids = Array.from({ length: end - start }, (_, i) => start + i);

          await tf.tidy(async () => {
            const orig = testData.xs.gather(ids); // clean batch
            const labels = testData.ys.gather(ids);
            const noisy = window.dataLoader.addNoiseToBatch(orig, noiseLevel);
            const preds = model.predict(noisy);
            // per-image mse
            const perImgMse = preds.sub(orig).square().mean([1, 2, 3]); // [bs]
            const mseArr = await perImgMse.data();
            const labArr = await labels.argMax(-1).data();

            for (let i = 0; i < mseArr.length; ++i) {
              const lab = labArr[i];
              sumMse[lab] += mseArr[i];
              countCls[lab] += 1;
              totalMse += mseArr[i];
            }

            // dispose
            orig.dispose(); labels.dispose(); noisy.dispose(); preds.dispose(); perImgMse.dispose();
          });

          // keep UI responsive
          await new Promise(r => requestAnimationFrame(r));
        }

        const overallMse = totalMse / testSize;
        appendLog(`Overall test MSE: ${overallMse.toFixed(6)}`);
        evalSummary.textContent = `Overall test MSE: ${overallMse.toFixed(6)} (noise=${noiseLevel})`;

        // per-class MSE
        const perClassMse = sumMse.map((s, i) => countCls[i] ? s / countCls[i] : 0);
        const barData = { values: perClassMse.map((v, i) => ({ x: String(i), y: v })) };
        tfvis.render.barchart({ name: 'Per-class MSE', tab: 'Evaluation' }, barData);

        // simple table
        const table = document.createElement('table');
        table.style.borderCollapse = 'collapse';
        table.style.marginTop = '8px';
        table.innerHTML = '<tr><th style="padding:6px;border:1px solid #eee">Class</th><th style="padding:6px;border:1px solid #eee">MSE</th><th style="padding:6px;border:1px solid #eee">Count</th></tr>';
        for (let i = 0; i < 10; ++i) {
          const row = document.createElement('tr');
          row.innerHTML = `<td style="padding:6px;border:1px solid #f2f2f2">${i}</td><td style="padding:6px;border:1px solid #f2f2f2">${(perClassMse[i]||0).toFixed(6)}</td><td style="padding:6px;border:1px solid #f2f2f2">${countCls[i]}</td>`;
          table.appendChild(row);
        }
        visRoot.appendChild(table);
      } catch (err) {
        console.error(err);
        alert('Evaluation error: ' + (err && err.message));
      }
    });

    // TEST 5 RANDOM: display denoising results for five randomly selected images
    testFiveBtn.addEventListener('click', async () => {
      try {
        if (!testData || !model) { alert('Ensure test data is loaded and model is trained/loaded.'); return; }
        previewStrip.innerHTML = '';
        const k = 5;
        const noiseLevel = parseFloat(noiseRange.value) || 0.25;

        // Sample k random images
        const { orig, noisy, ys } = window.dataLoader.getRandomTestBatch(testData.xs, testData.ys, k, noiseLevel);
        // Predict denoised images
        const denoised = tf.tidy(() => model.predict(noisy));

        // For each of k images, create a small column showing original, noisy, denoised and metadata
        for (let i = 0; i < k; ++i) {
          const item = document.createElement('div');
          item.className = 'preview-item';

          // canvases
          const origCanv = document.createElement('canvas'); origCanv.className = 'preview-canvas';
          const noisyCanv = document.createElement('canvas'); noisyCanv.className = 'preview-canvas';
          const denoisedCanv = document.createElement('canvas'); denoisedCanv.className = 'preview-canvas';

          // scale set to 6 for good visibility
          origCanv.width = noisyCanv.width = denoisedCanv.width = 28 * 6;
          origCanv.height = noisyCanv.height = denoisedCanv.height = 28 * 6;

          // draw images (use tensor slices)
          const oi = orig.gather([i]);
          const ni = noisy.gather([i]);
          const di = denoised.gather([i]);

          window.dataLoader.draw28x28ToCanvas(oi, origCanv, 6);
          window.dataLoader.draw28x28ToCanvas(ni, noisyCanv, 6);
          window.dataLoader.draw28x28ToCanvas(di, denoisedCanv, 6);

          // compute per-image MSE and label
          const mseTensor = tf.tidy(() => di.sub(oi).square().mean());
          const mseVal = (await mseTensor.data())[0];
          const labelIdx = (await ys.argMax(-1).data())[i];

          // caption
          const caption = document.createElement('div');
          caption.className = 'muted';
          caption.style.textAlign = 'center';
          caption.style.fontSize = '13px';
          caption.innerHTML = `<strong>Label:</strong> ${labelIdx} <br><strong>MSE:</strong> ${mseVal.toFixed(6)}`;

          // layout: horizontal row of three canvases
          const row = document.createElement('div');
          row.style.display = 'flex';
          row.style.gap = '6px';
          row.appendChild(origCanv); row.appendChild(noisyCanv); row.appendChild(denoisedCanv);

          item.appendChild(row);
          item.appendChild(caption);
          previewStrip.appendChild(item);

          // dispose small tensors
          oi.dispose(); ni.dispose(); di.dispose(); mseTensor.dispose();
        }

        // dispose batches
        orig.dispose(); noisy.dispose(); ys.dispose(); denoised.dispose();
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
        appendLog('Saving model (download) ...');
        await model.save('downloads://mnist-denoiser');
        appendLog('Model download triggered.');
      } catch (err) {
        console.error(err);
        alert('Save error: ' + (err && err.message));
      }
    });

    // LOAD model from local files selected by user
    loadModelBtn.addEventListener('click', async () => {
      try {
        const jsonFile = uploadJson.files[0];
        const binFile = uploadWeights.files[0];
        if (!jsonFile || !binFile) { alert('Please select both model.json and weights.bin'); return; }
        appendLog('Loading model from files...');
        if (model) { model.dispose(); model = null; }
        const loaded = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
        model = loaded;
        modelInfo.textContent = modelSummaryToString(model);
        appendLog('Model loaded from files.');
        setButtonsForDataLoaded(true);
      } catch (err) {
        console.error(err);
        alert('Load model error: ' + (err && err.message));
      }
    });

    // Toggle tfjs-vis visor
    toggleVisor.addEventListener('click', () => {
      const v = tfvis.visor();
      if (v.isOpen()) { v.close(); appendLog('Visor closed.'); } else { v.open(); appendLog('Visor opened.'); }
    });

    // Initially disabled buttons until data loaded
    setButtonsForDataLoaded(false);

    // Ensure UI stays responsive on long operations: helper to yield control
    async function breathe() { return new Promise(r => requestAnimationFrame(r)); }

    // On unload, dispose resources
    window.addEventListener('beforeunload', () => {
      safeDispose(trainData && trainData.xs); safeDispose(trainData && trainData.ys);
      safeDispose(testData && testData.xs); safeDispose(testData && testData.ys);
      if (model) model.dispose();
    });
  });
})();
