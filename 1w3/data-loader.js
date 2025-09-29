// data-loader.js
// File-based CSV parsing utilities for MNIST CSVs (label + 784 pixel values).
// Exposes:
//   loadTrainFromFiles(file) -> { xs, ys }
//   loadTestFromFiles(file)  -> { xs, ys }
//   splitTrainVal(xs, ys, valRatio=0.1) -> { trainXs, trainYs, valXs, valYs }
//   getRandomTestBatch(xs, ys, k=5, noiseLevel=0.25) -> { orig, noisy, ys }
//   addNoiseToBatch(xs, noiseLevel=0.25) -> tensor
//   draw28x28ToCanvas(tensor, canvas, scale=4)

(function () {
  // Helper: read file as text (simple)
  function _readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = (e) => reject(e);
      fr.readAsText(file);
    });
  }

  // Helper: parse CSV text into array of rows (arrays of strings)
  function _parseCsvText(text) {
    // Split by lines, robust to CRLF / LF. Remove blank lines.
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    return lines.map(line => line.split(',').map(s => s.trim()));
  }

  // Convert rows to tensors
  function _rowsToTensors(rows) {
    const N = rows.length;
    // Float32 for pixels. We'll allocate a single Float32Array of length N*784
    const pixels = new Float32Array(N * 784);
    const labels = new Uint8Array(N * 10); // one-hot

    for (let i = 0; i < N; ++i) {
      const r = rows[i];
      const lab = Number(r[0]) | 0;
      if (lab < 0 || lab > 9 || Number.isNaN(lab)) {
        // fallback to 0
        labels[i * 10 + 0] = 1;
      } else {
        labels[i * 10 + lab] = 1;
      }
      for (let p = 0; p < 784; ++p) {
        const v = Number(r[1 + p]) || 0;
        pixels[i * 784 + p] = v / 255.0;
      }
    }

    // Build tensors
    const xs = tf.tensor4d(pixels, [N, 28, 28, 1], 'float32'); // normalized [0,1]
    const ys = tf.tensor2d(labels, [N, 10], 'int32');
    return { xs, ys };
  }

  // Public: load train CSV file (returns {xs, ys})
  async function loadTrainFromFiles(file) {
    if (!file) throw new Error('No train file provided.');
    const text = await _readFileAsText(file);
    const rows = _parseCsvText(text);
    return _rowsToTensors(rows);
  }

  // Public: load test CSV file (returns {xs, ys})
  async function loadTestFromFiles(file) {
    if (!file) throw new Error('No test file provided.');
    const text = await _readFileAsText(file);
    const rows = _parseCsvText(text);
    return _rowsToTensors(rows);
  }

  // Split a dataset into train and validation sets.
  // Note: does not modify original tensors; returns new tensors (gathered).
  function splitTrainVal(xs, ys, valRatio = 0.1) {
    const N = xs.shape[0];
    const valCount = Math.max(1, Math.floor(N * valRatio));
    const trainCount = N - valCount;
    const indices = tf.util.createShuffledIndices(N);
    const trainIdx = indices.slice(0, trainCount);
    const valIdx = indices.slice(trainCount);

    const trainXs = tf.gather(xs, trainIdx);
    const trainYs = tf.gather(ys, trainIdx);
    const valXs = tf.gather(xs, valIdx);
    const valYs = tf.gather(ys, valIdx);

    return { trainXs, trainYs, valXs, valYs };
  }

  // Add Gaussian noise to a batch tensor. Returns a new tensor (caller must dispose).
  function addNoiseToBatch(xs, noiseLevel = 0.25) {
    return tf.tidy(() => {
      if (noiseLevel <= 0) return xs.clone();
      const noise = tf.randomNormal(xs.shape, 0, noiseLevel);
      const noisy = xs.add(noise).clipByValue(0, 1);
      return noisy;
    });
  }

  // Return random test batch; orig = clean images, noisy = noisy inputs, ys = one-hot labels
  function getRandomTestBatch(xs, ys, k = 5, noiseLevel = 0.25) {
    if (!xs || xs.shape[0] === 0) throw new Error('Empty xs');
    const N = xs.shape[0];
    const ids = tf.util.createShuffledIndices(N).slice(0, k);
    const orig = tf.gather(xs, ids);
    const labels = ys ? tf.gather(ys, ids) : null;
    const noisy = addNoiseToBatch(orig, noiseLevel);
    return { orig, noisy, ys: labels };
  }

  // Draw a [28,28] or [28,28,1] or [1,28,28,1] tensor to a canvas, scaling each pixel by `scale`.
  function draw28x28ToCanvas(tensor, canvas, scale = 4) {
    tf.tidy(() => {
      let t = tensor;
      if (t.rank === 4) t = t.squeeze([0]);         // [28,28,1]
      if (t.rank === 3 && t.shape[2] === 1) t = t.squeeze([2]); // [28,28]
      const data = t.mul(255).toInt().dataSync(); // 0..255
      const w = 28, h = 28;
      canvas.width = w * scale;
      canvas.height = h * scale;
      const ctx = canvas.getContext('2d');
      // create ImageData
      const img = ctx.createImageData(w, h);
      for (let i = 0; i < w * h; ++i) {
        const v = data[i];
        img.data[i * 4 + 0] = v;
        img.data[i * 4 + 1] = v;
        img.data[i * 4 + 2] = v;
        img.data[i * 4 + 3] = 255;
      }
      // draw to a temporary canvas then scale to keep crisp pixels
      const tmp = document.createElement('canvas');
      tmp.width = w; tmp.height = h;
      tmp.getContext('2d').putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    });
  }

  // Expose under window.dataLoader
  window.dataLoader = {
    loadTrainFromFiles,
    loadTestFromFiles,
    splitTrainVal,
    addNoiseToBatch,
    getRandomTestBatch,
    draw28x28ToCanvas
  };
})();
