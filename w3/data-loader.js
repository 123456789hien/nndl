// data-loader.js
// Responsible for parsing uploaded MNIST CSV files and producing tensors suitable for training/testing.
// CSV format assumed: label (0-9), followed by 784 pixel values (0-255), no header.
// Exports async functions to load train/test, split train/val, get random preview batches, and draw to canvas.

// Utility: parse CSV text into rows (split robustly, ignore empty lines)
function _parseCsvText(text) {
  // Support CRLF and LF; ignore blank lines
  const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
  return lines.map(line => {
    // split by comma, trim spaces
    return line.split(',').map(s => s.trim());
  });
}

// Convert parsed rows into tensors {xs, ys}
// xs: float32 tensor shape [N,28,28,1] values in [0,1]
// ys: one-hot depth 10, shape [N,10] (we may not need labels for denoiser but keep for compatibility)
function _rowsToTensors(rows) {
  const N = rows.length;
  const xsBuffer = new Float32Array(N * 28 * 28);
  const ysBuffer = new Uint8Array(N * 10);

  for (let i = 0; i < N; ++i) {
    const r = rows[i];
    const label = Number(r[0]) | 0;
    // one-hot
    ysBuffer[i * 10 + label] = 1;
    // pixels
    for (let p = 0; p < 784; ++p) {
      const v = Number(r[1 + p]) || 0;
      xsBuffer[i * 784 + p] = v / 255.0;
    }
  }

  // create tensors
  // reshape xs to [N,28,28,1]
  const xs = tf.tensor4d(xsBuffer, [N, 28, 28, 1], 'float32');
  const ys = tf.tensor2d(ysBuffer, [N, 10], 'int32');
  return { xs, ys };
}

// Async file reader that returns text
function _readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = (e) => reject(e);
    fr.readAsText(file);
  });
}

// Public API

/**
 * loadTrainFromFiles(file): read train CSV file and return {xs, ys}
 * @param {File} file - CSV file
 * @returns {Promise<{xs:tf.Tensor4D, ys:tf.Tensor2D}>}
 */
async function loadTrainFromFiles(file) {
  if (!file) throw new Error('No train file provided.');
  const text = await _readFileAsText(file);
  const rows = _parseCsvText(text);
  const { xs, ys } = _rowsToTensors(rows);
  return { xs, ys };
}

/**
 * loadTestFromFiles(file): read test CSV and return {xs, ys}
 * @param {File} file - CSV file
 * @returns {Promise<{xs:tf.Tensor4D, ys:tf.Tensor2D}>}
 */
async function loadTestFromFiles(file) {
  if (!file) throw new Error('No test file provided.');
  const text = await _readFileAsText(file);
  const rows = _parseCsvText(text);
  const { xs, ys } = _rowsToTensors(rows);
  return { xs, ys };
}

/**
 * splitTrainVal(xs, ys, valRatio=0.1)
 * Splits tensors into training & validation sets. Returns {trainXs, trainYs, valXs, valYs}
 * Note: returns new tensors (slices) and does NOT dispose the inputs.
 */
function splitTrainVal(xs, ys, valRatio = 0.1) {
  const N = xs.shape[0];
  const valCount = Math.max(1, Math.floor(N * valRatio));
  const trainCount = N - valCount;

  // shuffle indices
  const indices = tf.util.createShuffledIndices(N);

  const trainIdx = indices.slice(0, trainCount);
  const valIdx = indices.slice(trainCount);

  const trainXs = tf.gather(xs, trainIdx);
  const trainYs = tf.gather(ys, trainIdx);
  const valXs = tf.gather(xs, valIdx);
  const valYs = tf.gather(ys, valIdx);

  return { trainXs, trainYs, valXs, valYs };
}

/**
 * addNoiseToBatch(xs, noiseLevel)
 * Adds random Gaussian noise clipped to [0,1]. Returns a new tensor.
 * xs expected shape [k,28,28,1]
 */
function addNoiseToBatch(xs, noiseLevel = 0.25) {
  return tf.tidy(() => {
    if (noiseLevel <= 0) return xs.clone();
    const noise = tf.randomNormal(xs.shape, 0, noiseLevel);
    const noisy = xs.add(noise).clipByValue(0, 1);
    return noisy;
  });
}

/**
 * getRandomTestBatch(xs, ys, k=5, noiseLevel=0.25)
 * Returns an object with:
 *  - orig: tensor [k,28,28,1] original clean images
 *  - noisy: tensor [k,28,28,1] noisy inputs
 *  - ys: labels one-hot [k,10]
 */
function getRandomTestBatch(xs, ys, k = 5, noiseLevel = 0.25) {
  if (!xs || xs.shape[0] === 0) throw new Error('Test xs empty.');
  const N = xs.shape[0];
  const ids = tf.util.createShuffledIndices(N).slice(0, k);
  const orig = tf.gather(xs, ids);
  const ybatch = ys ? tf.gather(ys, ids) : null;
  const noisy = addNoiseToBatch(orig, noiseLevel);
  return { orig, noisy, ys: ybatch };
}

/**
 * draw28x28ToCanvas(tensor, canvas, scale=4)
 * tensor: [28,28] or [28,28,1] or [1,28,28,1]
 * canvas: HTMLCanvasElement
 */
function draw28x28ToCanvas(tensor, canvas, scale = 4) {
  // Accept several shapes; create an imageData and render
  tf.tidy(() => {
    let t = tensor;
    if (t.rank === 4) t = t.squeeze([0]); // [28,28,1]
    if (t.rank === 3 && t.shape[2] === 1) t = t.squeeze([2]); // [28,28]
    const data = t.mul(255).toInt().dataSync(); // values 0..255
    const ctx = canvas.getContext('2d');
    const w = 28;
    const h = 28;
    canvas.width = w * scale;
    canvas.height = h * scale;
    // create ImageData with RGBA
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; ++i) {
      const v = data[i];
      img.data[i * 4 + 0] = v; // r
      img.data[i * 4 + 1] = v; // g
      img.data[i * 4 + 2] = v; // b
      img.data[i * 4 + 3] = 255; // a
    }
    // scale up using a temporary canvas
    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    tmp.getContext('2d').putImageData(img, 0, 0);
    // draw scaled image onto destination canvas with pixelated rendering
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  });
}

// Export functions to global scope (browser)
window.dataLoader = {
  loadTrainFromFiles,
  loadTestFromFiles,
  splitTrainVal,
  getRandomTestBatch,
  addNoiseToBatch,
  draw28x28ToCanvas
};
