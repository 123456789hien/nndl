// data-loader.js
// CSV parsing + normalization + tensor helpers for MNIST (label + 784 pixels per row).
// Exports functions to window so app.js can call them from the browser environment.

/*
  Important:
  - Each CSV row: label (0-9), 784 pixel values (0-255), no header.
  - Pixels normalized to [0,1], reshaped to [N,28,28,1]
  - Labels converted to one-hot tensors depth 10
*/

async function textFromFile(file) {
  // For typical CSVs read fully; for enormous files we would stream. For student demo readAsText is acceptable.
  return await file.text();
}

/**
 * Parse CSV text into arrays then tensors.
 * Skips empty lines. Robust to trailing commas/spaces.
 */
function parseMNISTCSV(text) {
  const lines = text.split(/\r?\n/);
  const images = [];
  const labels = [];
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i].trim();
    if (!raw) continue;
    const parts = raw.split(',').map(s => s.trim());
    if (parts.length < 785) continue; // malformed or header
    // first value label
    const lab = parseInt(parts[0], 10);
    if (Number.isNaN(lab)) continue;
    // next 784 values
    const pixels = new Array(784);
    let ok = true;
    for (let j = 0; j < 784; j++) {
      const v = Number(parts[j + 1]);
      if (Number.isNaN(v)) { ok = false; break; }
      pixels[j] = v / 255.0;
    }
    if (!ok) continue;
    images.push(pixels);
    labels.push(lab);
  }
  return { images, labels };
}

/**
 * Create tensors from parsed arrays. Returns { xs, ys } where:
 * - xs: float32 tensor shape [N,28,28,1]
 * - ys: one-hot float32 tensor shape [N,10]
 */
function arraysToTensors(images, labels) {
  if (!images.length) {
    throw new Error('No valid rows found in CSV');
  }
  // xs: Tensor2d -> reshape
  const xs2d = tf.tensor2d(images, [images.length, 784], 'float32');
  const xs = xs2d.reshape([images.length, 28, 28, 1]);
  const labs = tf.tensor1d(labels, 'int32');
  const ys = tf.oneHot(labs, 10).toFloat();
  // dispose intermediate if needed (xs2d and labs will be kept by engine until they are garbage collected or disposed)
  labs.dispose();
  return { xs, ys };
}

/**
 * Load train/test file (same behaviour) and return tensors.
 * Caller should dispose xs, ys when done.
 */
async function loadTrainFromFiles(file) {
  const text = await textFromFile(file);
  const { images, labels } = parseMNISTCSV(text);
  return arraysToTensors(images, labels);
}
async function loadTestFromFiles(file) {
  const text = await textFromFile(file);
  const { images, labels } = parseMNISTCSV(text);
  return arraysToTensors(images, labels);
}

/**
 * Split training set into train/validation (no copying via slicing tensors).
 * Returns small views — caller must manage disposal of returned tensors separately.
 */
function splitTrainVal(xs, ys, valRatio = 0.1) {
  const total = xs.shape[0];
  const valSize = Math.max(1, Math.floor(total * valRatio));
  const trainSize = total - valSize;
  const trainXs = xs.slice([0, 0, 0, 0], [trainSize, 28, 28, 1]);
  const trainYs = ys.slice([0, 0], [trainSize, 10]);
  const valXs = xs.slice([trainSize, 0, 0, 0], [valSize, 28, 28, 1]);
  const valYs = ys.slice([trainSize, 0], [valSize, 10]);
  return { trainXs, trainYs, valXs, valYs };
}

/**
 * Add Gaussian noise tensor to an image batch. Returns a new tensor.
 * Note: caller is responsible for disposing returned tensor.
 */
function addNoise(xs, noiseStd = 0.25) {
  return tf.tidy(() => {
    const noise = tf.randomNormal(xs.shape, 0, noiseStd, 'float32');
    const noisy = xs.add(noise).clipByValue(0, 1);
    // Return a tensor that escapes the tidy (so cannot be auto-disposed).
    return noisy.clone();
  });
}

/**
 * Get k random samples from test set. Returns { xs, ys, indices } where xs and ys are tensors
 * of shape [k,28,28,1] and [k,10] respectively.
 *
 * Note: Returned tensors must be disposed by caller.
 */
function getRandomTestBatch(xs, ys, k = 5) {
  const total = xs.shape[0];
  if (k > total) k = total;
  const idxs = tf.util.createShuffledIndices(total).slice(0, k);
  const imgs = [];
  const labs = [];
  for (let i = 0; i < idxs.length; i++) {
    imgs.push(xs.slice([idxs[i], 0, 0, 0], [1, 28, 28, 1]));
    labs.push(ys.slice([idxs[i], 0], [1, 10]));
  }
  const xsBatch = tf.concat(imgs, 0);
  const ysBatch = tf.concat(labs, 0);
  // Dispose the small single-row tensors (slices left behind)
  imgs.forEach(t => t.dispose());
  labs.forEach(t => t.dispose());
  return { xs: xsBatch, ys: ysBatch, indices: idxs };
}

/**
 * Draw [28,28,1] tensor to canvas (scale many pixels up).
 * Accepts either a 2D or 3D tensor; it will read synchronously via dataSync.
 */
function draw28x28ToCanvas(tensor, canvas, scale = 4) {
  // Ensure tensor is CPU-synced
  const t = tensor.reshape([28, 28]);
  const data = t.dataSync(); // synchronous read
  const w = 28, h = 28;
  const ctx = canvas.getContext('2d');
  canvas.width = w * scale;
  canvas.height = h * scale;
  // create ImageData at original resolution then scale up with imageSmoothing disabled
  const imgData = new ImageData(w, h);
  for (let i = 0; i < data.length; ++i) {
    const v = Math.max(0, Math.min(255, Math.round(data[i] * 255)));
    imgData.data[i * 4 + 0] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  tmp.getContext('2d').putImageData(imgData, 0, 0);
  const destCtx = ctx;
  destCtx.imageSmoothingEnabled = false;
  destCtx.clearRect(0, 0, canvas.width, canvas.height);
  destCtx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  t.dispose();
}

//
// Expose helpers to global scope for app.js to use
//
window.loadTrainFromFiles = loadTrainFromFiles;
window.loadTestFromFiles = loadTestFromFiles;
window.splitTrainVal = splitTrainVal;
window.getRandomTestBatch = getRandomTestBatch;
window.addNoise = addNoise;
window.draw28x28ToCanvas = draw28x28ToCanvas;
