// data-loader.js
// CSV parsing + normalization + tensor helpers for MNIST (label + 784 pixels per row).
// This file uses FileReader/TextDecoder (readAsText) and exposes helpers on window.
// It carefully disposes intermediate tensors where applicable.

'use strict';

/**
 * Read a file as text using FileReader. Returns a Promise<string>.
 * Using FileReader is more compatible/robust across browsers than file.text().
 */
function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (e) => reject(new Error('Failed to read file: ' + e.target.error));
    reader.readAsText(file);
  });
}

/**
 * Parse a CSV text for MNIST rows:
 * - each row: label (0-9), 784 pixel values (0-255), no header
 * - ignore empty lines
 * Returns { xs: Tensor4d [N,28,28,1], ys: Tensor2d [N,10] }
 *
 * Important: caller is responsible for disposing returned tensors.
 */
async function loadCSVFile(file) {
  const text = await readFileAsText(file);
  // split lines robustly, preserve empty lines for skip
  const lines = text.split(/\r?\n/);
  const images = [];
  const labels = [];

  for (let i = 0; i < lines.length; ++i) {
    const raw = lines[i].trim();
    if (!raw) continue;
    const parts = raw.split(',').map(s => s.trim());
    if (parts.length < 785) {
      // skip malformed lines; warn to console for debugging
      console.warn(`Skipping CSV line ${i} (cols=${parts.length})`);
      continue;
    }
    const lab = parseInt(parts[0], 10);
    if (Number.isNaN(lab)) {
      console.warn(`Skipping CSV line ${i} (invalid label)`);
      continue;
    }
    // Convert pixel strings to floats in [0,1]
    const pix = new Array(784);
    let ok = true;
    for (let j = 0; j < 784; ++j) {
      const v = Number(parts[j + 1]);
      if (Number.isNaN(v)) { ok = false; break; }
      pix[j] = v / 255.0;
    }
    if (!ok) {
      console.warn(`Skipping CSV line ${i} (invalid pixel)`);
      continue;
    }
    images.push(pix);
    labels.push(lab);
  }

  if (images.length === 0) throw new Error('No valid rows found in CSV file: ' + file.name);

  // build tensors
  const xs2d = tf.tensor2d(images, [images.length, 784], 'float32'); // [N,784]
  const xs = xs2d.reshape([images.length, 28, 28, 1]); // [N,28,28,1]
  const labs = tf.tensor1d(labels, 'int32');
  const ys = tf.oneHot(labs, 10).toFloat(); // [N,10]

  // dispose intermediates
  xs2d.dispose();
  labs.dispose();

  console.log(`Loaded ${images.length} samples from ${file.name}`);
  return { xs, ys };
}

// Exposed functions for app.js
async function loadTrainFromFiles(file) { return loadCSVFile(file); }
async function loadTestFromFiles(file) { return loadCSVFile(file); }

/**
 * Split training set into train/validation views.
 * Returns slices (not copies). Caller must manage disposing of returned tensors.
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
 * Add Gaussian noise to image batch xs. Returns a new tensor (caller must dispose).
 * Uses tf.tidy then clone so returned tensor survives outside tidy.
 */
function addNoise(xs, noiseStd = 0.25) {
  return tf.tidy(() => {
    const noise = tf.randomNormal(xs.shape, 0, noiseStd, 'float32');
    const noisy = xs.add(noise).clipByValue(0, 1);
    return noisy.clone();
  });
}

/**
 * Get k random samples from dataset xs, ys.
 * Returns { xs, ys, indices } where xs:[k,28,28,1] ys:[k,10].
 * Caller must dispose returned tensors.
 */
function getRandomTestBatch(xs, ys, k = 5) {
  const total = xs.shape[0];
  if (k > total) k = total;
  const idx = tf.util.createShuffledIndices(total).slice(0, k);
  const imgs = [];
  const labs = [];
  for (let i = 0; i < idx.length; ++i) {
    imgs.push(xs.slice([idx[i], 0, 0, 0], [1, 28, 28, 1]));
    labs.push(ys.slice([idx[i], 0], [1, 10]));
  }
  const xsBatch = tf.concat(imgs, 0);
  const ysBatch = tf.concat(labs, 0);
  // dispose the individual slices (the concat created new tensors)
  imgs.forEach(t => t.dispose());
  labs.forEach(t => t.dispose());
  return { xs: xsBatch, ys: ysBatch, indices: idx };
}

/**
 * Draw a [28,28,1] or [28,28] tensor to canvas. scale default 4 => 112x112 canvas.
 * This function reads dataSync (synchronous) for simplicity in browser UI.
 */
function draw28x28ToCanvas(tensor, canvas, scale = 4) {
  // ensure shape [28,28]
  let t = tensor;
  if (tensor.rank === 4) {
    t = tensor.reshape([28, 28]);
  } else if (tensor.rank === 3 && tensor.shape[2] === 1) {
    t = tensor.reshape([28, 28]);
  }
  const data = t.dataSync();
  const w = 28, h = 28;
  canvas.width = w * scale;
  canvas.height = h * scale;
  const ctx = canvas.getContext('2d');
  // build ImageData at native resolution then scale up
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < data.length; ++i) {
    const v = Math.max(0, Math.min(255, Math.round(data[i] * 255)));
    img.data[i * 4 + 0] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  tmp.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  if (t !== tensor) t.dispose();
}

// Expose to window for app.js
window.loadTrainFromFiles = loadTrainFromFiles;
window.loadTestFromFiles = loadTestFromFiles;
window.splitTrainVal = splitTrainVal;
window.getRandomTestBatch = getRandomTestBatch;
window.draw28x28ToCanvas = draw28x28ToCanvas;
window.addNoise = addNoise;
