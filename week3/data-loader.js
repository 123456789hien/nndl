// data-loader.js
'use strict';

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (e) => reject(new Error('Failed to read file: ' + e.target.error));
    reader.readAsText(file);
  });
}

async function loadCSVFile(file) {
  const text = await readFileAsText(file);
  const lines = text.split(/\r?\n/);
  const images = [];
  const labels = [];

  for (let i = 0; i < lines.length; ++i) {
    const raw = lines[i].trim();
    if (!raw) continue;
    const parts = raw.split(',').map(s => s.trim());
    if (parts.length < 785) { console.warn(`Skipping CSV line ${i} (cols=${parts.length})`); continue; }
    const lab = parseInt(parts[0], 10);
    if (Number.isNaN(lab)) { console.warn(`Skipping CSV line ${i} (invalid label)`); continue; }
    const pix = new Array(784);
    let ok = true;
    for (let j = 0; j < 784; ++j) {
      const v = Number(parts[j + 1]);
      if (Number.isNaN(v)) { ok = false; break; }
      pix[j] = v / 255.0;
    }
    if (!ok) { console.warn(`Skipping CSV line ${i} (invalid pixel)`); continue; }
    images.push(pix);
    labels.push(lab);
  }

  if (images.length === 0) throw new Error('No valid rows found in CSV file: ' + file.name);

  const xs2d = tf.tensor2d(images, [images.length, 784], 'float32');
  const xs = xs2d.reshape([images.length, 28, 28, 1]);
  const labs = tf.tensor1d(labels, 'int32');
  const ys = tf.oneHot(labs, 10);

  return { xs, ys, labels: labs };
}

// Add noise to tensor (0-1 normalized)
function addNoise(xs, noiseStd = 0.25) {
  return tf.tidy(() => {
    const noise = tf.randomNormal(xs.shape, 0, noiseStd);
    return xs.add(noise).clipByValue(0, 1);
  });
}

// Draw a single 28x28 image to canvas
function draw28x28ToCanvas(tensor, canvas, scale = 3) {
  const [h, w, c] = tensor.shape;
  canvas.width = w * scale;
  canvas.height = h * scale;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(w, h);
  const data = tensor.dataSync();
  for (let i = 0; i < w * h; ++i) {
    const v = Math.floor(data[i] * 255);
    imgData.data[i * 4 + 0] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = w; tmpCanvas.height = h;
  tmpCanvas.getContext('2d').putImageData(imgData, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmpCanvas, 0, 0, w * scale, h * scale);
}

window.addNoise = addNoise;
window.draw28x28ToCanvas = draw28x28ToCanvas;
window.loadCSVFile = loadCSVFile;
