// data-loader.js
// Utility functions for reading local MNIST CSV files and preparing tensors

async function parseCsvFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const lines = reader.result.split(/\r?\n/).filter(l => l.trim().length > 0);
        const images = [];
        const labels = [];
        for (const line of lines) {
          const values = line.split(',').map(v => parseFloat(v));
          const label = values[0];
          const pixels = values.slice(1).map(v => v / 255.0);
          images.push(pixels);
          labels.push(label);
        }
        resolve({images, labels});
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = err => reject(err);
    reader.readAsText(file);
  });
}

function tensorsFromData(images, labels) {
  const xs = tf.tensor2d(images).reshape([images.length, 28, 28, 1]);
  const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10);
  return {xs, ys};
}

async function loadTrainFromFiles(file) {
  const {images, labels} = await parseCsvFile(file);
  return tensorsFromData(images, labels);
}

async function loadTestFromFiles(file) {
  const {images, labels} = await parseCsvFile(file);
  return tensorsFromData(images, labels);
}

function splitTrainVal(xs, ys, valRatio = 0.1) {
  const total = xs.shape[0];
  const valSize = Math.floor(total * valRatio);
  const trainSize = total - valSize;
  const trainXs = xs.slice([0, 0, 0, 0], [trainSize, 28, 28, 1]);
  const trainYs = ys.slice([0, 0], [trainSize, 10]);
  const valXs = xs.slice([trainSize, 0, 0, 0], [valSize, 28, 28, 1]);
  const valYs = ys.slice([trainSize, 0], [valSize, 10]);
  return {trainXs, trainYs, valXs, valYs};
}

function getRandomTestBatch(xs, ys, k = 5) {
  const total = xs.shape[0];
  const indices = [];
  for (let i = 0; i < k; i++) {
    indices.push(Math.floor(Math.random() * total));
  }
  const batchXs = tf.gather(xs, indices);
  const batchYs = tf.gather(ys, indices);
  return {batchXs, batchYs, indices};
}

function draw28x28ToCanvas(tensor, canvas, scale = 4) {
  const [h, w] = [28, 28];
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(w, h);
  const data = tensor.dataSync();
  for (let i = 0; i < h * w; i++) {
    const val = Math.floor(data[i] * 255);
    imageData.data[i * 4 + 0] = val;
    imageData.data[i * 4 + 1] = val;
    imageData.data[i * 4 + 2] = val;
    imageData.data[i * 4 + 3] = 255;
  }
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = w; tmpCanvas.height = h;
  tmpCanvas.getContext('2d').putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(tmpCanvas, 0, 0, w * scale, h * scale);
}
