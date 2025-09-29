// data-loader.js
// Handles CSV parsing, normalization, one-hot encoding, preview drawing, and adding noise.

async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = e => reject(new Error("Failed to read file: " + e.target.error));
    reader.readAsText(file);
  });
}

async function loadCSVFile(file) {
  const text = await readFileAsText(file); // always reliable
  const lines = text.trim().split(/\r?\n/);
  const images = [];
  const labels = [];

  for (const line of lines) {
    if (!line) continue; // skip empty line
    const parts = line.split(',').map(Number);
    if (parts.length !== 785) continue; // each row = label + 784 pixels
    const label = parts[0];
    const pixels = parts.slice(1).map(v => v / 255); // normalize 0-255 → 0-1
    images.push(pixels);
    labels.push(label);
  }

  // Build tensors
  const xs = tf.tensor2d(images).reshape([images.length, 28, 28, 1]);
  const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10);

  return {xs, ys};
}

async function loadTrainFromFiles(file) { return loadCSVFile(file); }
async function loadTestFromFiles(file) { return loadCSVFile(file); }

function splitTrainVal(xs, ys, valRatio = 0.1) {
  const total = xs.shape[0];
  const valSize = Math.floor(total * valRatio);
  const trainSize = total - valSize;
  return {
    trainXs: xs.slice([0,0,0,0],[trainSize,28,28,1]),
    trainYs: ys.slice([0,0],[trainSize,10]),
    valXs: xs.slice([trainSize,0,0,0],[valSize,28,28,1]),
    valYs: ys.slice([trainSize,0],[valSize,10]),
  };
}

// Add Gaussian noise to data
function addNoise(xs, noiseStd=0.3) {
  return tf.tidy(() => {
    const noise = tf.randomNormal(xs.shape,0,noiseStd);
    return xs.add(noise).clipByValue(0,1);
  });
}

// Get random test batch
function getRandomTestBatch(xs, ys, k=5) {
  const total = xs.shape[0];
  const idx = tf.util.createShuffledIndices(total).slice(0,k);
  const imgs = [];
  const labs = [];
  idx.forEach(i=>{
    imgs.push(xs.slice([i,0,0,0],[1,28,28,1]));
    labs.push(ys.slice([i,0],[1,10]));
  });
  return {
    xs: tf.concat(imgs,0),
    ys: tf.concat(labs,0),
    indices: idx
  };
}

// Draw image tensor to canvas
function draw28x28ToCanvas(tensor, canvas, scale=4) {
  const [h,w] = [28,28];
  const ctx = canvas.getContext('2d');
  canvas.width = w*scale;
  canvas.height = h*scale;
  const data = tensor.dataSync();
  const imgData = ctx.createImageData(w,h);
  for(let i=0;i<data.length;i++){
    const val = Math.floor(data[i]*255);
    imgData.data[i*4+0]=val;
    imgData.data[i*4+1]=val;
    imgData.data[i*4+2]=val;
    imgData.data[i*4+3]=255;
  }
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width=w;tempCanvas.height=h;
  tempCanvas.getContext('2d').putImageData(imgData,0,0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tempCanvas,0,0,w*scale,h*scale);
}

// export to window
window.loadTrainFromFiles = loadTrainFromFiles;
window.loadTestFromFiles = loadTestFromFiles;
window.splitTrainVal = splitTrainVal;
window.getRandomTestBatch = getRandomTestBatch;
window.draw28x28ToCanvas = draw28x28ToCanvas;
window.addNoise = addNoise;
