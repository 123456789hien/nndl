// app.js
'use strict';

let trainData, testData;
let modelCNN, modelDenoiser;

// UI
const dataStatus = document.getElementById('data-status');
const logs = document.getElementById('training-logs');
const previewRow = document.getElementById('preview-row');
const modelInfo = document.getElementById('model-info');

// ======================== CNN & Denoiser ========================
function buildCNN() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 32, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return model;
}

function buildDenoiser() {
  const input = tf.input({ shape: [28, 28, 1] });
  const x = tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(input);
  const x2 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x);
  const x3 = tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(x2);
  const x4 = tf.layers.upSampling2d({ size: 2 }).apply(x3);
  const out = tf.layers.conv2d({ filters: 1, kernelSize: 3, activation: 'sigmoid', padding: 'same' }).apply(x4);
  const model = tf.model({ inputs: input, outputs: out });
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  return model;
}

// ======================== Event Handlers ========================
document.getElementById('load-data').onclick = async () => {
  const trainFile = document.getElementById('train-csv').files[0];
  const testFile = document.getElementById('test-csv').files[0];
  if (!trainFile || !testFile) { alert('Select CSV files'); return; }

  dataStatus.innerText = 'Loading train CSV...';
  trainData = await window.loadCSVFile(trainFile);
  dataStatus.innerText = 'Loading test CSV...';
  testData = await window.loadCSVFile(testFile);
  dataStatus.innerText = `Loaded Train=${trainData.labels.shape[0]} | Test=${testData.labels.shape[0]}`;
};

document.getElementById('train-cnn').onclick = async () => {
  if (!trainData) { alert('Load data first'); return; }
  modelCNN = buildCNN();
  modelInfo.innerText = modelCNN.summary();
  logs.innerText = 'Training CNN...';

  await modelCNN.fit(trainData.xs, trainData.ys, {
    epochs: 3, batchSize: 64,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (epoch, logs_) => {
        logs.innerText += `\nEpoch ${epoch+1}: loss=${logs_.loss.toFixed(4)}, acc=${logs_.accuracy.toFixed(4)}`;
      }
    }
  });
  logs.innerText += '\nCNN training completed';
};

document.getElementById('train-denoiser').onclick = async () => {
  if (!trainData) { alert('Load data first'); return; }
  modelDenoiser = buildDenoiser();
  logs.innerText = 'Training denoiser...';

  const noisy = window.addNoise(trainData.xs, 0.25);
  await modelDenoiser.fit(noisy, trainData.xs, {
    epochs: 3, batchSize: 64,
    validationSplit: 0.1,
    callbacks: tfvis.show.fitCallbacks({ name: 'Denoiser Training', tab: 'Training' }, ['loss', 'val_loss'])
  });
  logs.innerText += '\nDenoiser training completed';
};

document.getElementById('evaluate').onclick = async () => {
  if (!modelCNN || !testData) { alert('Train CNN and load test data first'); return; }
  const evalOut = modelCNN.evaluate(testData.xs, testData.ys);
  const loss = (await evalOut[0].data())[0].toFixed(4);
  const acc = (await evalOut[1].data())[0].toFixed(4);
  logs.innerText += `\nTest Loss=${loss}, Test Accuracy=${acc}`;

  // Per-class accuracy
  const preds = modelCNN.predict(testData.xs).argMax(-1);
  const labels = testData.labels;
  const correctCounts = new Array(10).fill(0);
  const totalCounts = new Array(10).fill(0);
  const predData = await preds.data();
  const labelData = await labels.data();
  for (let i = 0; i < predData.length; ++i) {
    totalCounts[labelData[i]] += 1;
    if (predData[i] === labelData[i]) correctCounts[labelData[i]] += 1;
  }
  const perClassAcc = correctCounts.map((v,i)=>v/totalCounts[i]);
  tfvis.render.barchart({name:'Per-Class Accuracy', tab:'Evaluation'}, {values: perClassAcc.map((v,i)=>({x:i,y:v}))});
};

document.getElementById('test-five').onclick = async () => {
  if (!testData) return alert('Load test data');
  const n = 5, total = testData.xs.shape[0];
  previewRow.innerHTML = '';
  const indices = Array.from({length: n},()=>Math.floor(Math.random()*total));
  const batchXs = tf.gather(testData.xs, indices);
  const batchLabels = tf.gather(testData.labels, indices);
  const noisy = window.addNoise(batchXs,0.25);

  let displayXs = noisy;
  if (modelDenoiser) displayXs = modelDenoiser.predict(noisy);

  const labelsArr = await batchLabels.data();
  const predsArr = await modelCNN.predict(displayXs).argMax(-1).data();
  
  for (let i=0;i<n;i++){
    const canvas = document.createElement('canvas');
    const div = document.createElement('div');
    div.className = 'preview-item';
    const labelText = document.createElement('div');
    const correct = labelsArr[i]===predsArr[i];
    labelText.innerText = `GT:${labelsArr[i]} Pred:${predsArr[i]}`;
    labelText.className = correct?'correct':'wrong';
    window.draw28x28ToCanvas(displayXs.slice([i,0,0,0],[1,28,28,1]).reshape([28,28,1]), canvas,3);
    div.appendChild(canvas); div.appendChild(labelText);
    previewRow.appendChild(div);
  }
};

// ======================== Save/Load ========================
document.getElementById('save-model').onclick = async () => {
  if (!modelDenoiser) return alert('Train denoiser first');
  await modelDenoiser.save('downloads://mnist-denoiser');
};

document.getElementById('load-model').onclick = async () => {
  const jsonFile = document.getElementById('upload-json').files[0];
  const binFile = document.getElementById('upload-weights').files[0];
  if (!jsonFile || !binFile) return alert('Select both JSON and BIN files');
  modelDenoiser = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
  logs.innerText += '\nDenoiser model loaded';
};

// ======================== Reset ========================
document.getElementById('reset').onclick = () => {
  trainData = testData = null;
  modelCNN = modelDenoiser = null;
  dataStatus.innerText = logs.innerText = previewRow.innerHTML = modelInfo.innerText = '';
};
