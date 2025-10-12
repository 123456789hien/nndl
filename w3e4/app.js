// app.js
'use strict';

let trainXs = null, trainYs = null, testXs = null, testYs = null;
let modelCNN = null, modelDenoiser = null;
let bestValAcc = 0, trainStartTime = null;

const statusDiv = document.getElementById('data-status');
const logsDiv = document.getElementById('training-logs');
const metricsDiv = document.getElementById('metrics');
const modelInfo = document.getElementById('model-info');
const previewRow = document.getElementById('preview-row');

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('load-data').addEventListener('click', onLoadData);
  document.getElementById('train-cnn').addEventListener('click', onTrainCNN);
  document.getElementById('train-denoiser').addEventListener('click', onTrainDenoiser);
  document.getElementById('evaluate').addEventListener('click', onEvaluate);
  document.getElementById('test-five').addEventListener('click', onTestFive);
  document.getElementById('save-model').addEventListener('click', onSaveModel);
  document.getElementById('load-model').addEventListener('click', onLoadModel);
  document.getElementById('reset').addEventListener('click', onReset);
  document.getElementById('toggle-visor').addEventListener('click', () => tfvis.visor().toggle());
});

// ----- Utilities -----
function safeDispose(t) {
  try { if (t && typeof t.dispose === 'function') t.dispose(); } 
  catch (e) { console.warn('Dispose error', e); }
}

function setStatus(txt) { statusDiv.innerText = txt; }
function log(txt) { logsDiv.innerText = txt; console.log(txt); }

function showModelSummary(m) {
  modelInfo.innerText = '';
  m.summary(null, null, line => { modelInfo.innerText += line + '\n'; });
}

function countParams(m) {
  try { return m.countParams(); } 
  catch (e) { return 'n/a'; }
}

function safeFixed(v, digits=4) { return (v ?? 0).toFixed(digits); }

// ----- Load Data -----
async function onLoadData() {
  const trainFile = document.getElementById('train-csv').files[0];
  const testFile = document.getElementById('test-csv').files[0];
  if (!trainFile || !testFile) { setStatus('Please select both train and test CSV files'); return; }

  setStatus('Loading CSVs...');
  log('Loading training CSV...');
  safeDispose(trainXs); safeDispose(trainYs);
  safeDispose(testXs); safeDispose(testYs);

  const trainData = await window.loadTrainFromFiles(trainFile);
  const testData = await window.loadTestFromFiles(testFile);

  trainXs = trainData.xs; trainYs = trainData.ys;
  testXs = testData.xs; testYs = testData.ys;

  setStatus(`Loaded train: ${trainXs.shape[0]} samples, test: ${testXs.shape[0]} samples`);
}

// ----- Create CNN Model -----
function createCNNModel() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({inputShape:[28,28,1], filters:32, kernelSize:3, activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.conv2d({filters:64, kernelSize:3, activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({units:128, activation:'relu'}));
  m.add(tf.layers.dense({units:10, activation:'softmax'}));
  m.compile({optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy']});
  return m;
}

// ----- Train CNN -----
async function onTrainCNN() {
  if (!trainXs || !trainYs) { setStatus('Load data first'); return; }
  safeDispose(modelCNN);
  modelCNN = createCNNModel();
  showModelSummary(modelCNN);

  const {trainXs:trX, trainYs:trY, valXs:vX, valYs:vY} = window.splitTrainVal(trainXs, trainYs, 0.1);
  bestValAcc = 0; trainStartTime = performance.now();

  setStatus('Training CNN...');
  await modelCNN.fit(trX, trY, {
    epochs: 5,
    batchSize: 64,
    validationData:[vX,vY],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const valAccSafe = logs.val_acc ?? logs.val_accuracy ?? 0;
        logsDiv.innerText = `Epoch ${epoch+1}: loss=${safeFixed(logs.loss)} val_loss=${safeFixed(logs.val_loss)} acc=${safeFixed(logs.acc)} val_acc=${safeFixed(valAccSafe)}`;
        if (valAccSafe > bestValAcc) bestValAcc = valAccSafe;
        await tf.nextFrame();
      }
    }
  });

  trX.dispose(); trY.dispose(); vX.dispose(); vY.dispose();
  setStatus(`CNN Training completed. Best val_acc=${safeFixed(bestValAcc)}`);
}

// ----- Create Denoiser -----
function createDenoiser() {
  const inp = tf.input({shape:[28,28,1]});
  let x = tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(inp);
  x = tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  x = tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(x);
  x = tf.layers.upSampling2d({size:2}).apply(x);
  const out = tf.layers.conv2d({filters:1,kernelSize:3,padding:'same',activation:'sigmoid'}).apply(x);
  const m = tf.model({inputs:inp, outputs:out});
  m.compile({optimizer:'adam', loss:'meanSquaredError'});
  return m;
}

// ----- Train Denoiser -----
async function onTrainDenoiser() {
  if (!trainXs) { setStatus('Load data first'); return; }
  safeDispose(modelDenoiser);
  modelDenoiser = createDenoiser();
  showModelSummary(modelDenoiser);

  const {trainXs:trX, trainYs:trY, valXs:vX} = window.splitTrainVal(trainXs, trainYs, 0.1);
  const trXnoisy = window.addNoise(trX, 0.25);
  const vXnoisy = window.addNoise(vX, 0.25);

  setStatus('Training denoiser...');
  await modelDenoiser.fit(trXnoisy, trX, {
    epochs:5, batchSize:64,
    validationData:[vXnoisy, vX],
    callbacks: {
      onEpochEnd: async (epoch, logs)=>{
        logsDiv.innerText = `Epoch ${epoch+1}: loss=${safeFixed(logs.loss)} val_loss=${safeFixed(logs.val_loss)}`;
        await tf.nextFrame();
      }
    }
  });

  trX.dispose(); trY.dispose(); vX.dispose(); trXnoisy.dispose(); vXnoisy.dispose();
  setStatus('Denoiser training completed');
}

// ----- Evaluate CNN -----
async function onEvaluate() {
  if (!modelCNN || !testXs || !testYs) { setStatus('Train CNN and load test data first'); return; }
  setStatus('Evaluating...');
  const evalRes = await modelCNN.evaluate(testXs, testYs, {batchSize:64});
  const loss = Array.isArray(evalRes)? evalRes[0].dataSync()[0]:evalRes.dataSync()[0];
  const acc = Array.isArray(evalRes)? evalRes[1].dataSync()[0]:evalRes.dataSync()[1];

  // Confusion matrix
  const preds = modelCNN.predict(testXs).argMax(-1);
  const labels = testYs.argMax(-1);
  const predArr = Array.from(preds.dataSync());
  const labelArr = Array.from(labels.dataSync());
  const conf = Array.from({length:10},()=>Array(10).fill(0));
  for(let i=0;i<labelArr.length;i++) conf[labelArr[i]][predArr[i]]++;
  const perClassAcc = conf.map((row,i)=>{
    const totalRow = row.reduce((a,b)=>a+b,0)||1;
    return {label:String(i), value:row[i]/totalRow};
  });

  preds.dispose(); labels.dispose();

  metricsDiv.innerHTML = `
    Loss: ${safeFixed(loss)}<br>
    Accuracy: ${safeFixed(acc)}<br>
    Per-class accuracy:<br>
    ${perClassAcc.map(c=>`${c.label}: ${safeFixed(c.value)}`).join('<br>')}
  `;

  setStatus('Evaluation completed');
}

// ----- Test 5 Random -----
function onTestFive() {
  if (!modelCNN || !testXs || !testYs) { setStatus('Train CNN and load test data first'); return; }
  const {xs, ys, indices} = window.getRandomTestBatch(testXs, testYs, 5);
  previewRow.innerHTML = '';

  const preds = modelCNN.predict(xs).argMax(-1).dataSync();
  const labels = ys.argMax(-1).dataSync();

  for(let i=0;i<5;i++){
    const div=document.createElement('div'); div.className='preview-item';
    const c=document.createElement('canvas');
    window.draw28x28ToCanvas(xs.slice([i,0,0,0],[1,28,28,1]), c);
    const p=document.createElement('div');
    p.innerText=`Pred: ${preds[i]}`; 
    p.className = (preds[i]===labels[i])?'correct':'wrong';
    div.appendChild(c); div.appendChild(p);
    previewRow.appendChild(div);
  }

  xs.dispose(); ys.dispose();
  setStatus('Test 5 random samples completed');
}

// ----- Save/Load Model -----
async function onSaveModel() {
  if(!modelCNN) { setStatus('Train CNN first'); return; }
  await modelCNN.save('downloads://mnist_cnn');
  setStatus('Model saved (download)');
}

async function onLoadModel() {
  const jsonFile = document.getElementById('upload-json').files[0];
  const binFile = document.getElementById('upload-weights').files[0];
  if(!jsonFile || !binFile) { setStatus('Select JSON and BIN files'); return; }
  modelCNN = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
  showModelSummary(modelCNN);
  setStatus('Model loaded from files');
}

// ----- Reset -----
function onReset() {
  safeDispose(trainXs); safeDispose(trainYs);
  safeDispose(testXs); safeDispose(testYs);
  safeDispose(modelCNN); safeDispose(modelDenoiser);
  trainXs=trainYs=testXs=testYs=modelCNN=modelDenoiser=null;
  bestValAcc=0; trainStartTime=null;
  metricsDiv.innerHTML=''; previewRow.innerHTML=''; logsDiv.innerText=''; setStatus('Reset completed');
}
