'use strict';

let trainXs=null, trainYs=null, testXs=null, testYs=null;
let modelCNN=null, modelDenoiser=null;
let bestValAcc=0, trainStartTime=null;

const statusDiv=document.getElementById('data-status');
const logsDiv=document.getElementById('training-logs');
const metricsDiv=document.getElementById('metrics');
const modelInfo=document.getElementById('model-info');
const previewRow=document.getElementById('preview-row');

document.getElementById('load-data').addEventListener('click', onLoadData);
document.getElementById('train-cnn').addEventListener('click', onTrainCNN);
document.getElementById('train-denoiser').addEventListener('click', onTrainDenoiser);
document.getElementById('evaluate').addEventListener('click', onEvaluate);
document.getElementById('test-five').addEventListener('click', onTestFive);
document.getElementById('save-model').addEventListener('click', onSaveModel);
document.getElementById('load-model').addEventListener('click', onLoadModel);
document.getElementById('reset').addEventListener('click', onReset);
document.getElementById('toggle-visor').addEventListener('click',()=>tfvis.visor().toggle());

function safeDispose(t){try{if(t && typeof t.dispose==='function') t.dispose();}catch(e){console.warn('Dispose error',e);}}
function setStatus(txt){statusDiv.innerText=txt;}
function log(txt){logsDiv.innerText=txt; console.log(txt);}
function showModelSummary(m){modelInfo.innerText=''; m.summary(null,null,line=>{modelInfo.innerText+=line+'\n';});}
function countParams(m){try{return m.countParams();}catch(e){return 'n/a';}}

// ---------------- Data Loading ----------------
async function onLoadData(){
  try{
    const trainFile=document.getElementById('train-csv').files[0];
    const testFile=document.getElementById('test-csv').files[0];
    if(!trainFile || !testFile) throw new Error('Please select both train and test CSV files.');

    safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
    setStatus('Loading CSVs...');
    const trainData=await window.loadTrainFromFiles(trainFile);
    const testData=await window.loadTestFromFiles(testFile);
    trainXs=trainData.xs; trainYs=trainData.ys; testXs=testData.xs; testYs=testData.ys;
    setStatus(`Loaded Train:${trainXs.shape[0]} Test:${testXs.shape[0]}`);

    // show random 5
    onTestFive();
  }catch(e){setStatus('Error loading CSVs: '+e.message);}
}

// ---------------- Model Definitions ----------------
function createCNN(){
  const model=tf.sequential();
  model.add(tf.layers.conv2d({inputShape:[28,28,1],filters:16,kernelSize:3,activation:'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize:2}));
  model.add(tf.layers.conv2d({filters:32,kernelSize:3,activation:'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize:2}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units:64,activation:'relu'}));
  model.add(tf.layers.dense({units:10,activation:'softmax'}));
  model.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});
  return model;
}

function createDenoiser(){
  const input=tf.input({shape:[28,28,1]});
  const x=tf.layers.flatten().apply(input);
  const encoded=tf.layers.dense({units:128,activation:'relu'}).apply(x);
  const decoded=tf.layers.dense({units:28*28,activation:'sigmoid'}).apply(encoded);
  const output=tf.layers.reshape({targetShape:[28,28,1]}).apply(decoded);
  const model=tf.model({inputs:input,outputs:output});
  model.compile({optimizer:'adam',loss:'meanSquaredError'});
  return model;
}

// ---------------- Training ----------------
async function onTrainCNN(){
  if(!trainXs) return setStatus('Load data first');
  if(!modelCNN){modelCNN=createCNN(); showModelSummary(modelCNN);}
  const {trainXs:trX,trainYs:trY,valXs:valX,valYs:valY}=window.splitTrainVal(trainXs,trainYs);
  setStatus('Training CNN...');
  await modelCNN.fit(trX,trY,{epochs:3,batchSize:64,validationData:[valX,valY],callbacks:tfvis.show.fitCallbacks({name:'CNN Training'},['loss','val_loss','accuracy','val_accuracy'])});
  setStatus('CNN training done');
}

async function onTrainDenoiser(){
  if(!trainXs) return setStatus('Load data first');
  if(!modelDenoiser){modelDenoiser=createDenoiser(); showModelSummary(modelDenoiser);}
  const {trainXs:trX,valXs:valX}=window.splitTrainVal(trainXs,trainYs);
  const noisyX=window.addNoise(trX);
  const noisyValX=window.addNoise(valX);
  setStatus('Training Denoiser...');
  await modelDenoiser.fit(noisyX,trX,{epochs:3,batchSize:64,validationData:[noisyValX,valX],callbacks:tfvis.show.fitCallbacks({name:'Denoiser Training'},['loss','val_loss'])});
  setStatus('Denoiser training done');
  safeDispose(noisyX); safeDispose(noisyValX);
}

// ---------------- Evaluation ----------------
async function onEvaluate(){
  if(!modelCNN || !testXs) return setStatus('Load data and train CNN first');
  setStatus('Evaluating...');
  const preds=modelCNN.predict(testXs);
  const predLabels=preds.argMax(-1).dataSync();
  const trueLabels=testYs.argMax(-1).dataSync();
  safeDispose(preds);

  const confusion=tf.math.confusionMatrix(tf.tensor1d(trueLabels,'int32'),tf.tensor1d(predLabels,'int32'),10);
  const cmData=await confusion.array();
  const perClassAcc=cmData.map((row,i)=>{const sum=row.reduce((a,b)=>a+b,0);return sum?row[i]/sum:0;});
  metricsDiv.innerText='';
  metricsDiv.innerText='Per-class Accuracy:\n'+perClassAcc.map((v,i)=>`Class ${i}: ${(v*100).toFixed(2)}%`).join('\n');

  const cmDiv=document.createElement('div');
  cmDiv.style.width='300px'; cmDiv.style.height='300px';
  document.querySelector('.col.right').appendChild(cmDiv);
  tfvis.render.confusionMatrix(cmDiv,{values:cmData,labels:[0,1,2,3,4,5,6,7,8,9]});
  confusion.dispose();
}

// ---------------- Random 5 ----------------
function onTestFive(){
  if(!testXs) return;
  const {xs,ys}=window.getRandomTestBatch(testXs,testYs,5);
  previewRow.innerHTML='';
  const labels=ys.argMax(-1).dataSync();
  for(let i=0;i<5;i++){
    const div=document.createElement('div'); div.className='preview-item';
    const c=document.createElement('canvas'); div.appendChild(c);
    window.draw28x28ToCanvas(xs.slice([i,0,0,0],[1,28,28,1]),c);
    div.appendChild(document.createTextNode(labels[i]));
    previewRow.appendChild(div);
  }
  safeDispose(xs); safeDispose(ys);
}

// ---------------- Save/Load ----------------
async function onSaveModel(){
  if(modelCNN) await modelCNN.save('downloads://mnist-cnn');
  if(modelDenoiser) await modelDenoiser.save('downloads://mnist-denoiser');
}

async function onLoadModel(){
  try{
    const jsonFile=document.getElementById('upload-json').files[0];
    const binFile=document.getElementById('upload-weights').files[0];
    if(!jsonFile || !binFile) return setStatus('Select JSON and BIN files');
    const handler=tf.io.browserFiles([jsonFile,binFile]);
    const loadedModel=await tf.loadLayersModel(handler);
    if(loadedModel.outputs[0].shape[1]===10) modelCNN=loadedModel;
    else modelDenoiser=loadedModel;
    showModelSummary(loadedModel);
    setStatus('Model loaded successfully');
  }catch(e){setStatus('Error loading model: '+e.message);}
}

// ---------------- Reset ----------------
function onReset(){
  safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
  safeDispose(modelCNN); safeDispose(modelDenoiser);
  trainXs=trainYs=testXs=testYs=modelCNN=modelDenoiser=null;
  previewRow.innerHTML=''; metricsDiv.innerText=''; logsDiv.innerText=''; modelInfo.innerText='';
  setStatus('Reset done');
}
