// app.js
'use strict';

/* global tf, tfvis */

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
document.getElementById('toggle-visor').addEventListener('click', ()=>tfvis.visor().toggle());

function safeDispose(t){ try{ if(t && typeof t.dispose==='function') t.dispose(); }catch(e){console.warn('Dispose error',e); } }
function setStatus(txt){statusDiv.innerText=txt;}
function log(txt){logsDiv.innerText=txt; console.log(txt);}

function showModelSummary(m){ modelInfo.innerText=''; m.summary(null,null,line=>{modelInfo.innerText+=line+'\n';}); }
function countParams(m){ try{ return m.countParams(); }catch(e){ return 'n/a'; } }

async function onLoadData(){
  try{
    const trainFile=document.getElementById('train-csv').files[0];
    const testFile=document.getElementById('test-csv').files[0];
    if(!trainFile||!testFile) throw new Error('Please select both train and test CSV files.');
    safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
    trainXs=trainYs=testXs=testYs=null;
    previewRow.innerHTML=''; modelInfo.innerText='';
    setStatus('Loading CSV files...'); await tf.nextFrame();
    const t0=performance.now();
    const train=await window.loadTrainFromFiles(trainFile);
    const test=await window.loadTestFromFiles(testFile);
    const t1=performance.now();
    trainXs=train.xs; trainYs=train.ys; testXs=test.xs; testYs=test.ys;
    if(trainXs.shape[1]!==28||trainXs.shape[2]!==28) throw new Error('Train images not 28x28');
    if(testXs.shape[1]!==28||testXs.shape[2]!==28) throw new Error('Test images not 28x28');
    setStatus(`Loaded train: ${trainXs.shape[0]} samples, test: ${testXs.shape[0]} samples (in ${(t1-t0).toFixed(0)}ms)`);
  }catch(err){ setStatus('Error: '+err.message); console.error(err);}
}

function buildCNN(){
  const m=tf.sequential();
  m.add(tf.layers.conv2d({inputShape:[28,28,1],filters:32,kernelSize:3,activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.conv2d({filters:64,kernelSize:3,activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({units:128,activation:'relu'}));
  m.add(tf.layers.dense({units:10,activation:'softmax'}));
  m.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});
  return m;
}

async function onTrainCNN(){
  if(!trainXs||!trainYs){ setStatus('Load data first'); return; }
  safeDispose(modelCNN); modelCNN=null; bestValAcc=0;
  setStatus('Training CNN...');
  modelCNN=buildCNN();
  showModelSummary(modelCNN);
  const {trainXs:trX,trainYs:trY,valXs:valX,valYs:valY}=window.splitTrainVal(trainXs,trainYs,0.1);
  await modelCNN.fit(trX,trY,{
    epochs:3,batchSize:64,validationData:[valX,valY],
    callbacks:{
      onEpochEnd:(epoch,logs)=>{ log(`Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)}, val_acc=${(logs.val_accuracy*100).toFixed(2)}%`); },
      onTrainEnd:()=>{ log('CNN training finished'); safeDispose(trX); safeDispose(trY); safeDispose(valX); safeDispose(valY);}
    }
  });
}

function buildDenoiser(){
  const input=tf.input({shape:[28,28,1]});
  const x1=tf.layers.flatten().apply(input);
  const encoded=tf.layers.dense({units:64,activation:'relu'}).apply(x1);
  const decoded=tf.layers.dense({units:784,activation:'sigmoid'}).apply(encoded);
  const output=tf.layers.reshape({targetShape:[28,28,1]}).apply(decoded);
  const m=tf.model({inputs:input,outputs:output});
  m.compile({optimizer:'adam',loss:'meanSquaredError'});
  return m;
}

async function onTrainDenoiser(){
  if(!trainXs){ setStatus('Load data first'); return; }
  safeDispose(modelDenoiser); modelDenoiser=null;
  setStatus('Training Denoiser...');
  modelDenoiser=buildDenoiser();
  showModelSummary(modelDenoiser);
  const {trainXs:trX,valXs:valX}=window.splitTrainVal(trainXs,trainYs,0.1);
  const trXnoisy=window.addNoise(trX,0.25);
  const valXnoisy=window.addNoise(valX,0.25);
  await modelDenoiser.fit(trXnoisy,trX,{epochs:3,batchSize:64,validationData:[valXnoisy,valX],
    callbacks:{ onEpochEnd:(epoch,logs)=>{ log(`Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)}, val_loss=${logs.val_loss.toFixed(4)}`); },
    onTrainEnd:()=>{ safeDispose(trX); safeDispose(valX); safeDispose(trXnoisy); safeDispose(valXnoisy); log('Denoiser training finished');} }});
}

async function onEvaluate(){
  if(!modelCNN||!testXs||!testYs){ setStatus('Model or test data missing'); return; }
  const evalRes=await modelCNN.evaluate(testXs,testYs,{batchSize:64});
  const acc=(evalRes[1].dataSync()[0]*100).toFixed(2);
  setStatus(`Test Accuracy: ${acc}%`);
  // Compute per-class accuracy but do NOT render chart
  const preds=modelCNN.predict(testXs);
  const trueLabels=testYs.argMax(-1);
  const predLabels=preds.argMax(-1);
  const perClassAcc=[];
  for(let i=0;i<10;i++){
    const mask=tf.equal(trueLabels,i);
    const correct=tf.logicalAnd(mask,tf.equal(predLabels,i));
    const acc_i=tf.sum(correct).dataSync()[0]/Math.max(1,tf.sum(mask).dataSync()[0]);
    perClassAcc.push(acc_i);
    mask.dispose(); correct.dispose();
  }
  trueLabels.dispose(); predLabels.dispose(); preds.dispose();
  console.log('Per-class accuracy (hidden chart):', perClassAcc);
}

async function onTestFive(){
  if(!testXs||!testYs){ setStatus('Load test data first'); return; }
  const batch=window.getRandomTestBatch(testXs,testYs,5);
  previewRow.innerHTML='';
  for(let i=0;i<5;i++){
    const div=document.createElement('div'); div.className='preview-item';
    const c=document.createElement('canvas'); div.appendChild(c);
    window.draw28x28ToCanvas(batch.xs.slice([i,0,0,0],[1,28,28,1]),c,4);
    const pred=modelCNN?modelCNN.predict(batch.xs.slice([i,0,0,0],[1,28,28,1])).argMax(-1).dataSync()[0]:null;
    const label=batch.ys.argMax(-1).dataSync()[i];
    const span=document.createElement('span'); span.innerText=`Label:${label} Pred:${pred!==null?pred:'-'}`;
    if(pred===label) span.className='correct'; else span.className='wrong';
    div.appendChild(span); previewRow.appendChild(div);
  }
  batch.xs.dispose(); batch.ys.dispose();
}

async function onSaveModel(){
  if(modelCNN) await modelCNN.save('downloads://mnist-cnn');
  if(modelDenoiser) await modelDenoiser.save('downloads://mnist-denoiser');
}

async function onLoadModel(){
  try{
    const jsonFile=document.getElementById('upload-json').files[0];
    const binFile=document.getElementById('upload-weights').files[0];
    if(!jsonFile||!binFile) throw new Error('Select both JSON and BIN files');
    const jsonUrl=URL.createObjectURL(jsonFile);
    const weightsUrl=URL.createObjectURL(binFile);
    let loaded=null;
    loaded=await tf.loadLayersModel(tf.io.browserFiles([jsonFile,binFile]));
    // Determine if it's CNN or Denoiser by output shape
    if(loaded.outputs[0].shape[1]===10){ modelCNN=loaded; } 
    else { modelDenoiser=loaded; } 
    showModelSummary(loaded); // Always update info of loaded model
    log('Model loaded from files');
  }catch(err){ setStatus('Error loading model: '+err.message); console.error(err);}
}

function onReset(){
  safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
  safeDispose(modelCNN); safeDispose(modelDenoiser);
  trainXs=trainYs=testXs=testYs=null;
  modelCNN=modelDenoiser=null;
  previewRow.innerHTML=''; modelInfo.innerText=''; setStatus('Reset done'); logsDiv.innerText='';
}
