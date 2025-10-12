// app.js
'use strict';
/* global tf, tfvis */

let trainXs=null, trainYs=null, testXs=null, testYs=null;
let modelCNN=null, modelDenoiser=null;

const statusDiv=document.getElementById('data-status');
const logsDiv=document.getElementById('training-logs');
const metricsDiv=document.getElementById('metrics');
const modelInfo=document.getElementById('model-info');
const previewRow=document.getElementById('preview-row');
const progressBar=document.getElementById('progress-bar');

document.getElementById('load-data').addEventListener('click', onLoadData);
document.getElementById('train-cnn').addEventListener('click', onTrainCNN);
document.getElementById('train-denoiser').addEventListener('click', onTrainDenoiser);
document.getElementById('evaluate').addEventListener('click', onEvaluate);
document.getElementById('test-five').addEventListener('click', onTestFive);
document.getElementById('toggle-visor').addEventListener('click', ()=>{ if(typeof tfvis!=='undefined') tfvis.visor().toggle(); });

async function onLoadData(){
  const trainFile=document.getElementById('train-csv').files[0];
  const testFile=document.getElementById('test-csv').files[0];
  if(!trainFile||!testFile){ alert('Chọn cả train & test CSV'); return; }
  statusDiv.textContent='Loading training CSV...';
  const trainData=await window.loadTrainFromFiles(trainFile);
  statusDiv.textContent='Loading test CSV...';
  const testData=await window.loadTestFromFiles(testFile);
  if(trainXs) trainXs.dispose(); if(trainYs) trainYs.dispose();
  if(testXs) testXs.dispose(); if(testYs) testYs.dispose();
  trainXs=trainData.xs; trainYs=trainData.ys;
  testXs=testData.xs; testYs=testData.ys;
  statusDiv.textContent=`Loaded ${trainXs.shape[0]} train & ${testXs.shape[0]} test samples`;
}

function createCNN(){
  const model=tf.sequential();
  model.add(tf.layers.conv2d({inputShape:[28,28,1],filters:32,kernelSize:3,activation:'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize:2}));
  model.add(tf.layers.conv2d({filters:64,kernelSize:3,activation:'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize:2}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units:128,activation:'relu'}));
  model.add(tf.layers.dense({units:10,activation:'softmax'}));
  model.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});
  return model;
}

function createDenoiser(){
  const input=tf.input({shape:[28,28,1]});
  let x=tf.layers.conv2d({filters:32,kernelSize:3,activation:'relu',padding:'same'}).apply(input);
  x=tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  x=tf.layers.conv2d({filters:64,kernelSize:3,activation:'relu',padding:'same'}).apply(x);
  x=tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  x=tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,activation:'relu',padding:'same'}).apply(x);
  x=tf.layers.conv2dTranspose({filters:1,kernelSize:3,strides:2,activation:'sigmoid',padding:'same'}).apply(x);
  const model=tf.model({inputs:input,outputs:x});
  model.compile({optimizer:'adam',loss:'meanSquaredError'});
  return model;
}

// ==================== TRAINING WITH BATCH PROGRESS ===================
async function onTrainCNN(){
  if(!trainXs){ alert('Chưa load dữ liệu'); return; }
  if(modelCNN) modelCNN=null;
  modelCNN=createCNN();
  logsDiv.textContent='Training CNN...\n';
  progressBar.style.width='0%';

  const batchSize=64, epochs=10;
  const valSplit=0.1;
  const trainSize=Math.floor(trainXs.shape[0]*(1-valSplit));
  const valSize=trainXs.shape[0]-trainSize;

  const container={name:'CNN Training', tab:'Training'};
  const metrics=['loss','val_loss','acc','val_acc'];
  const tfvisCallbacks=tfvis.show.fitCallbacks(container, metrics);

  let startTime=Date.now();
  await modelCNN.fit(trainXs,trainYs,{
    epochs,
    batchSize,
    validationSplit:valSplit,
    callbacks:{
      onEpochBegin: async (epoch)=>{logsDiv.textContent+=`Epoch ${epoch+1} start...\n`;},
      onBatchEnd: async (batch, logs)=>{
        const batchProgress=((batch+1)/Math.ceil(trainSize/batchSize))*100;
        progressBar.style.width=`${batchProgress.toFixed(2)}%`;
        const elapsed=(Date.now()-startTime)/1000;
        const batchesLeft=(epochs-1)*Math.ceil(trainSize/batchSize)+(Math.ceil(trainSize/batchSize)-batch-1);
        const est=(elapsed/(batch+1)*batchesLeft);
        logsDiv.textContent+=`Batch ${batch+1}: loss=${logs.loss.toFixed(4)}, acc=${(logs.acc*100).toFixed(2)}%, ETA=${est.toFixed(1)}s\n`;
        logsDiv.scrollTop=logsDiv.scrollHeight;
      },
      onEpochEnd: async(epoch, logs)=>{
        logsDiv.textContent+=`Epoch ${epoch+1} end: loss=${logs.loss.toFixed(4)}, acc=${(logs.acc*100).toFixed(2)}%, val_loss=${logs.val_loss.toFixed(4)}, val_acc=${(logs.val_acc*100).toFixed(2)}%\n`;
      },
      ...tfvisCallbacks
    }
  });
  modelInfo.textContent='';
  modelCNN.summary(null, undefined, line=>modelInfo.textContent+=line+'\n');
}

async function onTrainDenoiser(){
  if(!trainXs){ alert('Chưa load dữ liệu'); return; }
  if(modelDenoiser) modelDenoiser=null;
  modelDenoiser=createDenoiser();
  logsDiv.textContent='Training Denoiser...\n';
  progressBar.style.width='0%';

  const {trainXs:trX}=window.splitTrainVal(trainXs,trainYs,0.1);
  const trXnoisy=window.addNoise(trX,0.25);
  const batchSize=64, epochs=10;
  const trainSize=trX.shape[0];

  const container={name:'Denoiser Training', tab:'Training'};
  const metrics=['loss','val_loss'];
  const tfvisCallbacks=tfvis.show.fitCallbacks(container, metrics);
  let startTime=Date.now();

  await modelDenoiser.fit(trXnoisy,trX,{
    epochs,
    batchSize,
    validationSplit:0.1,
    callbacks:{
      onBatchEnd: async(batch, logs)=>{
        const batchProgress=((batch+1)/Math.ceil(trainSize/batchSize))*100;
        progressBar.style.width=`${batchProgress.toFixed(2)}%`;
        const elapsed=(Date.now()-startTime)/1000;
        const batchesLeft=(epochs-1)*Math.ceil(trainSize/batchSize)+(Math.ceil(trainSize/batchSize)-batch-1);
        logsDiv.textContent+=`Batch ${batch+1}: loss=${logs.loss.toFixed(4)}, ETA=${(elapsed/(batch+1)*batchesLeft).toFixed(1)}s\n`;
        logsDiv.scrollTop=logsDiv.scrollHeight;
      },
      onEpochEnd: async(epoch, logs)=>{
        logsDiv.textContent+=`Epoch ${epoch+1} end: loss=${logs.loss.toFixed(4)}, val_loss=${logs.val_loss.toFixed(4)}\n`;
      },
      ...tfvisCallbacks
    }
  });

  trX.dispose(); trXnoisy.dispose();
  modelInfo.textContent='';
  modelDenoiser.summary(null, undefined, line=>modelInfo.textContent+=line+'\n');
}

async function onEvaluate(){
  if(!modelCNN||!testXs) return;
  const evalRes=await modelCNN.evaluate(testXs,testYs);
  metricsDiv.textContent=`Loss: ${evalRes[0].dataSync()[0].toFixed(4)}  Accuracy: ${(evalRes[1].dataSync()[0]*100).toFixed(2)}%`;
}

async function onTestFive(){
  if(!testXs||!modelCNN) return;
  const batch=window.getRandomTestBatch(testXs,testYs,5);
  previewRow.innerHTML='';
  const preds=modelCNN.predict(batch.xs).argMax(-1).dataSync();
  const labels=batch.ys.argMax(-1).dataSync();
  for(let i=0;i<5;i++){
    const div=document.createElement('div'); div.className='preview-item';
    const canvas=document.createElement('canvas');
    window.draw28x28ToCanvas(batch.xs.slice([i,0,0,0],[1,28,28,1]),canvas);
    const span=document.createElement('span');
    span.textContent=`P:${preds[i]} / L:${labels[i]}`;
    span.className=preds[i]===labels[i]?'correct':'wrong';
    div.appendChild(canvas); div.appendChild(span);
    previewRow.appendChild(div);
  }
  batch.xs.dispose(); batch.ys.dispose();
}
