// app.js
'use strict';

let trainXs=null,trainYs=null,testXs=null,testYs=null;
let modelCNN=null, modelDenoiser=null, bestValAcc=0, trainStartTime=null;

const statusDiv=document.getElementById('data-status');
const logsDiv=document.getElementById('training-logs');
const metricsDiv=document.getElementById('metrics');
const modelInfo=document.getElementById('model-info');
const previewRow=document.getElementById('preview-row');
const progressFill=document.getElementById('train-progress');

document.getElementById('load-data').addEventListener('click',onLoadData);
document.getElementById('train-cnn').addEventListener('click',onTrainCNN);
document.getElementById('train-denoiser').addEventListener('click',onTrainDenoiser);
document.getElementById('evaluate').addEventListener('click',onEvaluate);
document.getElementById('test-five').addEventListener('click',onTestFive);
document.getElementById('save-model').addEventListener('click',onSaveModel);
document.getElementById('load-model').addEventListener('click',onLoadModel);
document.getElementById('reset').addEventListener('click',onReset);
document.getElementById('toggle-visor').addEventListener('click',()=>tfvis.visor().toggle());

function safeDispose(t){try{if(t&&typeof t.dispose==='function')t.dispose();}catch(e){console.warn('Dispose error',e);}}
function setStatus(txt){statusDiv.innerText=txt;}
function log(txt){logsDiv.innerText=txt+'\n'+logsDiv.innerText; console.log(txt);}
function showModelSummary(m){modelInfo.innerText=''; m.summary(null,null,line=>{modelInfo.innerText+=line+'\n';});}
function countParams(m){try{return m.countParams();}catch(e){return 'n/a';}}

async function onLoadData(){
  try{
    const trainFile=document.getElementById('train-csv').files[0];
    const testFile=document.getElementById('test-csv').files[0];
    if(!trainFile||!testFile)throw new Error('Please select both train and test CSV files.');
    safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
    setStatus('Loading training CSV...');
    ({xs:trainXs,ys:trainYs}=await window.loadTrainFromFiles(trainFile));
    setStatus('Loading test CSV...');
    ({xs:testXs,ys:testYs}=await window.loadTestFromFiles(testFile));
    setStatus(`Loaded ${trainXs.shape[0]} train samples and ${testXs.shape[0]} test samples.`);
  }catch(e){setStatus('Error: '+e.message); console.error(e);}
}

// Build CNN model
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

// Build Denoiser
function buildDenoiser(){
  const input=tf.input({shape:[28,28,1]});
  const x=tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(input);
  const x2=tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  const x3=tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(x2);
  const encoded=tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x3);
  const x4=tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(encoded);
  const x5=tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x4);
  const decoded=tf.layers.conv2d({filters:1,kernelSize:3,padding:'same',activation:'sigmoid'}).apply(x5);
  const m=tf.model({inputs:input,outputs:decoded});
  m.compile({optimizer:'adam',loss:'meanSquaredError'});
  return m;
}

// Helper: train with batch-level logging
async function trainModel(model,xs,ys,epochs=1,batchSize=64,valXs=null,valYs=null,isDenoiser=false){
  const totalBatches=Math.ceil(xs.shape[0]/batchSize);
  let startTime=Date.now();
  for(let e=0;e<epochs;e++){
    log(`Epoch ${e+1}/${epochs}`);
    for(let b=0;b<totalBatches;b++){
      const start=b*batchSize;
      const end=Math.min((b+1)*batchSize,xs.shape[0]);
      const batchXs=xs.slice([start,0,0,0],[end-start,28,28,1]);
      const batchYs=ys.slice([start,0],[end-start,isDenoiser?28*28:10]);
      const h=await model.fit(batchXs,batchYs,{epochs:1,batchSize:end-start,shuffle:false});
      const loss=h.history.loss[0], acc=h.history.accuracy? h.history.accuracy[0] : NaN;
      const elapsed=(Date.now()-startTime)/1000;
      const eta=(elapsed/(b+1)*(totalBatches-b-1)).toFixed(1);
      log(`Batch ${b+1}/${totalBatches} loss:${loss.toFixed(4)} acc:${acc?acc.toFixed(4):'-'} ETA:${eta}s`);
      progressFill.style.width=Math.floor(((b+1)/totalBatches + e)/epochs*100)+'%';
      safeDispose(batchXs); safeDispose(batchYs);
      await tf.nextFrame();
    }
  }
}

// Event handlers
async function onTrainCNN(){
  if(!trainXs) return alert('Load data first');
  modelCNN=modelCNN||buildCNN();
  showModelSummary(modelCNN);
  const {trainXs:trXs,trainYs:trYs,valXs:vlXs,valYs:vlYs}=window.splitTrainVal(trainXs,trainYs,0.1);
  await trainModel(modelCNN,trXs,trYs,5,64,vlXs,vlYs,false);
  safeDispose(trXs); safeDispose(trYs); safeDispose(vlXs); safeDispose(vlYs);
  setStatus('CNN Training completed');
}

async function onTrainDenoiser(){
  if(!trainXs) return alert('Load data first');
  modelDenoiser=modelDenoiser||buildDenoiser();
  showModelSummary(modelDenoiser);
  const {trainXs:trXs}=window.splitTrainVal(trainXs,trainYs,0.1);
  const noisyXs=window.addNoise(trXs);
  await trainModel(modelDenoiser,noisyXs,trXs,5,64,null,null,true);
  safeDispose(trXs); safeDispose(noisyXs);
  setStatus('Denoiser Training completed');
}

async function onEvaluate(){
  if(!modelCNN||!testXs) return alert('Train CNN and load test data first');
  const evalRes=await modelCNN.evaluate(testXs,testYs);
  const loss=evalRes[0].dataSync()[0], acc=evalRes[1].dataSync()[0];
  metricsDiv.innerText=`Test Loss: ${loss.toFixed(4)}\nTest Acc: ${acc.toFixed(4)}`;
}

async function onTestFive(){
  if(!modelCNN||!testXs) return alert('Train CNN and load test data first');
  previewRow.innerHTML='';
  const {xs:batchXs,ys:batchYs,indices}=window.getRandomTestBatch(testXs,testYs,5);
  const preds=modelCNN.predict(batchXs);
  const predLabels=preds.argMax(-1).dataSync();
  const trueLabels=batchYs.argMax(-1).dataSync();
  for(let i=0;i<5;i++){
    const div=document.createElement('div'); div.className='preview-item';
    const can=document.createElement('canvas'); div.appendChild(can);
    window.draw28x28ToCanvas(batchXs.slice([i,0,0,0],[1,28,28,1]),can);
    const lbl=document.createElement('div');
    lbl.className=(predLabels[i]===trueLabels[i])?'correct':'wrong';
    lbl.innerText=`${predLabels[i]} (true:${trueLabels[i]})`; div.appendChild(lbl);
    previewRow.appendChild(div);
  }
  safeDispose(batchXs); safeDispose(batchYs); safeDispose(preds);
}

async function onSaveModel(){
  if(!modelCNN) return alert('Train CNN first');
  await modelCNN.save('downloads://mnist-cnn');
  if(modelDenoiser) await modelDenoiser.save('downloads://mnist-denoiser');
}

async function onLoadModel(){
  try{
    const jsonFile=document.getElementById('upload-json').files[0];
    const binFile=document.getElementById('upload-weights').files[0];
    if(!jsonFile||!binFile) throw new Error('Select both JSON and BIN');
    modelCNN=modelCNN||await tf.loadLayersModel(tf.io.browserFiles([jsonFile,binFile]));
    showModelSummary(modelCNN);
    setStatus('Model loaded from files');
  }catch(e){alert(e.message);}
}

function onReset(){
  safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
  safeDispose(modelCNN); safeDispose(modelDenoiser);
  trainXs=null; trainYs=null; testXs=null; testYs=null;
  modelCNN=null; modelDenoiser=null;
  logsDiv.innerText=''; metricsDiv.innerText=''; previewRow.innerHTML=''; modelInfo.innerText='';
  progressFill.style.width='0%'; setStatus('Reset completed');
}
