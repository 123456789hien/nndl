'use strict';

let trainXs=null, trainYs=null, testXs=null, testYs=null;
let modelCNN=null, modelDenoiser=null;

const dataStatus=document.getElementById('data-status');
const metricsDiv=document.getElementById('metrics');
const modelInfo=document.getElementById('model-info');
const previewRow=document.getElementById('preview-row');

// Load data
document.getElementById('load-data').onclick=async()=>{
  try{
    const trainFile=document.getElementById('train-csv').files[0];
    const testFile=document.getElementById('test-csv').files[0];
    if(!trainFile||!testFile) return alert('Select train/test CSVs');
    dataStatus.innerText='Loading...';
    const trainData=await window.loadTrainFromFiles(trainFile);
    const testData=await window.loadTestFromFiles(testFile);
    trainXs=trainData.xs; trainYs=trainData.ys;
    testXs=testData.xs; testYs=testData.ys;
    dataStatus.innerText=`Loaded Train: ${trainXs.shape[0]} samples, Test: ${testXs.shape[0]} samples`;
  }catch(e){console.error(e); dataStatus.innerText='Error: '+e.message;}
};

// Create CNN
function createCNN(){
  const model=tf.sequential();
  model.add(tf.layers.conv2d({inputShape:[28,28,1],filters:16,kernelSize:3,activation:'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize:2}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units:64,activation:'relu'}));
  model.add(tf.layers.dense({units:10,activation:'softmax'}));
  model.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});
  return model;
}

// Train CNN
document.getElementById('train-cnn').onclick=async()=>{
  if(!trainXs||!trainYs) return alert('Load data first');
  modelCNN=createCNN();
  modelInfo.innerText='Training CNN...';
  await modelCNN.fit(trainXs,trainYs,{epochs:3,validationSplit:0.1,callbacks:{onEpochEnd:(epoch,logs)=>{modelInfo.innerText=`Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)} val_loss=${logs.val_loss.toFixed(4)}`;}}});
  modelInfo.innerText+=' \nTraining done';
};

// Train Denoiser
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

document.getElementById('train-denoiser').onclick=async()=>{
  if(!trainXs) return alert('Load data first');
  modelDenoiser=createDenoiser();
  modelInfo.innerText='Training Denoiser...';
  const noisy=window.addNoise(trainXs);
  await modelDenoiser.fit(noisy,trainXs,{epochs:3,batchSize:32});
  modelInfo.innerText+=' \nDenoiser trained';
  noisy.dispose();
};

// Load model from JSON + BIN
document.getElementById('load-model').onclick=async()=>{
  const jsonFile=document.getElementById('upload-json').files[0];
  const binFile=document.getElementById('upload-weights').files[0];
  if(!jsonFile||!binFile) return alert('Select JSON + BIN');
  try{
    const handler=tf.io.browserFiles([jsonFile,binFile]);
    const loadedModel=await tf.loadLayersModel(handler);
    if(jsonFile.name.includes('denoiser')) modelDenoiser=loadedModel;
    else modelCNN=loadedModel;
    modelInfo.innerText+=`\nLoaded model: ${loadedModel.name || jsonFile.name}`;
  }catch(e){console.error(e); alert('Load model error: '+e.message);}
};

// Evaluate
document.getElementById('evaluate').onclick=async()=>{
  if(!modelCNN||!testXs||!testYs) return alert('Need CNN model & test data');
  const preds=modelCNN.predict(testXs);
  const predLabels=preds.argMax(1).dataSync();
  const trueLabels=testYs.argMax(1).dataSync();

  let correct=0;
  const perClassAcc=Array(10).fill(0);
  const perClassCount=Array(10).fill(0);
  for(let i=0;i<trueLabels.length;i++){
    const t=trueLabels[i], p=predLabels[i];
    if(t===p) correct++;
    perClassCount[t]++;
    if(t===p) perClassAcc[t]++;
  }
  metricsDiv.innerText=`Overall Accuracy: ${(correct/trueLabels.length*100).toFixed(2)}%`;

  // Per-class accuracy
  const chartData=perClassAcc.map((v,i)=>({index:i,value: perClassCount[i]>0?(v/perClassCount[i])*100:0}));
  tfvis.render.barchart({name:'Per-class Accuracy',tab:'Evaluation'},
                        {values: chartData.map(d=>d.value), labels: chartData.map(d=>d.index.toString())});

  // Confusion Matrix
  const confMatrix=tf.math.confusionMatrix(tf.tensor1d(trueLabels,'int32'), tf.tensor1d(predLabels,'int32'), 10);
  tfvis.render.confusionMatrix({name:'Confusion Matrix',tab:'Evaluation'}, {values: await confMatrix.array()});
  confMatrix.dispose();
  preds.dispose();
};

// Test 5 random
document.getElementById('test-five').onclick=()=>{
  if(!testXs||!testYs) return;
  const batch=window.getRandomTestBatch(testXs,testYs,5);
  previewRow.innerHTML='';
  for(let i=0;i<batch.xs.shape[0];i++){
    const div=document.createElement('div'); div.className='preview-item';
    const canvas=document.createElement('canvas');
    window.draw28x28ToCanvas(batch.xs.slice([i,0,0,0],[1,28,28,1]),canvas,4);
    const label=batch.ys.argMax(1).dataSync()[i];
    div.appendChild(canvas);
    const span=document.createElement('span'); span.innerText=`Label: ${label}`;
    div.appendChild(span);
    previewRow.appendChild(div);
  }
  batch.xs.dispose(); batch.ys.dispose();
};

// Save model
document.getElementById('save-model').onclick=()=>{ 
  if(modelCNN) modelCNN.save('downloads://cnn-model');
  if(modelDenoiser) modelDenoiser.save('downloads://denoiser-model');
};

// Reset
document.getElementById('reset').onclick=()=>{
  trainXs?.dispose(); trainYs?.dispose(); testXs?.dispose(); testYs?.dispose();
  trainXs=trainYs=testXs=testYs=null; modelCNN=modelDenoiser=null;
  metricsDiv.innerText=''; modelInfo.innerText=''; previewRow.innerHTML=''; dataStatus.innerText='';
};

// Toggle visor
document.getElementById('toggle-visor').onclick=()=>tfvis.visor().toggle();
