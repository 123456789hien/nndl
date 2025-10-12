// app.js
'use strict';

let trainXs=null, trainYs=null, testXs=null, testYs=null;
let modelCNN=null, modelDenoiser=null, bestValAcc=0, trainStartTime=null;

const statusDiv=document.getElementById('data-status');
const logsDiv=document.getElementById('training-logs');
const metricsDiv=document.getElementById('metrics');
const modelInfo=document.getElementById('model-info');
const previewRow=document.getElementById('preview-row');

document.getElementById('load-data')?.addEventListener('click', onLoadData);
document.getElementById('train-cnn')?.addEventListener('click', onTrainCNN);
document.getElementById('train-denoiser')?.addEventListener('click', onTrainDenoiser);
document.getElementById('evaluate')?.addEventListener('click', onEvaluate);
document.getElementById('test-five')?.addEventListener('click', onTestFive);
document.getElementById('save-model')?.addEventListener('click', onSaveModel);
document.getElementById('load-model')?.addEventListener('click', onLoadModel);
document.getElementById('reset')?.addEventListener('click', onReset);
document.getElementById('toggle-visor')?.addEventListener('click', ()=>tfvis.visor().toggle());

function safeDispose(t){try{if(t&&typeof t.dispose==='function') t.dispose();}catch(e){console.warn('Dispose error',e);}}
function setStatus(txt){statusDiv.innerText=txt;}
function log(txt){logsDiv.innerText=txt; console.log(txt);}
function showModelSummary(m){modelInfo.innerText=''; m.summary(null,null,line=>{modelInfo.innerText+=line+'\n';});}
function countParams(m){try{return m.countParams();}catch(e){return'n/a';}}

// ----- Data loading -----
async function onLoadData(){ /* unchanged as your original code */ ... }

// ----- Model builders -----
function buildCNN(){ /* unchanged */ ... }
function buildDenoiser(){ /* unchanged */ ... }

// ----- Training CNN -----
async function onTrainCNN(){ /* unchanged */ ... }

// ----- Training Denoiser -----
async function onTrainDenoiser(){ /* unchanged */ ... }

// ----- Evaluate -----
async function onEvaluate(){
  try{
    if(!modelCNN) throw new Error('Train or load CNN first.');
    if(!testXs||!testYs) throw new Error('Load data first.');
    setStatus('Evaluating on test set...');
    log('Evaluating test set...');

    const evalOutput=await modelCNN.evaluate(testXs,testYs,{batchSize:128});
    let lossTensor=null, accTensor=null;
    if(Array.isArray(evalOutput)){lossTensor=evalOutput[0]; accTensor=evalOutput[1];}else{lossTensor=evalOutput; accTensor=null;}
    const loss=lossTensor?(await lossTensor.data())[0]:NaN;
    const acc=accTensor?(await accTensor.data())[0]:NaN;
    metricsDiv.innerText=`Test Accuracy: ${(acc*100).toFixed(2)}% | Loss: ${loss.toFixed(4)}`;

    // Confusion matrix
    const predsArr=[], labelsArr=[];
    const BATCH=256, total=testXs.shape[0];
    for(let i=0;i<total;i+=BATCH){
      const end=Math.min(i+BATCH,total);
      const batchX=testXs.slice([i,0,0,0],[end-i,28,28,1]);
      const logits=modelCNN.predict(batchX);
      const pred=logits.argMax(-1);
      const label=testYs.slice([i,0],[end-i,10]).argMax(-1);
      predsArr.push(...Array.from(await pred.data()));
      labelsArr.push(...Array.from(await label.data()));
      batchX.dispose(); logits.dispose(); pred.dispose(); label.dispose();
      await tf.nextFrame();
    }

    const numClasses=10;
    const conf=Array.from({length:numClasses},()=>Array(numClasses).fill(0));
    for(let i=0;i<labelsArr.length;++i) conf[labelsArr[i]][predsArr[i]]+=1;

    tfvis.render.confusionMatrix({name:'Confusion Matrix',tab:'Evaluation'},{values:conf,tickLabels:[...Array(numClasses).keys()].map(String)});

    // --- PER-CLASS ACCURACY: only bars, no numbers
    const perClassAcc=conf.map((row,i)=>{
      const totalRow=row.reduce((a,b)=>a+b,0)||1;
      return {label:String(i), value: row[i]/totalRow};
    });
    tfvis.render.barchart({name:'Per-class accuracy',tab:'Evaluation'},{values:perClassAcc.map(x=>x.value),labels:perClassAcc.map(x=>x.label),options:{barNumberFormat:''}});

    setStatus(`Evaluation done. Accuracy ${(acc*100).toFixed(2)}%`);
    log('Evaluation complete.');
  }catch(err){
    console.error(err);
    setStatus('Evaluate error: '+(err.message||err));
    log('Evaluate error: '+(err.message||err));
  }
}

// ----- Test 5 Random -----
async function onTestFive(){ /* unchanged */ ... }

// ----- Save / Load models -----
async function onSaveModel(){ /* unchanged */ ... }

async function onLoadModel(){
  try{
    const jsonFile=document.getElementById('upload-json').files[0];
    const binFile=document.getElementById('upload-weights').files[0];
    if(!jsonFile||!binFile) throw new Error('Select both JSON and BIN weight files.');
    setStatus('Loading model from files...');
    const m=await tf.loadLayersModel(tf.io.browserFiles([jsonFile,binFile]));

    const outShape=m.outputs[0].shape;
    if(outShape && outShape.length>=2 && outShape[outShape.length-1]===10){
      if(modelCNN) modelCNN.dispose();
      modelCNN=m;
      showModelSummary(modelCNN);
      setStatus('CNN loaded from files.');
      log('CNN loaded.');
    }else{
      if(modelDenoiser) modelDenoiser.dispose();
      modelDenoiser=m;
      // IMPORTANT: keep existing Model Info panel untouched
      setStatus('Denoiser loaded from files.');
      log('Denoiser loaded.');
    }
  }catch(err){
    console.error(err);
    setStatus('Load error: '+(err.message||err));
    log('Load error: '+(err.message||err));
  }
}

// ----- Reset -----
function onReset(){ /* unchanged */ ... }

window.addEventListener('beforeunload',()=>{
  safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
  safeDispose(modelCNN); safeDispose(modelDenoiser);
});
