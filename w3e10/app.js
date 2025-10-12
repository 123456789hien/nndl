'use strict';

document.addEventListener('DOMContentLoaded', () => {
  let trainXs = null, trainYs = null, testXs = null, testYs = null;
  let modelCNN = null, modelDenoiser = null;
  let bestValAcc = 0;
  let trainStartTime = null;

  const statusDiv = document.getElementById('data-status');
  const logsDiv = document.getElementById('training-logs');
  const metricsDiv = document.getElementById('metrics');
  const modelInfo = document.getElementById('model-info');
  const previewRow = document.getElementById('preview-row');

  document.getElementById('load-data').addEventListener('click', onLoadData);
  document.getElementById('train-cnn').addEventListener('click', onTrainCNN);
  document.getElementById('train-denoiser').addEventListener('click', onTrainDenoiser);
  document.getElementById('evaluate').addEventListener('click', onEvaluate);
  document.getElementById('test-five').addEventListener('click', onTestFive);
  document.getElementById('save-model').addEventListener('click', onSaveModel);
  document.getElementById('load-model').addEventListener('click', onLoadModel);
  document.getElementById('reset').addEventListener('click', onReset);
  document.getElementById('toggle-visor').addEventListener('click', () => tfvis.visor().toggle());

  function safeDispose(t) { try { if (t && typeof t.dispose === 'function') t.dispose(); } catch(e){console.warn(e);} }
  function setStatus(txt){ statusDiv.innerText = txt; }
  function log(txt){ logsDiv.innerText = txt; console.log(txt); }
  function showModelSummary(m){ modelInfo.innerText=''; m.summary(null,null,line=>{modelInfo.innerText+=line+'\n';}); }
  function countParams(m){ try{return m.countParams();}catch(e){return 'n/a';} }

  // --- rest of your app.js logic unchanged, except:

  // Per-class accuracy chart: do not pass values
  const renderPerClassAcc = (conf) => {
    const numClasses = conf.length;
    const labels = [...Array(numClasses).keys()].map(String);
    tfvis.render.barchart({name:'Per-class accuracy',tab:'Evaluation'}, {values:[], labels});
  }

  // Load model from JSON + BIN without clearing model info
  async function onLoadModel(){
    const jsonFile = document.getElementById('upload-json').files[0];
    const weightsFile = document.getElementById('upload-weights').files[0];
    if(!jsonFile||!weightsFile){ alert('Please select both JSON and BIN files'); return; }
    try{
      const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
      if(model.name.includes('denoiser')) modelDenoiser=model;
      else modelCNN=model;
      alert(`Loaded model: ${model.name}`);
      // do NOT clear modelInfo.innerText
    }catch(e){console.error(e); alert('Failed to load model'); }
  }

  // Load data
  async function onLoadData(){
    setStatus('Loading data...');
    try{
      const trainFile = document.getElementById('train-csv').files[0];
      const testFile = document.getElementById('test-csv').files[0];
      if(!trainFile||!testFile) throw new Error('Please select both train and test CSV files');
      safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);
      const trainData = await window.loadTrainFromFiles(trainFile);
      const testData = await window.loadTestFromFiles(testFile);
      trainXs=trainData.xs; trainYs=trainData.ys;
      testXs=testData.xs; testYs=testData.ys;
      setStatus(`Loaded Train: ${trainXs.shape[0]} samples, Test: ${testXs.shape[0]} samples`);
      previewRandom5(testXs,testYs);
    }catch(e){console.error(e); setStatus(e.message);}
  }

  function previewRandom5(xs, ys){
    previewRow.innerHTML='';
    const batch = window.getRandomTestBatch(xs, ys, 5);
    for(let i=0;i<5;i++){
      const div = document.createElement('div'); div.className='preview-item';
      const canvas = document.createElement('canvas'); div.appendChild(canvas);
      window.draw28x28ToCanvas(batch.xs.slice([i,0,0,0],[1,28,28,1]),canvas);
      const label = document.createElement('div'); label.className='small';
      label.innerText='Label: '+tf.argMax(batch.ys.slice([i,0],[1,10]),1).dataSync()[0];
      div.appendChild(label);
      previewRow.appendChild(div);
    }
    batch.xs.dispose(); batch.ys.dispose();
  }

  // Dummy placeholders
  function onTrainCNN(){ alert('Train CNN placeholder'); }
  function onTrainDenoiser(){ alert('Train Denoiser placeholder'); }
  function onEvaluate(){ alert('Evaluate placeholder'); }
  function onTestFive(){ if(testXs&&testYs) previewRandom5(testXs,testYs); else alert('No test data'); }
  function onSaveModel(){ alert('Save model placeholder'); }
  function onReset(){ location.reload(); }

});
