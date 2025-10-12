'use strict';

document.addEventListener('DOMContentLoaded', () => {
  let trainXs=null, trainYs=null, testXs=null, testYs=null;
  let modelCNN=null, modelDenoiser=null;
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

  const safeDispose = t=>{ try{if(t&&t.dispose) t.dispose();}catch(e){console.warn(e);} };
  const setStatus=txt=>statusDiv.innerText=txt;
  const log=txt=>{logsDiv.innerText=txt; console.log(txt);}
  const showModelSummary=m=>{ modelInfo.innerText=''; m.summary(null,null,line=>{modelInfo.innerText+=line+'\n';}); }

  // ------------------------
  async function onLoadData(){
    setStatus('Loading data...');
    try{
      const trainFile=document.getElementById('train-csv').files[0];
      const testFile=document.getElementById('test-csv').files[0];
      if(!trainFile||!testFile) throw new Error('Please select both train and test CSV files');

      safeDispose(trainXs); safeDispose(trainYs); safeDispose(testXs); safeDispose(testYs);

      const trainData = await window.loadTrainFromFiles(trainFile);
      const testData = await window.loadTestFromFiles(testFile);

      trainXs=trainData.xs; trainYs=trainData.ys;
      testXs=testData.xs; testYs=testData.ys;

      setStatus(`Loaded Train: ${trainXs.shape[0]} samples, Test: ${testXs.shape[0]} samples`);
      previewRandom5(testXs,testYs);
    }catch(e){ console.error(e); setStatus(e.message);}
  }

  // ------------------------
  function previewRandom5(xs, ys){
    previewRow.innerHTML='';
    const batch=window.getRandomTestBatch(xs, ys, 5);
    for(let i=0;i<5;i++){
      const div=document.createElement('div'); div.className='preview-item';
      const canvasOrig=document.createElement('canvas'); div.appendChild(canvasOrig);
      window.draw28x28ToCanvas(batch.xs.slice([i,0,0,0],[1,28,28,1]),canvasOrig);

      // noisy + denoised
      if(modelDenoiser){
        const noisy = window.addNoise(batch.xs.slice([i,0,0,0],[1,28,28,1]));
        const den = modelDenoiser.predict(noisy);
        const canvasDeno=document.createElement('canvas'); div.appendChild(canvasDeno);
        window.draw28x28ToCanvas(den,canvasDeno);
        noisy.dispose(); den.dispose();
      }

      const label=document.createElement('div'); label.className='small';
      const pred=modelCNN?tf.argMax(modelCNN.predict(batch.xs.slice([i,0,0,0],[1,28,28,1])),1).dataSync()[0]:'-';
      const real=tf.argMax(batch.ys.slice([i,0],[1,10]),1).dataSync()[0];
      label.innerText=`Label:${real} Pred:${pred}`;
      div.appendChild(label);
      previewRow.appendChild(div);
    }
    batch.xs.dispose(); batch.ys.dispose();
  }

  // ------------------------
  function buildCNNModel(){
    const model=tf.sequential();
    model.add(tf.layers.conv2d({inputShape:[28,28,1], filters:32, kernelSize:3, activation:'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize:2}));
    model.add(tf.layers.conv2d({filters:64, kernelSize:3, activation:'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize:2}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units:128, activation:'relu'}));
    model.add(tf.layers.dense({units:10, activation:'softmax'}));
    model.compile({optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy']});
    return model;
  }

  function buildDenoiserModel(){
    const input=tf.input({shape:[28,28,1]});
    let x=tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(input);
    x=tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
    x=tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(x);
    x=tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
    x=tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
    x=tf.layers.conv2dTranspose({filters:1,kernelSize:3,strides:2,padding:'same',activation:'sigmoid'}).apply(x);
    const model=tf.model({inputs:input,outputs:x});
    model.compile({optimizer:'adam',loss:'meanSquaredError'});
    return model;
  }

  // ------------------------
  async function onTrainCNN(){
    if(!trainXs||!trainYs){ alert('Load data first'); return; }
    modelCNN = buildCNNModel();
    showModelSummary(modelCNN);

    await modelCNN.fit(trainXs,trainYs,{epochs:3,batchSize:64,
      validationSplit:0.1,
      callbacks:[tfvis.show.fitCallbacks({name:'CNN Training',tab:'Training'},['loss','val_loss','acc','val_acc'],{callbacks:['onEpochEnd']})]
    });
    alert('CNN training done!');
  }

  async function onTrainDenoiser(){
    if(!trainXs){ alert('Load data first'); return; }
    modelDenoiser = buildDenoiserModel();
    showModelSummary(modelDenoiser);

    const noisy = window.addNoise(trainXs,0.3);
    await modelDenoiser.fit(noisy,trainXs,{epochs:3,batchSize:64,
      validationSplit:0.1,
      callbacks:[tfvis.show.fitCallbacks({name:'Denoiser Training',tab:'Training'},['loss','val_loss'],{callbacks:['onEpochEnd']})]
    });
    noisy.dispose();
    alert('Denoiser training done!');
  }

  async function onEvaluate(){
    if(!modelCNN||!testXs||!testYs){ alert('Need CNN model and test data'); return; }
    const preds=modelCNN.predict(testXs);
    const predLabels=preds.argMax(1).dataSync();
    const trueLabels=testYs.argMax(1).dataSync();
    let correct=0;
    const perClassAcc=Array(10).fill(0);
    const perClassCount=Array(10).fill(0);
    for(let i=0;i<trueLabels.length;i++){
      const t=trueLabels[i]; const p=predLabels[i];
      if(t===p) correct++;
      perClassCount[t]++; if(t===p) perClassAcc[t]++;
    }
    const overallAcc=(correct/trueLabels.length)*100;
    metricsDiv.innerText=`Overall Test Accuracy: ${overallAcc.toFixed(2)}%`;

    // show per-class accuracy chart
    const chartData = perClassAcc.map((v,i)=>({index:i,value:(v/perClassCount[i])*100}));
    tfvis.render.barchart({name:'Per-class Accuracy',tab:'Evaluation'},{values:chartData.map(d=>d.value),labels:chartData.map(d=>d.index.toString())});
    preds.dispose();
  }

  function onTestFive(){ if(testXs&&testYs) previewRandom5(testXs,testYs); else alert('No test data'); }

  function onSaveModel(){ 
    if(modelCNN) modelCNN.save('downloads://mnist-cnn-model');
    if(modelDenoiser) modelDenoiser.save('downloads://mnist-denoiser-model');
  }

  async function onLoadModel(){
    const jsonFile=document.getElementById('upload-json').files[0];
    const weightsFile=document.getElementById('upload-weights').files[0];
    if(!jsonFile||!weightsFile){ alert('Please select both JSON and BIN files'); return; }
    try{
      const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
      if(model.name.includes('denoiser')) modelDenoiser=model;
      else modelCNN=model;
      alert(`Loaded model: ${model.name}`);
      // do NOT clear modelInfo.innerText
      showModelSummary(model);
    }catch(e){console.error(e); alert('Failed to load model'); }
  }

  function onReset(){ location.reload(); }

});
