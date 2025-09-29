// app.js
let trainXs, trainYs, testXs, testYs;
let modelCNN = null;
let modelDenoiser = null;
let noisyTestXs = null;

const statusDiv = document.getElementById('data-status');
const logsDiv = document.getElementById('training-logs');
const metricsDiv = document.getElementById('metrics');
const modelInfo = document.getElementById('model-info');
const previewRow = document.getElementById('preview-row');

document.getElementById('load-data').onclick = onLoadData;
document.getElementById('train-cnn').onclick = onTrainCNN;
document.getElementById('train-denoiser').onclick = onTrainDenoiser;
document.getElementById('evaluate').onclick = onEvaluate;
document.getElementById('test-five').onclick = onTestFive;
document.getElementById('save-model').onclick = onSaveModel;
document.getElementById('load-model').onclick = onLoadModel;
document.getElementById('reset').onclick = onReset;
document.getElementById('toggle-visor').onclick = ()=>tfvis.visor().toggle();

async function onLoadData(){
  try{
    logsDiv.innerText = 'Loading data...';
    const trainFile = document.getElementById('train-csv').files[0];
    const testFile = document.getElementById('test-csv').files[0];
    if(!trainFile||!testFile) throw new Error('Please select both CSV files');
    if(trainXs) {trainXs.dispose();trainYs.dispose();testXs.dispose();testYs.dispose();}
    ({xs:trainXs,ys:trainYs} = await loadTrainFromFiles(trainFile));
    ({xs:testXs,ys:testYs} = await loadTestFromFiles(testFile));
    noisyTestXs = addNoise(testXs); // add noise to test set
    statusDiv.innerText = `Train: ${trainXs.shape[0]} samples\nTest: ${testXs.shape[0]} samples\nNoisy test prepared.`;
    logsDiv.innerText = 'Data loaded successfully!';
  }catch(err){
    logsDiv.innerText = 'Error loading CSVs: '+err.message;
  }
}

function buildCNN(){
  const m = tf.sequential();
  m.add(tf.layers.conv2d({filters:32,kernelSize:3,activation:'relu',padding:'same',inputShape:[28,28,1]}));
  m.add(tf.layers.conv2d({filters:64,kernelSize:3,activation:'relu',padding:'same'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.dropout({rate:0.25}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({units:128,activation:'relu'}));
  m.add(tf.layers.dropout({rate:0.5}));
  m.add(tf.layers.dense({units:10,activation:'softmax'}));
  m.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});
  return m;
}

async function onTrainCNN(){
  try{
    if(!trainXs) throw new Error('Load data first');
    if(modelCNN) {modelCNN.dispose();}
    modelCNN = buildCNN();
    const {trainXs:trX,trainYs:trY,valXs,valYs} = splitTrainVal(trainXs,trainYs);
    const fitCallbacks = tfvis.show.fitCallbacks(
      {name:'CNN Training'},
      ['loss','val_loss','acc','val_acc'],
      {callbacks:['onEpochEnd']}
    );
    logsDiv.innerText='Training CNN...';
    await modelCNN.fit(trX,trY,{
      epochs:5,
      batchSize:64,
      validationData:[valXs,valYs],
      shuffle:true,
      callbacks:fitCallbacks
    });
    logsDiv.innerText='CNN training done.';
    modelInfo.innerText = modelCNN.summary(null,null,{printFn:line=>modelInfo.innerText+=line+'\n'});
  }catch(err){
    logsDiv.innerText='Training error: '+err.message;
  }
}

// Build CNN autoencoder for denoising
function buildDenoiser(){
  const input = tf.input({shape:[28,28,1]});
  let x = tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(input);
  x = tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  x = tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(x);
  x = tf.layers.upSampling2d({size:2}).apply(x);
  const output = tf.layers.conv2d({filters:1,kernelSize:3,padding:'same',activation:'sigmoid'}).apply(x);
  const m = tf.model({inputs:input,outputs:output});
  m.compile({optimizer:'adam',loss:'meanSquaredError'});
  return m;
}

async function onTrainDenoiser(){
  try{
    if(!trainXs) throw new Error('Load data first');
    if(modelDenoiser) modelDenoiser.dispose();
    modelDenoiser = buildDenoiser();
    const noisyTrain = addNoise(trainXs);
    logsDiv.innerText='Training Denoiser...';
    await modelDenoiser.fit(noisyTrain,trainXs,{
      epochs:5,
      batchSize:128,
      shuffle:true,
      validationSplit:0.1,
      callbacks:tfvis.show.fitCallbacks(
        {name:'Denoiser Training'},
        ['loss','val_loss'],
        {callbacks:['onEpochEnd']}
      )
    });
    noisyTrain.dispose();
    logsDiv.innerText='Denoiser training done.';
    modelInfo.innerText = modelDenoiser.summary(null,null,{printFn:line=>modelInfo.innerText+=line+'\n'});
  }catch(err){
    logsDiv.innerText='Training error: '+err.message;
  }
}

async function onEvaluate(){
  try{
    if(!modelCNN) throw new Error('Train or load CNN first');
    logsDiv.innerText='Evaluating...';
    const evalOutput = modelCNN.evaluate(testXs,testYs);
    const acc = (await evalOutput[1].data())[0];
    metricsDiv.innerText = `Test Accuracy: ${(acc*100).toFixed(2)}%`;
    logsDiv.innerText='Evaluation done.';
  }catch(err){
    logsDiv.innerText='Evaluation error: '+err.message;
  }
}

async function onTestFive(){
  try{
    if(!modelCNN) throw new Error('Train or load CNN first');
    previewRow.innerHTML='';
    const {xs:batchXs,ys:batchYs} = getRandomTestBatch(testXs,testYs,5);
    const noisyBatch = addNoise(batchXs); // preview with noise
    const denoised = modelDenoiser ? modelDenoiser.predict(noisyBatch) : noisyBatch;

    const preds = modelCNN.predict(denoised).argMax(-1);
    const labels = batchYs.argMax(-1);

    const predsArr = await preds.data();
    const labelsArr = await labels.data();

    for(let i=0;i<5;i++){
      const container = document.createElement('div');
      container.className='preview-item';
      const canvas = document.createElement('canvas');
      draw28x28ToCanvas(noisyBatch.slice([i,0,0,0],[1,28,28,1]).reshape([28,28,1]),canvas);
      container.appendChild(canvas);
      const lbl = document.createElement('div');
      lbl.innerText = `Pred: ${predsArr[i]} (GT:${labelsArr[i]})`;
      lbl.className = (predsArr[i]===labelsArr[i])?'correct':'wrong';
      container.appendChild(lbl);
      previewRow.appendChild(container);
    }

    tf.dispose([batchXs,batchYs,noisyBatch,denoised,preds,labels]);
  }catch(err){
    logsDiv.innerText='Test error: '+err.message;
  }
}

async function onSaveModel(){
  try{
    if(modelDenoiser){
      await modelDenoiser.save('downloads://mnist-denoiser');
      logsDiv.innerText='Denoiser model saved.';
    }else if(modelCNN){
      await modelCNN.save('downloads://mnist-cnn');
      logsDiv.innerText='CNN model saved.';
    }else throw new Error('No model to save');
  }catch(err){
    logsDiv.innerText='Save error: '+err.message;
  }
}

async function onLoadModel(){
  try{
    const jsonFile = document.getElementById('upload-json').files[0];
    const binFile = document.getElementById('upload-weights').files[0];
    if(!jsonFile||!binFile) throw new Error('Select both JSON and BIN files');
    const m = await tf.loadLayersModel(tf.io.browserFiles([jsonFile,binFile]));
    // Heuristic: check last layer units to know if it's CNN classifier or denoiser
    if(m.outputs[0].shape[1]===10){
      if(modelCNN) modelCNN.dispose();
      modelCNN = m;
      logsDiv.innerText='CNN model loaded.';
    }else{
      if(modelDenoiser) modelDenoiser.dispose();
      modelDenoiser = m;
      logsDiv.innerText='Denoiser model loaded.';
    }
    modelInfo.innerText = '';
    m.summary(null,null,{printFn:line=>modelInfo.innerText+=line+'\n'});
  }catch(err){
    logsDiv.innerText='Load error: '+err.message;
  }
}

function onReset(){
  if(modelCNN) {modelCNN.dispose(); modelCNN=null;}
  if(modelDenoiser) {modelDenoiser.dispose(); modelDenoiser=null;}
  if(trainXs){trainXs.dispose();trainYs.dispose();testXs.dispose();testYs.dispose();}
  statusDiv.innerText='';
  logsDiv.innerText='Reset done.';
  metricsDiv.innerText='';
  previewRow.innerHTML='';
  modelInfo.innerText='';
}
