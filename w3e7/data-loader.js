// data-loader.js
'use strict';

async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = e => reject(new Error('Failed to read file: ' + e.target.error));
    reader.readAsText(file);
  });
}

async function loadCSVFile(file) {
  const text = await readFileAsText(file);
  const lines = text.split(/\r?\n/);
  const images = [], labels = [];
  for (let i=0;i<lines.length;i++){
    const raw = lines[i].trim();
    if(!raw) continue;
    const parts = raw.split(',').map(s=>s.trim());
    if(parts.length<785){console.warn(`Skipping line ${i}`); continue;}
    const lab = parseInt(parts[0],10); if(Number.isNaN(lab)) continue;
    const pix = parts.slice(1,785).map(x=>Number(x)/255);
    if(pix.some(v=>Number.isNaN(v))) continue;
    images.push(pix); labels.push(lab);
  }
  if(images.length===0) throw new Error('No valid rows in '+file.name);
  const xs2d = tf.tensor2d(images,[images.length,784],'float32');
  const xs = xs2d.reshape([images.length,28,28,1]);
  const labs = tf.tensor1d(labels,'int32');
  const ys = tf.oneHot(labs,10).toFloat();
  xs2d.dispose(); labs.dispose();
  return {xs,ys};
}

async function loadTrainFromFiles(file){return loadCSVFile(file);}
async function loadTestFromFiles(file){return loadCSVFile(file);}

function splitTrainVal(xs, ys, valRatio=0.1){
  const total = xs.shape[0];
  const valSize = Math.max(1,Math.floor(total*valRatio));
  const trainSize = total-valSize;
  return {
    trainXs: xs.slice([0,0,0,0],[trainSize,28,28,1]),
    trainYs: ys.slice([0,0],[trainSize,10]),
    valXs: xs.slice([trainSize,0,0,0],[valSize,28,28,1]),
    valYs: ys.slice([trainSize,0],[valSize,10])
  };
}

function addNoise(xs, std=0.25){return tf.tidy(()=>xs.add(tf.randomNormal(xs.shape,0,std,'float32')).clipByValue(0,1)).clone();}

function getRandomTestBatch(xs,ys,k=5){
  const total=xs.shape[0];
  if(k>total) k=total;
  const idx=tf.util.createShuffledIndices(total).slice(0,k);
  const imgs=[], labs=[];
  for(let i=0;i<idx.length;i++){
    imgs.push(xs.slice([idx[i],0,0,0],[1,28,28,1]));
    labs.push(ys.slice([idx[i],0],[1,10]));
  }
  const xsBatch=tf.concat(imgs,0);
  const ysBatch=tf.concat(labs,0);
  imgs.forEach(t=>t.dispose()); labs.forEach(t=>t.dispose());
  return {xs: xsBatch, ys: ysBatch, indices: idx};
}

function draw28x28ToCanvas(tensor, canvas, scale=4){
  let t = tensor;
  if(tensor.rank===4||tensor.rank===3)t=tensor.reshape([28,28]);
  const data=t.dataSync(); const w=28,h=28; canvas.width=w*scale; canvas.height=h*scale;
  const ctx=canvas.getContext('2d'); const img=ctx.createImageData(w,h);
  for(let i=0;i<data.length;i++){const v=Math.max(0,Math.min(255,Math.round(data[i]*255))); img.data[i*4+0]=v; img.data[i*4+1]=v; img.data[i*4+2]=v; img.data[i*4+3]=255;}
  const tmp=document.createElement('canvas'); tmp.width=w; tmp.height=h;
  tmp.getContext('2d').putImageData(img,0,0);
  ctx.imageSmoothingEnabled=false;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(tmp,0,0,canvas.width,canvas.height);
  if(t!==tensor)t.dispose();
}

window.loadTrainFromFiles=loadTrainFromFiles;
window.loadTestFromFiles=loadTestFromFiles;
window.splitTrainVal=splitTrainVal;
window.addNoise=addNoise;
window.getRandomTestBatch=getRandomTestBatch;
window.draw28x28ToCanvas=draw28x28ToCanvas;
