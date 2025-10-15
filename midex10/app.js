// app.js
let monthly=[]; let groups=[]; let scaler={min:[], max:[]};
let chartCancel=null, chartPrice=null, chartHeat=null, chartPredict=null;
let model=null;

// UI refs
const btnLoad=document.getElementById('btn-load');
const noteLoad=document.getElementById('note-load');
const mergeDiv=document.getElementById('merge-overview');
const missDiv=document.getElementById('missing-overview');

// file upload
document.getElementById('train-file').addEventListener('change',()=>{noteLoad.textContent='Train file selected';});
document.getElementById('test-file').addEventListener('change',()=>{noteLoad.textContent='Test file selected';});
document.getElementById('model-json').addEventListener('change',()=>{noteLoad.textContent='Model JSON selected';});
document.getElementById('model-bin').addEventListener('change',()=>{noteLoad.textContent='Model BIN selected';});

// load model
async function loadModelFromFiles(jsonFile, binFile){
  if(!jsonFile||!binFile) return null;
  const buffer = await binFile.arrayBuffer();
  const jsonText = await jsonFile.text();
  model = new mljs.NeuralNetwork(JSON.parse(jsonText),new Uint8Array(buffer));
  return model;
}

// compute scaler
function computeScaler(series){
  const fields=['cancellation_rate','avg_room_price','lead_time_avg'];
  scaler.min=[]; scaler.max=[];
  fields.forEach(f=>{
    const vals=series.map(d=>d[f]);
    scaler.min.push(Math.min(...vals)); scaler.max.push(Math.max(...vals));
  });
}

// normalize input
function normalizeInput(input){
  return input.map((v,i)=> (v - scaler.min[i])/(scaler.max[i]-scaler.min[i]||1));
}

// denormalize
function denormalize(value,idx){ return value*(scaler.max[idx]-scaler.min[idx])+scaler.min[idx];}

// show merge overview
function renderMergeOverview(){
  if(!monthly||monthly.length===0) return;
  const head10=monthly.slice(0,10);
  mergeDiv.innerHTML='<table border="1"><tr>'+Object.keys(head10[0]).map(k=>`<th>${k}</th>`).join('')+'</tr>'+head10.map(r=>'<tr>'+Object.values(r).map(v=>`<td>${v}</td>`).join('')+'</tr>').join('')+'</table>';
}

// render cancel chart
function renderCancelChart(group){
  const series=monthly.filter(d=>d.group===group).sort((a,b)=>a.month_index-b.month_index);
  const labels=series.map(d=>`${d.year}-${d.month}`);
  const data=series.map(d=>d.cancellation_rate);
  if(chartCancel) chartCancel.destroy();
  chartCancel=new Chart(document.getElementById('chart-cancel'),{
    type:'line',
    data:{labels, datasets:[{label:'Cancellation Rate',data,color:'red'}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });
}

// render price chart
function renderPriceChart(group){
  const series=monthly.filter(d=>d.group===group).sort((a,b)=>a.month_index-b.month_index);
  const labels=series.map(d=>`${d.year}-${d.month}`);
  const data=series.map(d=>d.avg_room_price);
  if(chartPrice) chartPrice.destroy();
  chartPrice=new Chart(document.getElementById('chart-price'),{
    type:'line',
    data:{labels,datasets:[{label:'Avg Room Price',data}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });
}

// predict next month
async function predictNext(group,year,month){
  if(!model) {alert('Please load model first'); return;}
  const series=monthly.filter(d=>d.group===group).sort((a,b)=>a.month_index-b.month_index);
  if(series.length<12){alert('Not enough history'); return;}
  const targetIndex=series.findIndex(s=>s.year===year&&s.month===month);
  let seqWindow;
  if(targetIndex>=12){ seqWindow=series.slice(targetIndex-12,targetIndex); } else { seqWindow=series.slice(-12); }
  const inputRaw=seqWindow.map(d=>[d.cancellation_rate,d.avg_room_price,d.lead_time_avg]).flat();
  computeScaler(series);
  const normInput=normalizeInput(inputRaw);
  const predNorm=model.predict(normInput);
  const pred=denormalize(predNorm[0],0);

  if(chartPredict) chartPredict.destroy();
  chartPredict=new Chart(document.getElementById('chart-predict'),{
    type:'line',
    data:{labels:[`${year}-${month}`], datasets:[{label:'Predicted Cancellation Rate',data:[pred]}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });
}

// initialize
btnLoad.addEventListener('click', async ()=>{
  const trainF=document.getElementById('train-file').files[0];
  const testF=document.getElementById('test-file').files[0];
  const jsonF=document.getElementById('model-json').files[0];
  const binF=document.getElementById('model-bin').files[0];
  if(!trainF||!testF){alert('Select train & test'); return;}
  monthly=await prepareMonthly(trainF,testF,'linear');
  renderMergeOverview();
  groups=[...new Set(monthly.map(d=>d.group))];
  if(groups.length>0){renderCancelChart(groups[0]); renderPriceChart(groups[0]);}
  if(jsonF&&binF){await loadModelFromFiles(jsonF,binF);}
});
