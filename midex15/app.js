let chartCancel, chartPrice, chartHeat, chartPredict;
let model;

function getFilteredData(){
  const grp = document.getElementById("group-key").value;
  return mergedData.filter(d=>grp==="All"||d.group===grp);
}

function renderCharts(){
  const data = getFilteredData();
  const smoothing = +document.getElementById("smoothing-slider").value;
  document.getElementById("smoothing-val").innerText = smoothing;

  // Cancellation Rate Line
  const labels = data.map(d=>`${d.year}-${d.month}`);
  const yCancel = smoothArray(data.map(d=>d.cancellation_rate),smoothing);
  if(chartCancel) chartCancel.destroy();
  chartCancel = new Chart(document.getElementById("chart-cancel"),{
    type:"line",
    data:{labels, datasets:[{label:"Cancellation Rate",data:yCancel,borderColor:"red",tension:0.2,pointRadius:4}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });

  // Avg Room Price Histogram
  const yPrice = data.map(d=>d.avg_room_price);
  if(chartPrice) chartPrice.destroy();
  chartPrice = new Chart(document.getElementById("chart-price"),{
    type:"bar",
    data:{labels, datasets:[{label:"Avg Room Price",data:yPrice,backgroundColor:"blue"}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });

  // Correlation Heatmap
  const corrData = correlationHeatmap(data);
  if(chartHeat) chartHeat.destroy();
  chartHeat = new Chart(document.getElementById("chart-heatmap"),{
    type:"matrix",
    data:{datasets:[{label:"Correlation",data:corrData,backgroundColor:ctx=>`rgba(2,84,138,${ctx.dataset.data[ctx.dataIndex].v})`}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });
}

function smoothArray(arr,window){
  if(window<=1) return arr;
  return arr.map((v,i)=>{
    const start=Math.max(0,i-window+1);
    const slice=arr.slice(start,i+1);
    return slice.reduce((a,b)=>a+b,0)/slice.length;
  });
}

function correlationHeatmap(data){
  const vars = ["cancellation_rate","avg_room_price","lead_time_avg"];
  const vals = vars.map(v=>data.map(d=>d[v]));
  const corr = [];
  for(let i=0;i<vars.length;i++){
    for(let j=0;j<vars.length;j++){
      const vi=vals[i], vj=vals[j];
      const meanVi=vi.reduce((a,b)=>a+b,0)/vi.length;
      const meanVj=vj.reduce((a,b)=>a+b,0)/vj.length;
      const num=vi.map((x,k)=> (x-meanVi)*(vj[k]-meanVj)).reduce((a,b)=>a+b,0);
      const den=Math.sqrt(vi.map(x=>Math.pow(x-meanVi,2)).reduce((a,b)=>a+b,0)*vj.map(x=>Math.pow(x-meanVj,2)).reduce((a,b)=>a+b,0));
      const r=den?num/den:0;
      corr.push({x:i,y:j,v:r});
    }
  }
  return corr;
}

document.getElementById("smoothing-slider").addEventListener("input",()=>renderCharts());
document.getElementById("group-key").addEventListener("change",()=>renderCharts());

document.getElementById("btn-train").addEventListener("click",async()=>{
  const data = getFilteredData().map(d=>[d.cancellation_rate,d.avg_room_price,d.lead_time_avg]);
  const inputTensor = tf.tensor2d(data);
  const modelLSTM = tf.sequential();
  for(let i=0;i<3;i++){
    modelLSTM.add(tf.layers.dense({units:50,activation:"relu",inputShape:i===0?[3]:undefined}));
  }
  modelLSTM.add(tf.layers.dense({units:1}));
  modelLSTM.compile({loss:"meanSquaredError",optimizer:"adam"});

  const epochs = +document.getElementById("input-epochs").value;
  const batch = +document.getElementById("input-batch").value;
  document.getElementById("train-status").innerText="Training...";
  await modelLSTM.fit(inputTensor,inputTensor,{epochs,batchSize:batch,callbacks:{
    onEpochEnd:(epoch,logs)=>{ document.getElementById("train-status").innerText=`Training epoch ${epoch+1}/${epochs}`; }
  }});
  document.getElementById("train-status").innerText="Train done";
  model = modelLSTM;
  document.getElementById("btn-download").disabled=false;
});

document.getElementById("btn-download").addEventListener("click",async()=>{
  if(!model) return;
  await model.save('downloads://nextstay-model');
});

document.getElementById("btn-predict").addEventListener("click",async()=>{
  if(!model) return alert("Model not ready");
  const grp = document.getElementById("sel-group").value;
  const year = +document.getElementById("inp-year").value;
  const data = mergedData.filter(d=>grp==="All"||d.group===grp);
  const predictions=[];
  for(let m=1;m<=12;m++){
    const input = data.filter(d=>d.month===m).map(d=>[d.cancellation_rate,d.avg_room_price,d.lead_time_avg]);
    if(input.length===0){ predictions.push(0); continue; }
    const pred = model.predict(tf.tensor2d(input)).dataSync();
    predictions.push(pred.reduce((a,b)=>a+b,0)/pred.length);
  }
  document.getElementById("predict-result").innerText="Prediction: "+predictions.map(v=>v.toFixed(2)).join(", ");

  // Render chart
  const labels = Array.from({length:12},(_,i)=>`${year}-${i+1}`);
  if(chartPredict) chartPredict.destroy();
  chartPredict = new Chart(document.getElementById("chart-predict"),{
    type:"line",
    data:{labels,datasets:[
      {label:"Predicted",data:predictions,borderColor:"red",tension:0.2,pointRadius:4},
    ]},
    options:{plugins:{tooltip:{enabled:true}}}
  });

  // Insight
  const avgPred = predictions.reduce((a,b)=>a+b,0)/predictions.length;
  const insightEl = document.getElementById("insight");
  if(avgPred>0.6) insightEl.className="insight high";
  else if(avgPred>0.3) insightEl.className="insight medium";
  else insightEl.className="insight low";
  insightEl.innerText = `Average predicted cancellation rate: ${avgPred.toFixed(2)}`;
});

document.addEventListener("DOMContentLoaded",()=>{ renderCharts(); });
