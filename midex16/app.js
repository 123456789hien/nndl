let chartCancel, chartPrice, chartHeat, chartPredict;
let model;
let minMax={};

function computeMinMax(){
  ["cancellation_rate","avg_room_price","lead_time_avg"].forEach(c=>{
    const vals = mergedData.map(d=>d[c]);
    minMax[c]={min:Math.min(...vals), max:Math.max(...vals)};
  });
}

function scaleValue(val,c){ const {min,max}=minMax[c]; return (val-min)/(max-min);}
function inverseScale(val,c){ const {min,max}=minMax[c]; return val*(max-min)+min;}

function getFilteredData(grpKey=document.getElementById("group-key").value){
  return mergedData.filter(d=>grpKey==="All"||d.group===grpKey);
}

function smoothArray(arr,window){if(window<=1) return arr; return arr.map((v,i)=>{ const start=Math.max(0,i-window+1); const slice=arr.slice(start,i+1); return slice.reduce((a,b)=>a+b,0)/slice.length; });}

function correlationHeatmap(data){
  const cols=["cancellation_rate","avg_room_price","lead_time_avg"];
  const res=[];
  for(let i=0;i<cols.length;i++){
    for(let j=0;j<cols.length;j++){
      const xi=data.map(d=>d[cols[i]]), yj=data.map(d=>d[cols[j]]);
      const meanX=xi.reduce((a,b)=>a+b,0)/xi.length;
      const meanY=yj.reduce((a,b)=>a+b,0)/yj.length;
      let num=0,denX=0,denY=0;
      for(let k=0;k<xi.length;k++){
        num+=(xi[k]-meanX)*(yj[k]-meanY);
        denX+=(xi[k]-meanX)**2;
        denY+=(yj[k]-meanY)**2;
      }
      const corr=num/Math.sqrt(denX*denY);
      res.push({x:i,y:j,v:corr});
    }
  }
  return res;
}

function renderCharts(){
  const data = getFilteredData();
  const smoothing=+document.getElementById("smoothing-slider").value;
  document.getElementById("smoothing-val").innerText=smoothing;

  const labels = data.map(d=>`${d.year}-${d.month}`);
  const yCancel = smoothArray(data.map(d=>d.cancellation_rate),smoothing);

  if(chartCancel) chartCancel.destroy();
  chartCancel=new Chart(document.getElementById("chart-cancel"),{
    type:"line",
    data:{labels,datasets:[{label:"Cancellation Rate",data:yCancel,borderColor:"red",tension:0.2,pointRadius:4}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });

  const yPrice = data.map(d=>d.avg_room_price);
  if(chartPrice) chartPrice.destroy();
  chartPrice=new Chart(document.getElementById("chart-price"),{
    type:"bar",
    data:{labels,datasets:[{label:"Avg Room Price",data:yPrice,backgroundColor:"blue"}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });

  const corrData = correlationHeatmap(data);
  if(chartHeat) chartHeat.destroy();
  chartHeat=new Chart(document.getElementById("chart-heatmap"),{
    type:"matrix",
    data:{datasets:[{label:"Correlation",data:corrData,backgroundColor:ctx=>`rgba(2,84,138,${ctx.dataset.data[ctx.dataIndex].v})`}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });
}

// Train
document.getElementById("btn-train").addEventListener("click",async()=>{
  const data=getFilteredData().map(d=>["cancellation_rate","avg_room_price","lead_time_avg"].map(c=>scaleValue(d[c],c)));
  const inputTensor=tf.tensor2d(data);

  const modelLSTM=tf.sequential();
  for(let i=0;i<3;i++) modelLSTM.add(tf.layers.dense({units:50,activation:"relu",inputShape:i===0?[3]:undefined}));
  modelLSTM.add(tf.layers.dense({units:3}));
  modelLSTM.compile({loss:"meanSquaredError",optimizer:"adam"});

  const epochs=+document.getElementById("input-epochs").value;
  const batch=+document.getElementById("input-batch").value;
  document.getElementById("train-status").innerText="Training...";
  await modelLSTM.fit(inputTensor,inputTensor,{epochs,batchSize:batch,callbacks:{
    onEpochEnd:(epoch,logs)=>{document.getElementById("train-status").innerText=`Training epoch ${epoch+1}/${epochs}`;}
  }});
  document.getElementById("train-status").innerText="Train done";
  model=modelLSTM;
  document.getElementById("btn-download").disabled=false;
});

// Download
document.getElementById("btn-download").addEventListener("click",async()=>{
  if(!model) return;
  await model.save('downloads://nextstay-model');
});

// Predict 12 months for 3 vars
document.getElementById("btn-predict").addEventListener("click",async()=>{
  if(!model) return alert("Model not ready");
  const grp=document.getElementById("sel-group").value;
  const year=+document.getElementById("inp-year").value;
  const data=mergedData.filter(d=>grp==="All"||d.group===grp);
  const predictions=[];

  for(let m=1;m<=12;m++){
    const input = data.filter(d=>d.month===m).map(d=>["cancellation_rate","avg_room_price","lead_time_avg"].map(c=>scaleValue(d[c],c)));
    if(input.length===0){ predictions.push([0,0,0]); continue;}
    const pred=model.predict(tf.tensor2d(input)).arraySync();
    const avgPred=pred[0].map((_,i)=>pred.reduce((a,b)=>a+b[i],0)/pred.length);
    predictions.push(avgPred.map((v,i)=>inverseScale(v,["cancellation_rate","avg_room_price","lead_time_avg"][i])));
  }

  const labels=Array.from({length:12},(_,i)=>`${year}-${i+1}`);
  const cancelPred=predictions.map(p=>p[0].toFixed(2));
  const pricePred=predictions.map(p=>p[1].toFixed(2));
  const leadPred=predictions.map(p=>p[2].toFixed(2));
  document.getElementById("predict-result").innerText=`Cancellation Rate: ${cancelPred.join(", ")}`;

  if(chartPredict) chartPredict.destroy();
  chartPredict=new Chart(document.getElementById("chart-predict"),{
    type:"line",
    data:{labels,datasets:[
      {label:"Cancellation Rate",data:cancelPred,borderColor:"red",tension:0.2,pointRadius:4},
      {label:"Avg Room Price",data:pricePred,borderColor:"blue",tension:0.2,pointRadius:4},
      {label:"Lead Time Avg",data:leadPred,borderColor:"green",tension:0.2,pointRadius:4},
    ]},
    options:{plugins:{tooltip:{enabled:true}}}
  });

  const avgPred=cancelPred.reduce((a,b)=>a+parseFloat(b),0)/cancelPred.length;
  const insightEl=document.getElementById("insight");
  if(avgPred>0.6) insightEl.className="insight high";
  else if(avgPred>0.3) insightEl.className="insight medium";
  else insightEl.className="insight low";
  insightEl.innerText=`Average predicted cancellation rate: ${avgPred.toFixed(2)}`;
});

document.addEventListener("DOMContentLoaded",()=>{
  computeMinMax();
  renderCharts();
  document.getElementById("smoothing-slider").addEventListener("input",()=>renderCharts());
  document.getElementById("group-key").addEventListener("change",()=>renderCharts());
});
