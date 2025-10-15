let chartCancel, chartPrice, chartHeat, chartPredict;
let model;

// smoothing slider
const smoothingSlider=document.getElementById("smoothing-slider");
smoothingSlider.addEventListener("input",()=>{
  document.getElementById("smoothing-val").textContent=smoothingSlider.value;
  drawEDA();
});

function drawEDA(){
  if(!mergedData.length) return;
  let smoothN=+smoothingSlider.value;
  let groups=[...new Set(mergedData.map(d=>d.group))];
  let grouped={};
  groups.forEach(g=>grouped[g]=mergedData.filter(d=>d.group===g));
  // cancellation_rate
  let ctxC=document.getElementById("chart-cancel").getContext("2d");
  let datasets=[];
  for(let g of groups){
    let arr=grouped[g].map(d=>d.cancellation_rate);
    let smoothed=arr.map((v,i)=>arr.slice(Math.max(0,i-smoothN+1),i+1).reduce((a,b)=>a+b,0)/Math.min(i+1,smoothN));
    datasets.push({label:g,data:smoothed,borderColor=getColor(g),fill:false,tension:0.2});
  }
  if(chartCancel) chartCancel.destroy();
  chartCancel=new Chart(ctxC,{type:"line",data:{labels:mergedData.filter(d=>d.group===groups[0]).map(d=>d.month_index),datasets},options:{plugins:{tooltip:{enabled:true}}}});
  
  // avg_room_price histogram
  let ctxP=document.getElementById("chart-price").getContext("2d");
  if(chartPrice) chartPrice.destroy();
  chartPrice=new Chart(ctxP,{type:"bar",data:{labels:mergedData.map(d=>d.month_index),datasets:[{label:"avg_room_price",data:mergedData.map(d=>d.avg_room_price),backgroundColor:"#0288d1"}]},options:{plugins:{tooltip:{enabled:true}}}});
  
  // correlation heatmap
  let ctxH=document.getElementById("chart-heatmap").getContext("2d");
  if(chartHeat) chartHeat.destroy();
  let corr=[["cancellation_rate","avg_room_price","lead_time_avg"]];
  chartHeat=new Chart(ctxH,{type:"matrix",data:{datasets:[{label:"corr",data:generateHeatData(),backgroundColor:(ctx)=>`rgba(2,84,138,0.7)`}]}});
}

function getColor(g){
  const colors=["#0288d1","#03a9f4","#00bcd4","#26c6da","#00acc1","#00838f","#006064"];
  let idx=parseInt(g.replace(/\D/g,""))||0;
  return colors[idx%colors.length];
}

function generateHeatData(){ // placeholder
  return mergedData.map((d,i)=>({x:i,y:i,v:0.5}));
}

// Train LSTM
document.getElementById("btn-train").addEventListener("click",async()=>{
  let epochs=+document.getElementById("input-epochs").value;
  let batch=+document.getElementById("input-batch").value;
  document.getElementById("train-status").textContent="Training...";
  model=tf.sequential();
  model.add(tf.layers.lstm({units:50,inputShape:[1,3],returnSequences:true}));
  model.add(tf.layers.lstm({units:50,returnSequences:true}));
  model.add(tf.layers.lstm({units:50}));
  model.add(tf.layers.dense({units:1}));
  model.compile({loss:"meanSquaredError",optimizer:"adam"});
  let x=tf.tensor3d(mergedData.map(d=>[d.cancellation_rate,d.avg_room_price,d.lead_time_avg]).map(a=>[a]));
  let y=tf.tensor2d(mergedData.map(d=>d.cancellation_rate).map(v=>[v]));
  for(let e=1;e<=epochs;e++){
    await model.fit(x,y,{epochs:1,batchSize:batch});
    document.getElementById("train-status").textContent=`Training epoch ${e}/${epochs} ...`;
  }
  document.getElementById("train-status").textContent="Train done.";
  document.getElementById("btn-download").disabled=false;
});

// Download model
document.getElementById("btn-download").addEventListener("click",async()=>{
  if(!model) return;
  await model.save("downloads://nextstay-model");
});

// Predict
document.getElementById("btn-predict").addEventListener("click",async()=>{
  const selGroup=document.getElementById("sel-group").value;
  const year=+document.getElementById("inp-year").value;
  if(!model) {alert("Upload or train model first!"); return;}
  // Filter
  let data=(selGroup==="All")?mergedData:mergedData.filter(d=>d.group===selGroup);
  let months=Array.from({length:12},(_,i)=>i+1);
  let predictions=[];
  for(let m of months){
    let input=tf.tensor3d([[[
      data[0].cancellation_rate,
      data[0].avg_room_price,
      data[0].lead_time_avg
    ]]]);
    let pred=(await model.predict(input).data())[0];
    predictions.push(pred);
  }
  // update chart
  let ctx=document.getElementById("chart-predict").getContext("2d");
  if(chartPredict) chartPredict.destroy();
  chartPredict=new Chart(ctx,{type:"line",data:{labels:months,datasets:[
    {label:"Historical",data:data.slice(0,12).map(d=>d.cancellation_rate),borderColor:"#0288d1",fill:false},
    {label:"Prediction",data:predictions,borderColor:"#d32f2f",fill:false,pointStyle:"rectRot"}
  ]},options:{plugins:{tooltip:{enabled:true}}}});

  // insight
  let avgPred=predictions.reduce((a,b)=>a+b,0)/predictions.length;
  let box=document.getElementById("insight");
  if(avgPred>0.6) box.className="insight high";
  else if(avgPred>0.3) box.className="insight medium";
  else box.className="insight low";
  box.textContent=`Average predicted cancellation_rate: ${avgPred.toFixed(2)}`;
});
