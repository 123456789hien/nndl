let chartCancel, chartPrice, chartHeatmap, chartPredict;
let model;
let smoothingWindow=1;
let currentGroup="room_type";

document.getElementById("smoothing-slider").addEventListener("input", e=>{
  smoothingWindow=parseInt(e.target.value);
  document.getElementById("smoothing-val").innerText=smoothingWindow;
  renderEDA();
});
document.getElementById("group-key").addEventListener("change", e=>{
  currentGroup=e.target.value;
  renderEDA();
});

// ------------------- EDA -------------------
function renderEDA(){
  if(!mergedData.length) return;
  const lineData=aggregateByMonth(mergedData,"cancellation_rate",currentGroup,smoothingWindow);
  const priceData=aggregateByMonth(mergedData,"avg_room_price",currentGroup,1);

  if(chartCancel) chartCancel.destroy();
  chartCancel=new Chart(document.getElementById("chart-cancel"),{
    type:"line",
    data:{labels:lineData.labels,datasets:[{label:"Cancellation Rate",data:lineData.data,borderColor:"#0288d1",tension:0.3}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });

  if(chartPrice) chartPrice.destroy();
  chartPrice=new Chart(document.getElementById("chart-price"),{
    type:"bar",
    data:{labels:priceData.labels,datasets:[{label:"Avg Room Price",data:priceData.data,backgroundColor:"#02548a"}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });

  // Heatmap placeholder
  if(chartHeatmap) chartHeatmap.destroy();
  chartHeatmap=new Chart(document.getElementById("chart-heatmap"),{
    type:"matrix",
    data:{datasets:[{label:"Corr",data:[],backgroundColor:()=>"#0288d1"}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });
}

function aggregateByMonth(data,col,groupKey,smooth=1){
  const grouped={};
  data.forEach(d=>{
    const key=d[groupKey]+"-"+d.year+"-"+d.month;
    if(!grouped[key]) grouped[key]=[];
    grouped[key].push(d[col]);
  });
  const labels=Object.keys(grouped).sort();
  let dataArr=labels.map(k=>grouped[k].reduce((a,b)=>a+b,0)/grouped[k].length);
  // smoothing
  if(smooth>1){
    dataArr=dataArr.map((v,i,arr)=>{
      const start=Math.max(0,i-smooth+1);
      const subset=arr.slice(start,i+1);
      return subset.reduce((a,b)=>a+b,0)/subset.length;
    });
  }
  return {labels,data:dataArr};
}

// ------------------- TRAIN LSTM -------------------
document.getElementById("btn-train").addEventListener("click", async ()=>{
  const epochs=parseInt(document.getElementById("input-epochs").value);
  const batch=parseInt(document.getElementById("input-batch").value);
  document.getElementById("train-status").innerText="Running…";
  await trainModel(epochs,batch);
  document.getElementById("train-status").innerText="Train done ✅";
  document.getElementById("btn-download").disabled=false;
});

async function trainModel(epochs,batch){
  // Using cancellation_rate as example
  const xs=tf.tensor2d([0,1,2,3,4,5,6,7,8,9],[10,1]);
  const ys=tf.tensor2d([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[10,1]);
  model=tf.sequential();
  model.add(tf.layers.dense({units:50,inputShape:[1],activation:"relu"}));
  model.add(tf.layers.dense({units:50,activation:"relu"}));
  model.add(tf.layers.dense({units:50,activation:"relu"}));
  model.add(tf.layers.dense({units:1}));
  model.compile({optimizer:"adam",loss:"meanSquaredError"});

  await model.fit(xs,ys,{epochs,batchSize:batch,callbacks:{
    onEpochEnd:(epoch,logs)=>{
      document.getElementById("train-status").innerText=`Epoch ${epoch+1}/${epochs} - loss: ${logs.loss.toFixed(4)}`;
    }
  }});
}

// ------------------- DOWNLOAD -------------------
document.getElementById("btn-download").addEventListener("click",async()=>{
  if(!model) return alert("Model not trained yet");
  await model.save("downloads://nextstay_model");
});

// ------------------- PREDICT -------------------
document.getElementById("btn-predict").addEventListener("click", async ()=>{
  const group=document.getElementById("sel-group").value;
  const year=parseInt(document.getElementById("inp-year").value);
  if(!model) return alert("Model not loaded or trained");

  const months=[...Array(12).keys()].map(i=>i+1);
  const input=tf.tensor2d(months,[12,1]);
  const predArr=(await model.predict(input).array()).map(v=>v[0]);

  document.getElementById("predict-result").innerText="Prediction: "+predArr.map(v=>v.toFixed(2)).join(", ");

  const avgPred=predArr.reduce((a,b)=>a+b,0)/predArr.length;
  const insightBox=document.getElementById("insight");
  if(avgPred>0.6) insightBox.className="insight high";
  else if(avgPred>0.3) insightBox.className="insight medium";
  else insightBox.className="insight low";
  insightBox.innerText=`Insight: ${insightBox.className.split(" ")[1].toUpperCase()}`;

  if(chartPredict) chartPredict.destroy();
  chartPredict=new Chart(document.getElementById("chart-predict"),{
    type:"line",
    data:{labels:months.map(m=>"Month "+m),datasets:[{label:"Bookings",data:predArr,borderColor:"#0288d1",fill:false,tension:0.3}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });
});
