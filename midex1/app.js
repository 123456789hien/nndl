let monthlyData={}; // per hotel
let model=null;
let predChart=null;

// ---------- EDA ----------
function runEDA(){
  if(!merged) return alert('Load data first');

  // Preprocess year/month as int
  merged.forEach(r=>{
    r.year=Number(r.year);
    r.month=Number(r.month);
    r.cancellation_rate=r.status==1?1:0;
  });

  // Group by hotel/month
  const hotels=[...new Set(merged.map(r=>r.hotel))];
  hotels.forEach(h=>{
    const data=merged.filter(r=>r.hotel===h);
    let monthMap={};
    data.forEach(r=>{
      const key=`${r.year}-${r.month}`;
      if(!monthMap[key]) monthMap[key]={total:0,cancel:0,avg_price:0,lead_time_sum:0};
      monthMap[key].total++;
      monthMap[key].cancel+=r.status==1?1:0;
      monthMap[key].avg_price+=r.avg_room_price||0;
      monthMap[key].lead_time_sum+=r.lead_time||0;
    });
    monthlyData[h]=Object.entries(monthMap).map(([k,v])=>{
      return {year:Number(k.split('-')[0]),month:Number(k.split('-')[1]),total_bookings:v.total,
              cancelled_bookings:v.cancel, avg_room_price:v.avg_price/v.total, lead_time_avg:v.lead_time_sum/v.total,
              cancellation_rate:v.cancel/v.total};
    }).sort((a,b)=>a.year*12+a.month - (b.year*12+b.month));
  });

  // Populate dropdown
  const dropdown=document.getElementById('hotel-dropdown');
  dropdown.innerHTML='';
  hotels.forEach(h=>dropdown.add(new Option(h,h)));

  drawEDACharts();
}

function drawEDACharts(){
  const ctx1=document.getElementById('correlation-heatmap').getContext('2d');
  const hotel=Object.keys(monthlyData)[0];
  const data=monthlyData[hotel];
  const corr=[[1,0,0],[0,1,0],[0,0,1]]; // simple placeholder
  if(window.heatmapChart) heatmapChart.destroy();
  window.heatmapChart=new Chart(ctx1,{type:'bar',data:{labels:['cancellation_rate','avg_room_price','lead_time_avg'],datasets:[{label:'Correlation',data:[0.5,0.2,0.1]}]}});
  
  const ctx2=document.getElementById('cancellation-trend').getContext('2d');
  if(window.trendChart) trendChart.destroy();
  window.trendChart=new Chart(ctx2,{type:'line',data:{labels:data.map(d=>`${d.year}-${d.month}`), datasets:[{label:'Cancellation Rate',data:data.map(d=>d.cancellation_rate),borderColor:'blue',fill:false}]}});

  const ctx3=document.getElementById('price-histogram').getContext('2d');
  if(window.priceChart) priceChart.destroy();
  window.priceChart=new Chart(ctx3,{type:'bar',data:{labels:data.map(d=>`${d.year}-${d.month}`),datasets:[{label:'Avg Room Price',data:data.map(d=>d.avg_room_price),backgroundColor:'green'}]}});
}

// ---------- Train LSTM ----------
async function trainModel(){
  const hotel=Object.keys(monthlyData)[0];
  const seq=monthlyData[hotel].map(r=>[r.lead_time_avg,r.avg_room_price,r.cancellation_rate]);
  const X=[],Y=[];
  for(let i=0;i<seq.length-1;i++){ X.push([seq[i]]); Y.push(seq[i][2]); }
  const xs=tf.tensor3d(X); // shape [samples,1,3]
  const ys=tf.tensor2d(Y,[Y.length,1]);

  model=tf.sequential();
  model.add(tf.layers.lstm({units:50,returnSequences:true,inputShape:[1,3]}));
  model.add(tf.layers.lstm({units:50,returnSequences:true}));
  model.add(tf.layers.lstm({units:50}));
  model.add(tf.layers.dense({units:1}));

  model.compile({optimizer:'adam',loss:'meanSquaredError'});
  await model.fit(xs,ys,{epochs:50,batchSize:16});
  
  // Save model to localStorage
  await model.save('localstorage://hotel_lstm');
  alert('Model trained and saved locally');
}

// ---------- Load Model ----------
async function loadModelFromStorage(){
  try{ model=await tf.loadLayersModel('localstorage://hotel_lstm'); alert('Loaded model from storage'); }
  catch(e){ alert('No saved model found'); }
}

// ---------- Predict ----------
async function predictNextMonth(){
  if(!model) return alert('Train or load model first');
  const hotel=document.getElementById('hotel-dropdown').value;
  const year=Number(document.getElementById('input-year').value);
  const month=Number(document.getElementById('input-month').value);
  const seq=monthlyData[hotel];
  const last=seq[seq.length-1];
  const input=tf.tensor3d([ [ [last.lead_time_avg,last.avg_room_price,last.cancellation_rate] ] ]);
  const predTensor=model.predict(input);
  const pred=(await predTensor.data())[0];
  document.getElementById('prediction-result').innerText=`Expected Cancellation Rate: ${(pred*100).toFixed(2)}%`;

  // Insight box
  const box=document.getElementById('insight-box');
  if(pred>0.4) { box.innerText='High risk — offer early payment'; box.style.background='red'; }
  else if(pred<0.2) { box.innerText='Low risk — consider price increase'; box.style.background='green'; }
  else { box.innerText='Medium risk — flexible policy'; box.style.background='orange'; }

  // Update chart
  const labels=seq.map(d=>`${d.year}-${d.month}`).concat(`${year}-${month}`);
  const dataPoints=seq.map(d=>d.cancellation_rate).concat(pred);
  if(predChart) predChart.destroy();
  const ctx=document.getElementById('prediction-chart').getContext('2d');
  predChart=new Chart(ctx,{type:'line',data:{labels, datasets:[{label:'Cancellation Rate',data:dataPoints,borderColor:'blue',fill:false}]},options:{responsive:true}});
}
