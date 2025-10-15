let monthlyAgg=null, model=null;
const colorPalette=['#0077b6','#00b4d8','#90e0ef','#007f5f','#ff9f1c'];

// ----- EDA -----
function runEDA(){
  if(!merged){ alert('No data loaded'); return; }

  monthlyAgg=[];
  const hotels=[...new Set(merged.map(r=>r.hotel_type||r.hotel||'Hotel'))];
  hotels.forEach(h=>{
    const rows=merged.filter(r=>(r.hotel_type||r.hotel||'Hotel')===h);
    const ymMap={};
    rows.forEach(r=>{
      const ym=`${r.year}-${r.month}`;
      if(!ymMap[ym]) ymMap[ym]={total:0,cancelled:0,priceSum:0,leadSum:0};
      ymMap[ym].total++;
      ymMap[ym].cancelled+=r.status==1?1:0;
      ymMap[ym].priceSum+=r.avg_room_price||0;
      ymMap[ym].leadSum+=r.lead_time||0;
    });
    Object.keys(ymMap).forEach(ym=>{
      const [year,month]=ym.split('-').map(Number);
      const d=ymMap[ym];
      monthlyAgg.push({
        hotel:h,year,month,
        total_bookings:d.total,
        cancelled_bookings:d.cancelled,
        cancellation_rate:d.total?d.cancelled/d.total:0,
        avg_room_price:d.total?d.priceSum/d.total:0,
        lead_time_avg:d.total?d.leadSum/d.total:0,
        month_index:(year-2018)*12+(month-1)
      });
    });
  });

  renderOverviewTable();
  renderMissingTable();
  renderEDACharts();
}

// ----- Overview & Missing -----
function renderOverviewTable(){
  const table=document.getElementById('overview-table');
  const hotels=[...new Set(monthlyAgg.map(d=>d.hotel))];
  let html='<table><tr><th>Hotel</th><th>Months</th><th>Total Bookings</th><th>Avg Cancellation Rate</th></tr>';
  hotels.forEach(h=>{
    const arr=monthlyAgg.filter(d=>d.hotel===h);
    html+=`<tr><td>${h}</td><td>${arr.length}</td><td>${arr.reduce((a,b)=>a+b.total_bookings,0)}</td><td>${(arr.reduce((a,b)=>a+b.cancellation_rate,0)/arr.length*100).toFixed(2)}%</td></tr>`;
  });
  html+='</table>';
  table.innerHTML=html;
}

function renderMissingTable(){
  const table=document.getElementById('missing-table');
  const hotels=[...new Set(monthlyAgg.map(d=>d.hotel))];
  let html='<table><tr><th>Hotel</th><th>Missing Months</th></tr>';
  hotels.forEach(h=>{
    const arr=monthlyAgg.filter(d=>d.hotel===h);
    const missing=12-arr.length;
    html+=`<tr><td>${h}</td><td>${missing}</td></tr>`;
  });
  html+='</table>';
  table.innerHTML=html;
}

// ----- Charts -----
function renderEDACharts(){
  // Histogram
  new Chart(document.getElementById('avg-room-price-hist'),{
    type:'bar',
    data:{labels:monthlyAgg.map(d=>`${d.hotel}-${d.month_index}`), datasets:[{label:'Avg Room Price',data:monthlyAgg.map(d=>d.avg_room_price),backgroundColor:'#0077b6'}]},
    options:{plugins:{tooltip:{enabled:true}}, responsive:true}
  });

  // Trend
  const hotels=[...new Set(monthlyAgg.map(d=>d.hotel))];
  const ctx=document.getElementById('cancellation-trend').getContext('2d');
  const datasets=hotels.map((h,i)=>({label:h,data:monthlyAgg.filter(d=>d.hotel===h).sort((a,b)=>a.month_index-b.month_index).map(d=>d.cancellation_rate*100),borderColor:colorPalette[i%colorPalette.length],fill:false}));
  new Chart(ctx,{type:'line',data:{labels:monthlyAgg.filter(d=>d.hotel===hotels[0]).sort((a,b)=>a.month_index-b.month_index).map(d=>d.month_index),datasets:datasets},options:{plugins:{tooltip:{enabled:true}}}});
  
  // Correlation Heatmap
  renderCorrelationHeatmap();
}

function renderCorrelationHeatmap(){
  const cols=['cancellation_rate','avg_room_price','lead_time_avg'];
  const n=cols.length;
  const data=[];
  for(let i=0;i<n;i++){
    for(let j=0;j<n;j++){
      const xi=monthlyAgg.map(d=>d[cols[i]]);
      const xj=monthlyAgg.map(d=>d[cols[j]]);
      const val=pearson(xi,xj);
      data.push({x:j,y:i,v:val});
    }
  }
  const ctx=document.getElementById('correlation-heatmap').getContext('2d');
  new Chart(ctx,{type:'matrix',data:{datasets:[{label:'Correlation',data:data,backgroundColor:data.map(d=>`rgba(0,123,255,${Math.abs(d.v)})`)}]},options:{plugins:{tooltip:{callbacks:{label:function(ctx){return `${cols[ctx.dataIndex%n]} vs ${cols[Math.floor(ctx.dataIndex/n)]}: ${data[ctx.dataIndex].v.toFixed(2)}`;}}}}}});
}

function pearson(x,y){
  const n=x.length; const mx=x.reduce((a,b)=>a+b,0)/n; const my=y.reduce((a,b)=>a+b,0)/n;
  let num=0,dx=0,dy=0; for(let i=0;i<n;i++){ num+=(x[i]-mx)*(y[i]-my); dx+=(x[i]-mx)**2; dy+=(y[i]-my)**2; }
  return num/Math.sqrt(dx*dy);
}

// ----- TF.js LSTM -----
async function trainModel(){
  if(!monthlyAgg){ alert('Run EDA first'); return; }
  const seqs=monthlyAgg.map(d=>[d.cancellation_rate,d.avg_room_price,d.lead_time_avg]);
  const xs=tf.tensor(seqs).reshape([seqs.length,1,3]);
  const ys=tf.tensor(monthlyAgg.map(d=>d.cancellation_rate));
  const m=tf.sequential();
  m.add(tf.layers.lstm({units:50,inputShape:[1,3],returnSequences:true}));
  m.add(tf.layers.lstm({units:50,returnSequences:true}));
  m.add(tf.layers.lstm({units:50}));
  m.add(tf.layers.dense({units:1}));
  m.compile({optimizer:'adam',loss:'meanSquaredError'});
  await m.fit(xs,ys,{epochs:50});
  model=m;
  await model.save('localstorage://hotel-model');
  alert('Model trained & saved');
}

async function loadTFModel(){
  model=await tf.loadLayersModel('localstorage://hotel-model');
  alert('Loaded from localStorage');
}

// ----- Predict + smoothing -----
function smoothSequence(seq,factor){
  if(factor<=0) return seq;
  const res=[...seq];
  for(let i=1;i<seq.length;i++) res[i]=res[i]*factor + res[i-1]*(1-factor);
  return res;
}

function predictCancellation(){
  if(!model){ alert('Train or load model'); return; }
  const hotel=document.getElementById('hotel-select').value;
  const year=+document.getElementById('year-input').value;
  const month=+document.getElementById('month-input').value;
  const smooth=parseFloat(document.getElementById('smoothing-slider').value);
  document.getElementById('smoothing-value').innerText=smooth;

  const seq=monthlyAgg.filter(d=>d.hotel===hotel)
    .sort((a,b)=>a.month_index-b.month_index)
    .slice(-12)
    .map(d=>[d.cancellation_rate,d.avg_room_price,d.lead_time_avg]);

  const smoothedSeq=smoothSequence(seq,smooth);
  const input=tf.tensor(smoothedSeq).reshape([1,12,3]);
  const pred=model.predict(input).dataSync()[0];

  document.getElementById('prediction-output').innerText=`Predicted Cancellation Rate: ${(pred*100).toFixed(2)} %`;

  const insight=document.getElementById('insight-box');
  if(pred>0.6){ insight.innerText='High risk — offer early payment discount'; insight.style.backgroundColor='red'; }
  else if(pred<0.3){ insight.innerText='Low risk — consider price increase'; insight.style.backgroundColor='green'; }
  else{ insight.innerText='Medium risk — flexible policy'; insight.style.backgroundColor='orange'; }

  updatePredictionChart(hotel,pred);
}

function updatePredictionChart(hotel,pred){
  const data=monthlyAgg.filter(d=>d.hotel===hotel).sort((a,b)=>a.month_index-b.month_index);
  const labels=data.map(d=>`${d.year}-${d.month}`);
  const histData=data.map(d=>d.cancellation_rate*100);
  if(window.predChart) window.predChart.destroy();
  window.predChart=new Chart(document.getElementById('prediction-chart').getContext('2d'),{
    type:'line',
    data:{labels:labels.concat(['Predicted']),datasets:[{label:'Cancellation Rate',data:histData.concat([pred*100]),borderColor:'#0077b6',fill:false}]},
    options:{plugins:{tooltip:{enabled:true}}}
  });
}
