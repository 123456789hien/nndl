let model=null, predChart=null;

// ---------------------- EDA ----------------------
function runEDA(){
  if(!merged) return alert('No data loaded');
  renderOverview();
  prepareMonthlyData();
  renderCharts();
  populateHotelDropdown();
}

// Merge monthly, fill missing, smoothing
function prepareMonthlyData(){
  monthlyData={};
  const groupCols=['hotel','year','month'];
  const hotels=[...new Set(merged.map(r=>r.hotel))];
  hotels.forEach(h=>{
    const hotelRows=merged.filter(r=>r.hotel===h);
    const months=[...new Set(hotelRows.map(r=>r.month))];
    const data=[];
    months.forEach(m=>{
      const rows=hotelRows.filter(r=>r.month==m);
      const total=rows.length;
      const cancelled=rows.filter(r=>r.status==1).length;
      const avgPrice=rows.reduce((s,r)=>s+Number(r.avg_room_price),0)/total;
      const leadTime=rows.reduce((s,r)=>s+Number(r.lead_time),0)/total;
      let cr=cancelled/total;
      data.push({year:rows[0].year, month:m, total, cancelled, avg_room_price:avgPrice, lead_time_avg:leadTime, cancellation_rate:cr});
    });
    // sort & smooth
    data.sort((a,b)=>a.year*12+a.month - (b.year*12+b.month));
    for(let i=1;i<data.length;i++){
      data[i].cancellation_rate=data[i].cancellation_rate*(1-smoothingLevel*0.1)+data[i-1].cancellation_rate*(smoothingLevel*0.1);
    }
    monthlyData[h]=data;
  });
}

function renderOverview(){
  document.getElementById('overview').innerText=`Merged shape: ${merged.length} rows × ${Object.keys(merged[0]).length} cols`;
}

// ---------------------- Populate Hotel Dropdown ----------------------
function populateHotelDropdown(){
  const dropdown=document.getElementById('hotel-dropdown');
  dropdown.innerHTML='';
  Object.keys(monthlyData).forEach(h=>{
    const opt=document.createElement('option'); opt.value=h; opt.text=h;
    dropdown.appendChild(opt);
  });
}

// ---------------------- Charts ----------------------
function renderCharts(){
  const hotel=document.getElementById('hotel-dropdown').value || Object.keys(monthlyData)[0];
  if(!hotel || !monthlyData[hotel]) return;
  const data=monthlyData[hotel];

  // Cancellation Rate over time
  const ctx=document.getElementById('prediction-chart').getContext('2d');
  if(predChart) predChart.destroy();
  predChart=new Chart(ctx,{
    type:'line',
    data:{
      labels:data.map(r=>`${r.year}-${r.month}`),
      datasets:[{
        label:'Cancellation Rate',
        data:data.map(r=>r.cancellation_rate),
        borderColor:'blue',
        fill:false
      }]
    }
  });
}

// ---------------------- LSTM ----------------------
async function trainLSTM(){
  const hotel=document.getElementById('hotel-dropdown').value;
  const seq=monthlyData[hotel].map(r=>[r.lead_time_avg,r.avg_room_price,r.cancellation_rate]);
  const X=[], Y=[];
  for(let i=0;i<seq.length-1;i++){ X.push([seq[i]]); Y.push(seq[i][2]); }

  const xs=tf.tensor3d(X); const ys=tf.tensor2d(Y,[Y.length,1]);

  model=tf.sequential();
  model.add(tf.layers.lstm({units:50, returnSequences:true, inputShape:[1,3]}));
  model.add(tf.layers.lstm({units:50, returnSequences:true}));
  model.add(tf.layers.lstm({units:50}));
  model.add(tf.layers.dense({units:1, activation:'linear'}));
  model.compile({optimizer:'adam', loss:'meanSquaredError'});
  await model.fit(xs,ys,{epochs:50});
  await model.save('localstorage://hotel_lstm_model');
  alert('Model trained & saved in localStorage!');
}

async function loadModelFromStorage(){
  try{
    model=await tf.loadLayersModel('localstorage://hotel_lstm_model');
    alert('Model loaded from localStorage!');
  }catch(e){ alert('No saved model found'); console.error(e);}
}

// ---------------------- Predict ----------------------
async function predictNextMonth(){
  if(!model) return alert('Train or load model first');
  const hotel=document.getElementById('hotel-dropdown').value;
  const year=Number(document.getElementById('input-year').value);
  const month=Number(document.getElementById('input-month').value);

  const seq=monthlyData[hotel].slice(-12).map(r=>[r.lead_time_avg,r.avg_room_price,r.cancellation_rate]);
  const padded=seq.length<12? Array(12-seq.length).fill([0,0,0]).concat(seq) : seq;
  const input=tf.tensor([padded]);
  const pred=model.predict(input).dataSync()[0];

  const rates=monthlyData[hotel].map(r=>r.cancellation_rate);
  const minCR=Math.min(...rates), maxCR=Math.max(...rates);
  const predUnscaled=pred*(maxCR-minCR)+minCR;

  const box=document.getElementById('insight-box');
  if(predUnscaled>0.7){ box.innerText="High risk — offer early payment / deposit"; box.style.backgroundColor='red'; }
  else if(predUnscaled<0.4){ box.innerText="Low risk — consider price increase"; box.style.backgroundColor='green'; }
  else{ box.innerText="Medium risk — flexible policy"; box.style.backgroundColor='orange'; }

  document.getElementById('prediction-result').innerText=`Predicted Cancellation Rate: ${(predUnscaled*100).toFixed(1)}%`;

  // Update Chart
  const ctx=document.getElementById('prediction-chart').getContext('2d');
  if(predChart) predChart.destroy();
  predChart=new Chart(ctx,{
    type:'line',
    data:{
      labels:monthlyData[hotel].slice(-12).map(r=>`${r.year}-${r.month}`).concat(`${year}-${month}`),
      datasets:[
        {label:'Historical Cancellation Rate', data:monthlyData[hotel].slice(-12).map(r=>r.cancellation_rate), borderColor:'blue', fill:false},
        {label:'Predicted Next Month', data:[...Array(12).fill(null), predUnscaled], borderColor:'red', fill:false, pointRadius:6}
      ]
    }
  });
}
