let model = null;
let hotelList = [];
let historyChart = null;

document.getElementById('run-eda-btn').addEventListener('click', runEDA);
document.getElementById('load-model-btn').addEventListener('click', loadTFModel);
document.getElementById('predict-btn').addEventListener('click', predictNextMonth);
document.getElementById('smoothing-slider').addEventListener('input', e => {
  document.getElementById('smoothing-value').innerText = e.target.value;
});

function runEDA() {
  if (!mergedData) return alert('No data loaded');

  // Overview table
  let cols = Object.keys(mergedData[0]);
  let tableHTML = '<table><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
  mergedData.slice(0,10).forEach(r=>{
    tableHTML += '<tr>' + cols.map(c=>`<td>${r[c]}</td>`).join('') + '</tr>';
  });
  tableHTML += '</tbody></table>';
  document.getElementById('overview').innerHTML = tableHTML;

  // Correlation heatmap
  let corrData = ['cancellation_rate','avg_room_price','lead_time_avg'].map(c=>mergedData.map(d=>d[c]||0));
  const ctxHeat = document.getElementById('correlation-heatmap').getContext('2d');
  new Chart(ctxHeat,{
    type:'matrix',
    data:{datasets:[{data:[],backgroundColor:()=>''}]}
  }); // placeholder, use chart.js matrix plugin or external lib for real heatmap

  // Cancellation over time
  const monthIdx = mergedData.map(d=>d.month_index||0);
  const cancelRate = mergedData.map(d=>d.cancellation_rate||0);
  const ctxTime = document.getElementById('cancellation-time').getContext('2d');
  new Chart(ctxTime,{type:'line',data:{labels:monthIdx,datasets:[{label:'Cancellation Rate',data:cancelRate,borderColor:'red',fill:false}]}});

  // Histogram avg_room_price
  const ctxHist = document.getElementById('price-hist').getContext('2d');
  new Chart(ctxHist,{type:'bar',data:{labels:mergedData.map(d=>d.avg_room_price),datasets:[{label:'Avg Room Price',data:mergedData.map(d=>d.avg_room_price),backgroundColor:'#0077b6'}]}});
  
  // Populate hotel dropdown
  hotelList = [...new Set(mergedData.map(d=>d.hotel))];
  const sel = document.getElementById('hotel-select');
  sel.innerHTML = hotelList.map(h=>`<option value="${h}">${h}</option>`).join('');
}

async function loadTFModel() {
  const jsonFile = document.getElementById('model-json').files[0];
  if (!jsonFile) return alert('Upload model JSON');
  model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile]));
  alert('Model loaded');
}

function smoothSequence(seq, alpha) {
  let result = [];
  seq.reduce((prev, curr, i) => result[i] = alpha*curr + (1-alpha)*prev, seq[0]);
  return result;
}

async function predictNextMonth() {
  if (!model) return alert('Model not loaded');
  const hotel = document.getElementById('hotel-select').value;
  const year = parseInt(document.getElementById('year-input').value);
  const month = parseInt(document.getElementById('month-input').value);
  const alpha = parseFloat(document.getElementById('smoothing-slider').value);

  let seq = mergedData.filter(d=>d.hotel===hotel).sort((a,b)=>a.month_index-b.month_index)
    .map(d=>[d.cancellation_rate,d.avg_room_price,d.lead_time_avg]);
  seq = smoothSequence(seq, alpha);

  const input = tf.tensor([seq.slice(-12)]).reshape([1,12,3]);
  let pred = model.predict(input);
  const value = pred.dataSync()[0];

  document.getElementById('prediction-result').innerText = `Predicted Cancellation Rate: ${(value*100).toFixed(2)}%`;

  // Insight box
  const insightBox = document.getElementById('insight-box');
  if (value>0.6) insightBox.innerText = 'High risk — offer early payment discount', insightBox.style.backgroundColor='red';
  else if (value>0.3) insightBox.innerText = 'Medium risk — flexible policy', insightBox.style.backgroundColor='yellow';
  else insightBox.innerText = 'Low risk — consider price increase next month', insightBox.style.backgroundColor='green';

  // Update chart
  const ctx = document.getElementById('history-chart').getContext('2d');
  if(historyChart) historyChart.destroy();
  historyChart = new Chart(ctx,{
    type:'line',
    data:{
      labels:seq.map((_,i)=>i),
      datasets:[{
        label:'Historical Cancellation Rate',
        data:seq.map(d=>d[0]),
        borderColor:'blue',
        fill:false
      },{
        label:'Predicted Next Month',
        data:[...Array(seq.length-1).fill(null),value],
        borderColor:'red',
        pointRadius:6,
        pointBackgroundColor:'red',
        fill:false
      }]
    }
  });
}
