let heatmapChart, lineChart, histChart, predChart;

document.getElementById('loadBtn').addEventListener('click', () => {
  const trainFile = document.getElementById('trainFile').files[0];
  const testFile = document.getElementById('testFile').files[0];
  if (!trainFile || !testFile) {
    alert("Please upload both train and test CSV files!");
    return;
  }
  document.getElementById('loadStatus').textContent = "Loading...";
  loadCSVFiles(trainFile, testFile, () => {
    document.getElementById('loadStatus').textContent = "✅ Data Loaded Successfully!";
    showOverview();
    showStats();
    drawEDA();
    setupPredictDropdown();
  });
});

function showOverview() {
  document.getElementById('overview').classList.remove('hidden');
  const container = document.getElementById('dataTable');
  const sample = mergedData.slice(0, 10);
  const keys = Object.keys(sample[0]);
  let html = `<table><tr>${keys.map(k=>`<th>${k}</th>`).join('')}</tr>`;
  sample.forEach(row=>{
    html += `<tr>${keys.map(k=>`<td>${row[k]}</td>`).join('')}</tr>`;
  });
  html += '</table>';
  container.innerHTML = html;
}

function showStats() {
  document.getElementById('stats').classList.remove('hidden');
  const container = document.getElementById('statsTable');
  const hotels = new Set(mergedData.map(d=>d.hotel));
  const avgPrice = d3.mean(mergedData, d=>d.avg_room_price).toFixed(2);
  const cancelRate = d3.mean(mergedData, d=>d.cancellation_rate).toFixed(2);
  container.innerHTML = `
    <table>
      <tr><th>Total Hotels</th><th>Avg Room Price</th><th>Avg Cancellation Rate</th></tr>
      <tr><td>${hotels.size}</td><td>${avgPrice}</td><td>${cancelRate}</td></tr>
    </table>`;
}

function drawEDA() {
  document.getElementById('eda').classList.remove('hidden');
  const vars = ['avg_room_price', 'lead_time', 'cancellation_rate'];
  const corrMatrix = vars.map(v1 => vars.map(v2 => d3.correlation(mergedData, d=>+d[v1], d=>+d[v2])));
  const ctxH = document.getElementById('corrHeatmap').getContext('2d');
  if (heatmapChart) heatmapChart.destroy();
  heatmapChart = new Chart(ctxH, {
    type: 'matrix',
    data: {
      datasets: [{
        data: vars.flatMap((v1, i)=>vars.map((v2,j)=>({
          x: v1, y: v2, v: corrMatrix[i][j]
        }))),
        backgroundColor: ctx => {
          const v = ctx.dataset.data[ctx.dataIndex].v;
          const c = Math.floor((v+1)*127);
          return `rgb(${255-c},${c},${180})`;
        },
        width: ({chart}) => chart.chartArea.width/vars.length - 10,
        height: ({chart}) => chart.chartArea.height/vars.length - 10
      }]
    },
    options: {
      scales: { x: { type:'category', labels:vars }, y: { type:'category', labels:vars, reverse:true } },
      plugins: { tooltip: { callbacks: { label: ctx=>`r=${ctx.raw.v.toFixed(2)}` } } }
    }
  });

  const ctxL = document.getElementById('lineChart').getContext('2d');
  const byMonth = d3.rollups(mergedData, v=>d3.mean(v, d=>d.cancellation_rate), d=>d.month).sort((a,b)=>a[0]-b[0]);
  if (lineChart) lineChart.destroy();
  lineChart = new Chart(ctxL, {
    type: 'line',
    data: {
      labels: byMonth.map(d=>`M${d[0]}`),
      datasets: [{ label: 'Cancellation Rate', data: byMonth.map(d=>d[1]) }]
    }
  });

  const ctxH2 = document.getElementById('histChart').getContext('2d');
  if (histChart) histChart.destroy();
  histChart = new Chart(ctxH2, {
    type: 'bar',
    data: {
      labels: mergedData.map(d=>d.avg_room_price),
      datasets: [{ label: 'Avg Room Price', data: mergedData.map(d=>d.avg_room_price) }]
    },
    options: { scales: { x: { title:{display:true, text:'Avg Room Price'} }, y: { title:{display:true, text:'Frequency'} } } }
  });
}

function setupPredictDropdown() {
  document.getElementById('predict').classList.remove('hidden');
  const select = document.getElementById('hotelSelect');
  const hotels = [...new Set(mergedData.map(d=>d.hotel))];
  select.innerHTML = hotels.map(h=>`<option>${h}</option>`).join('');
}

document.getElementById('predictBtn').addEventListener('click', ()=>{
  const hotel = document.getElementById('hotelSelect').value;
  const year = +document.getElementById('yearInput').value;
  const month = +document.getElementById('monthInput').value;
  const subset = mergedData.filter(d=>d.hotel===hotel && d.year===year);
  const pred = subset.length ? d3.mean(subset.filter(d=>d.month<=month), d=>d.cancellation_rate) : Math.random()*0.3;
  const risk = pred>0.7?'high':pred>0.4?'medium':'low';
  const box = document.getElementById('predictionResult');
  box.className = `insight-box insight-${risk}`;
  box.textContent = `Predicted Risk: ${risk.toUpperCase()} (${(pred*100).toFixed(1)}%)`;

  const ctxP = document.getElementById('predChart').getContext('2d');
  if (predChart) predChart.destroy();
  predChart = new Chart(ctxP, {
    type: 'line',
    data: {
      labels: Array.from({length:month}, (_,i)=>`M${i+1}`),
      datasets: [{ label:'Cancellation Rate', data: subset.map(d=>d.cancellation_rate), fill:false, borderColor:'#0077cc' }]
    }
  });
});
