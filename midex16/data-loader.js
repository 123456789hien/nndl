let trainData=[], testData=[], mergedData=[];

document.getElementById("btn-load").addEventListener("click",()=>{
  const trainFile = document.getElementById("train-file").files[0];
  const testFile = document.getElementById("test-file").files[0];
  if(!trainFile||!testFile){ alert("Select both train and test CSV"); return; }

  document.getElementById("note-load").innerText="Loading...";
  Papa.parse(trainFile,{header:true, skipEmptyLines:true, complete: function(results){
    trainData = cleanData(results.data);
    Papa.parse(testFile,{header:true, skipEmptyLines:true, complete: function(results){
      testData = cleanData(results.data);
      mergeCSV();
      document.getElementById("note-load").innerText="Load done";
    }});
  }});
});

function cleanData(data){
  return data.map(d=>{
    return {
      group:d.group || "undefined",
      year:+d.year,
      month:+d.month,
      total_bookings: +d.total_bookings || 0,
      cancelled_bookings: +d.cancelled_bookings || 0,
      cancellation_rate: +d.cancellation_rate || 0,
      avg_room_price: +d.avg_room_price || 0,
      lead_time_avg: +d.lead_time_avg || 0,
      month_index:+d.month_index||0
    }
  }).filter(d=>d.total_bookings>0); // remove zero rows
}

function mergeCSV(){
  mergedData = [...trainData,...testData];
  showOverview();
}

function showOverview(){
  const table = document.getElementById("merge-overview");
  let html = "<table><thead><tr><th>Group</th><th>Year</th><th>Month</th><th>Total</th><th>Cancelled</th><th>Cancel Rate</th><th>Avg Price</th><th>Lead Time</th></tr></thead><tbody>";
  mergedData.slice(0,10).forEach(d=>{
    html+=`<tr>
      <td>${d.group}</td>
      <td>${d.year}</td>
      <td>${d.month}</td>
      <td>${d.total_bookings}</td>
      <td>${d.cancelled_bookings}</td>
      <td>${d.cancellation_rate.toFixed(3)}</td>
      <td>${d.avg_room_price.toFixed(2)}</td>
      <td>${d.lead_time_avg.toFixed(2)}</td>
    </tr>`;
  });
  html+="</tbody></table>";
  table.innerHTML=html;

  // Stats table
  const cols = ["cancellation_rate","avg_room_price","lead_time_avg"];
  let stats="<table><thead><tr><th>Var</th><th>Mean</th><th>Min</th><th>Max</th></tr></thead><tbody>";
  cols.forEach(c=>{
    const vals = mergedData.map(d=>d[c]).filter(v=>v>0);
    stats+=`<tr><td>${c}</td><td>${(vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(4)}</td><td>${Math.min(...vals).toFixed(4)}</td><td>${Math.max(...vals).toFixed(4)}</td></tr>`;
  });
  stats+="</tbody></table>";
  document.getElementById("stats-table").innerHTML=stats;
}
