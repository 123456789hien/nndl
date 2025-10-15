let trainData=[], testData=[], mergedData=[];

function parseCSV(file, callback){
  Papa.parse(file, {
    header:true, skipEmptyLines:true,
    complete: results => callback(results.data)
  });
}

function mergeData(){
  mergedData = [...trainData,...testData].map(d=>{
    d.group = d.group || "All";
    d.total_bookings = +d.total_bookings || 0;
    d.cancelled_bookings = +d.cancelled_bookings || 0;
    d.cancellation_rate = d.total_bookings ? d.cancelled_bookings/d.total_bookings : 0;
    d.avg_room_price = +d.avg_room_price || 0;
    d.lead_time_avg = +d.lead_time_avg || 0;
    d.month_index = +d.month_index || 0;
    return d;
  });
  // Filter rows where all 3 columns are zero
  mergedData = mergedData.filter(d => d.cancellation_rate||d.avg_room_price||d.lead_time_avg);
  renderTables();
}

function renderTables(){
  const preview = mergedData.slice(0,10);
  document.getElementById("merge-overview").innerHTML = tableHTML(preview);
  document.getElementById("stats-table").innerHTML = statsHTML(mergedData);
  document.getElementById("missing-table").innerHTML = missingHTML(mergedData);
}

function tableHTML(data){
  if(!data.length) return "";
  const cols = Object.keys(data[0]);
  let html = "<table><thead><tr>"+cols.map(c=>`<th>${c}</th>`).join("")+"</tr></thead><tbody>";
  html += data.map(r=>"<tr>"+cols.map(c=>r[c]).join("")+"</tr>").join("");
  html += "</tbody></table>";
  return html;
}

function statsHTML(data){
  const cols = ["cancellation_rate","avg_room_price","lead_time_avg"];
  let html = "<table><thead><tr><th>var</th><th>mean</th><th>min</th><th>max</th></tr></thead><tbody>";
  cols.forEach(c=>{
    const vals = data.map(d=>d[c]);
    const mean = (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(4);
    const min = Math.min(...vals).toFixed(4);
    const max = Math.max(...vals).toFixed(4);
    html += `<tr><td>${c}</td><td>${mean}</td><td>${min}</td><td>${max}</td></tr>`;
  });
  html += "</tbody></table>";
  return html;
}

function missingHTML(data){
  const cols = Object.keys(data[0]);
  let html = "<table><thead><tr>"+cols.map(c=>`<th>${c}</th>`).join("")+"</tr></thead><tbody>";
  html += "<tr>"+cols.map(c=>{
    const missing = data.filter(d=>d[c]===null||d[c]===undefined||d[c]==="").length;
    return `<td>${missing}</td>`;
  }).join("")+"</tr></tbody></table>";
  return html;
}

document.getElementById("btn-load").addEventListener("click",()=>{
  const trainFile = document.getElementById("train-file").files[0];
  const testFile = document.getElementById("test-file").files[0];
  if(!trainFile || !testFile){ alert("Please select both files."); return; }
  document.getElementById("note-load").innerText = "Loading...";
  parseCSV(trainFile,data=>{ trainData = data;
    parseCSV(testFile,data2=>{ testData = data2;
      mergeData();
      document.getElementById("note-load").innerText = "Load done.";
    });
  });
});
