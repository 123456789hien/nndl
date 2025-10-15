let trainData=[], testData=[], mergedData=[];
let stats = {};

document.getElementById("btn-load").addEventListener("click",()=>{
  const trainFile = document.getElementById("train-file").files[0];
  const testFile = document.getElementById("test-file").files[0];
  if(!trainFile || !testFile){alert("Please select both train and test CSV."); return;}
  document.getElementById("note-load").textContent="Running...";

  Papa.parse(trainFile,{header:true, dynamicTyping:true, skipEmptyLines:true, complete: (res1)=>{
    Papa.parse(testFile,{header:true, dynamicTyping:true, skipEmptyLines:true, complete: (res2)=>{
      trainData=res1.data;
      testData=res2.data;
      mergeAndOverview();
      calcStats();
      document.getElementById("note-load").textContent="Load done.";
    }});
  }});
});

function mergeAndOverview(){
  mergedData=[...trainData,...testData].map(d=>{
    return {
      group:d.group,
      year:d.year,
      month:d.month,
      total_bookings:d.total_bookings,
      cancelled_bookings:d.cancelled_bookings,
      cancellation_rate:d.cancellation_rate,
      avg_room_price:d.avg_room_price,
      lead_time_avg:d.lead_time_avg,
      month_index:d.month_index
    };
  });
  // show preview 10 rows
  let html="<table><thead><tr>";
  Object.keys(mergedData[0]).forEach(k=>html+=`<th>${k}</th>`);
  html+="</tr></thead><tbody>";
  mergedData.slice(0,10).forEach(r=>{
    html+="<tr>"+Object.values(r).map(v=>`<td>${v}</td>`).join("")+"</tr>";
  });
  html+="</tbody></table>";
  document.getElementById("merge-overview").innerHTML=html;

  // missing
  let cols=Object.keys(mergedData[0]);
  let missHTML="<table><thead><tr><th>Column</th><th>Missing</th></tr></thead><tbody>";
  cols.forEach(c=>{
    let miss=mergedData.filter(r=>r[c]==null||r[c]==undefined||r[c]==0).length;
    missHTML+=`<tr><td>${c}</td><td>${miss}</td></tr>`;
  });
  missHTML+="</tbody></table>";
  document.getElementById("missing-table").innerHTML=missHTML;
}

function calcStats(){
  let cols=["cancellation_rate","avg_room_price","lead_time_avg"];
  stats={};
  cols.forEach(c=>{
    let arr=mergedData.map(r=>r[c]).filter(v=>v!=0 && v!=null);
    stats[c]={mean:(arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(4),
              min:Math.min(...arr).toFixed(4),
              max:Math.max(...arr).toFixed(4)};
  });
  let html="<table><thead><tr><th>var</th><th>mean</th><th>min</th><th>max</th></tr></thead><tbody>";
  for(let k in stats){
    html+=`<tr><td>${k}</td><td>${stats[k].mean}</td><td>${stats[k].min}</td><td>${stats[k].max}</td></tr>`;
  }
  html+="</tbody></table>";
  document.getElementById("stats-table").innerHTML=html;
}
