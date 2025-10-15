let trainData=[], testData=[], mergedData=[];
let groupKeys=[];
const edaCols=["cancellation_rate","avg_room_price","lead_time_avg"];

const removeZeroRows = data =>
  data.filter(d => d.total_bookings!=0 || d.cancelled_bookings!=0 || d.avg_room_price!=0);

const parseCSV = (file, callback)=>{
  Papa.parse(file,{
    header:true, skipEmptyLines:true, dynamicTyping:true,
    complete: results => callback(results.data)
  });
};

document.getElementById("btn-load").addEventListener("click",()=>{
  const trainFile=document.getElementById("train-file").files[0];
  const testFile=document.getElementById("test-file").files[0];
  if(!trainFile||!testFile) return alert("Please select both CSV files.");
  document.getElementById("note-load").innerText="Running…";

  parseCSV(trainFile,(train)=>{
    parseCSV(testFile,(test)=>{
      trainData=removeZeroRows(train);
      testData=removeZeroRows(test);
      mergedData=trainData.concat(testData);

      displayMergeOverview(mergedData);
      displayMissingValues(mergedData);
      displayStats(mergedData);

      groupKeys=[...new Set(mergedData.map(d=>d.room_type))];
      const selGroup=document.getElementById("sel-group");
      selGroup.innerHTML=groupKeys.map(g=>`<option value="${g}">${g}</option>`).join('');

      document.getElementById("note-load").innerText="Done ✅";
    });
  });
});

function displayMergeOverview(data){
  const preview=data.slice(0,10);
  let html="<table><thead><tr>";
  Object.keys(preview[0]||{}).forEach(k=>html+=`<th>${k}</th>`);
  html+="</tr></thead><tbody>";
  preview.forEach(r=>{
    html+="<tr>";
    Object.values(r).forEach(v=>html+=`<td>${v}</td>`);
    html+="</tr>";
  });
  html+="</tbody></table>";
  document.getElementById("merge-overview").innerHTML=html;
}

function displayMissingValues(data){
  const keys=Object.keys(data[0]||{});
  let html="<table><thead><tr><th>Column</th><th>Missing</th></tr></thead><tbody>";
  keys.forEach(k=>{
    const miss=data.filter(d=>d[k]==null||d[k]==="").length;
    html+=`<tr><td>${k}</td><td>${miss}</td></tr>`;
  });
  html+="</tbody></table>";
  document.getElementById("missing-table").innerHTML=html;
}

function displayStats(data){
  let html="<table><thead><tr><th>var</th><th>mean</th><th>min</th><th>max</th></tr></thead><tbody>";
  edaCols.forEach(k=>{
    const vals=data.map(d=>d[k]).filter(v=>typeof v==="number");
    if(vals.length){
      const mean=(vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(4);
      const min=Math.min(...vals).toFixed(4);
      const max=Math.max(...vals).toFixed(4);
      html+=`<tr><td>${k}</td><td>${mean}</td><td>${min}</td><td>${max}</td></tr>`;
    }
  });
  html+="</tbody></table>";
  document.getElementById("stats-table").innerHTML=html;
}
