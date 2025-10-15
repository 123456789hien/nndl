let trainRaw=null, testRaw=null, merged=null;
let monthlyData={}, smoothingLevel=1;

const papaOptions={header:true, dynamicTyping:false, skipEmptyLines:true};

function loadData(){
  const trainFile=document.getElementById('train-file').files[0];
  const testFile=document.getElementById('test-file').files[0];
  if(!trainFile||!testFile){ alert('Select both CSVs'); return; }
  setStatus('Parsing CSV files...');
  
  Papa.parse(trainFile,{...papaOptions, complete:(results)=>{
    trainRaw=results.data.map(normalizeRow);
    setStatus(`Train loaded: ${trainRaw.length} rows`);
    tryBuildMerged();
  }});
  
  Papa.parse(testFile,{...papaOptions, complete:(results)=>{
    testRaw=results.data.map(normalizeRow);
    setStatus(`Test loaded: ${testRaw.length} rows`);
    tryBuildMerged();
  }});
}

function normalizeRow(row){
  const out={};
  Object.keys(row).forEach(k=>{
    let v=row[k]; if(typeof v==='string') v=v.trim();
    out[k]=(v===''?null:v);
  });
  return out;
}

function tryBuildMerged(){
  if(trainRaw && testRaw){
    merged=trainRaw.map(r=>({...r,source:'train'})).concat(testRaw.map(r=>({...r,source:'test'})));
    document.getElementById('run-eda-btn').disabled=false;
    setStatus(`Data merged: ${merged.length} rows`);
  }
}

function setStatus(msg){ document.getElementById('load-status').innerText=msg; }

function updateSmoothing(value){
  smoothingLevel=Number(value);
  document.getElementById('smoothing-value').innerText=value;
  if(Object.keys(monthlyData).length>0){
    prepareMonthlyData();
    renderCharts();
  }
}
