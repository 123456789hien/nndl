let trainRaw = null, testRaw = null, merged = null;

// Helper: Convert empty string to null
function normalizeRow(row){
  const out={};
  Object.keys(row).forEach(k=>{
    let v=row[k]?.trim();
    out[k]=(v==='')?null:v;
  });
  return out;
}

function loadData(){
  const trainFile=document.getElementById('train-file').files[0];
  const testFile=document.getElementById('test-file').files[0];
  if(!trainFile || !testFile){ alert('Select both CSVs'); return; }

  Papa.parse(trainFile, { header:true, dynamicTyping:true, skipEmptyLines:true,
    complete: res => { trainRaw=res.data.map(normalizeRow); tryBuildMerged(); } });
  Papa.parse(testFile, { header:true, dynamicTyping:true, skipEmptyLines:true,
    complete: res => { testRaw=res.data.map(normalizeRow); tryBuildMerged(); } });
}

function tryBuildMerged(){
  if(trainRaw && testRaw){
    merged=trainRaw.map(r=>({...r, source:'train'})).concat(testRaw.map(r=>({...r, source:'test'})));
    document.getElementById('run-eda-btn').disabled=false;
    console.log(`Loaded train(${trainRaw.length}) + test(${testRaw.length})`);
  }
}
