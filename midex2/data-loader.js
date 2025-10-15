let trainRaw=null,testRaw=null,merged=null;

function normalizeRow(row){
  const out={};
  Object.keys(row).forEach(k=>{
    let v=row[k];
    if(v===null||v===undefined){ out[k]=null; }
    else if(typeof v==='string'){ out[k]=v.trim(); }
    else{ out[k]=v; }
  });
  return out;
}

function loadData(){
  const trainFile=document.getElementById('train-file').files[0];
  const testFile=document.getElementById('test-file').files[0];
  if(!trainFile || !testFile){ alert('Select both CSVs'); return; }

  Papa.parse(trainFile,{header:true,dynamicTyping:true,skipEmptyLines:true,delimiter:';',
    complete: res=>{ trainRaw=res.data.map(normalizeRow); tryBuildMerged(); }
  });

  Papa.parse(testFile,{header:true,dynamicTyping:true,skipEmptyLines:true,delimiter:';',
    complete: res=>{ testRaw=res.data.map(normalizeRow); tryBuildMerged(); }
  });
}

function tryBuildMerged(){
  if(trainRaw && testRaw){
    merged = trainRaw.map(r=>({...r, source:'train'}))
      .concat(testRaw.map(r=>({...r, source:'test'})));
    document.getElementById('run-eda-btn').disabled=false;
    populateHotelDropdown();
    console.log(`Loaded train(${trainRaw.length}) + test(${testRaw.length})`);
  }
}

function populateHotelDropdown(){
  const hotels=[...new Set(merged.map(r=>r.hotel_type||r.hotel||'Hotel'))];
  const select=document.getElementById('hotel-select');
  select.innerHTML='';
  hotels.forEach(h=>{ const opt=document.createElement('option'); opt.value=h; opt.text=h; select.appendChild(opt); });
}
