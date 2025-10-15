let trainRaw = null;
let testRaw = null;
let merged = null;
const papaOptions = { header:true, dynamicTyping:false, skipEmptyLines:true, quoteChar:'"', escapeChar:'"' };

function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    if(!trainFile || !testFile){ alert('Select both train and test CSV'); return; }
    document.getElementById('load-status').innerText='Parsing CSV...';
    
    Papa.parse(trainFile, {...papaOptions, complete:(results)=>{ trainRaw=results.data.map(normalizeRow); tryBuildMerged(); } });
    Papa.parse(testFile, {...papaOptions, complete:(results)=>{ testRaw=results.data.map(normalizeRow); tryBuildMerged(); } });
}

function normalizeRow(row){
    const out={};
    Object.keys(row).forEach(k=>{
        let v=row[k].trim();
        out[k]=v===''?null:(!isNaN(v)?Number(v):v);
    });
    return out;
}

function tryBuildMerged(){
    if(trainRaw && testRaw){
        merged=trainRaw.map(r=>({...r, source:'train'})).concat(testRaw.map(r=>({...r, source:'test'})));
        document.getElementById('run-eda-btn').disabled=false;
        document.getElementById('load-status').innerText=`Loaded train(${trainRaw.length}) & test(${testRaw.length})`;
        populateHotelDropdown();
    }
}

function populateHotelDropdown(){
    const hotels=[...new Set(merged.map(r=>r.hotel_type))];
    const sel=document.getElementById('hotel-dropdown');
    sel.innerHTML='';
    hotels.forEach(h=>{ const o=document.createElement('option'); o.value=h; o.innerText=h; sel.appendChild(o); });
}
