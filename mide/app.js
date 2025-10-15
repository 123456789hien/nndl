// ------------------ EDA ------------------
function runEDA(){
    if(!merged) return;
    renderPreview(merged.slice(0,8));
    renderMissing();
    renderStatsSummary();
    renderCharts();
    renderCorrelationHeatmap();
}

function renderPreview(rows){
    const container=document.getElementById('head-preview');
    const cols=Object.keys(rows[0]);
    let html='<table><thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
    rows.forEach(r=>{ html+='<tr>'+cols.map(c=>`<td>${r[c]}</td>`).join('')+'</tr>'; });
    html+='</tbody></table>'; container.innerHTML=html;
}

function renderMissing(){
    const cols=Object.keys(merged[0]);
    const missing={};
    cols.forEach(c=>{
        const count=merged.filter(r=>r[c]==null).length;
        missing[c]=+(count/merged.length*100).toFixed(2);
    });
    const chartEl=document.getElementById('missing-chart'); chartEl.innerHTML='';
    tfvis.render.barchart({dom:chartEl},Object.entries(missing).map(([k,v])=>({index:k,value:v})),{xLabel:'Column',yLabel:'Missing %'});
}

function renderStatsSummary(){
    const numericCols=['lead_time','avg_room_price','status'];
    const summary={};
    numericCols.forEach(c=>{
        const vals=merged.map(r=>r[c]).filter(v=>v!=null);
        summary[c]={mean:(vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(2), min:Math.min(...vals), max:Math.max(...vals)};
    });
    document.getElementById('stats-summary').innerHTML=`<pre>${JSON.stringify(summary,null,2)}</pre>`;
}

function renderCharts(){
    // Cancellation Rate over Time
    const monthly={};
    merged.forEach(r=>{
        const key=`${r.year}-${r.month}`;
        if(!monthly[key]) monthly[key]={total:0,cancelled:0};
        monthly[key].total++;
        if(r.status===1) monthly[key].cancelled++;
    });
    const labels=Object.keys(monthly).sort();
    const data=labels.map(k=>monthly[k].cancelled/monthly[k].total);
    const ctx=document.getElementById('charts-container').appendChild(document.createElement('canvas')).getContext('2d');
    new Chart(ctx,{type:'line',data:{labels,datasets:[{label:'Cancellation Rate',data,borderColor:'blue',fill:false}]},options:{}});

    // Histogram of avg_room_price
    const prices=merged.map(r=>r.avg_room_price).filter(v=>v!=null);
    const bins=20;
    const minPrice=Math.min(...prices), maxPrice=Math.max(...prices);
    const step=(maxPrice-minPrice)/bins;
    const counts=new Array(bins).fill(0);
    prices.forEach(p=>{ let idx=Math.min(bins-1,Math.floor((p-minPrice)/step)); counts[idx]++; });
    const histData=counts.map((c,i)=>({index:`${(minPrice+i*step).toFixed(0)}-${(minPrice+(i+1)*step).toFixed(0)}`, value:c}));
    const ctx2=document.getElementById('charts-container').appendChild(document.createElement('canvas')).getContext('2d');
    new Chart(ctx2,{type:'bar',data:{labels:histData.map(d=>d.index), datasets:[{label:'Room Price Distribution', data:histData.map(d=>d.value), backgroundColor:'orange'}]},options:{}});
}

// ------------------ Correlation Heatmap ------------------
function renderCorrelationHeatmap(){
    const cols=['cancellation_rate','avg_room_price','lead_time'];
    const data=[];
    for(let i=0;i<cols.length;i++){
        data[i]=[];
        for(let j=0;j<cols.length;j++){
            const xi=merged.map(r=>r[cols[i]]).filter(v=>v!=null);
            const xj=merged.map(r=>r[cols[j]]).filter(v=>v!=null);
            const minLen=Math.min(xi.length,xj.length);
            const corr=pearson(xi.slice(0,minLen),xj.slice(0,minLen));
            data[i][j]=corr;
        }
    }
    tfvis.render.heatmap({name:'Correlation Heatmap', tab:'Charts'}, {values:data, xLabels:cols, yLabels:cols});
}

function pearson(x,y){
    const n=x.length;
    const meanX=x.reduce((a,b)=>a+b,0)/n;
    const meanY=y.reduce((a,b)=>a+b,0)/n;
    let num=0,denX=0,denY=0;
    for(let i=0;i<n;i++){ num+=(x[i]-meanX)*(y[i]-meanY); denX+=(x[i]-meanX)**2; denY+=(y[i]-meanY)**2; }
    return num/Math.sqrt(denX*denY);
}

// ------------------ TF.js LSTM Model ------------------
let model=null;
async function loadModel(){ model=await tf.loadLayersModel('hotel_model.json'); }
loadModel();

function predictNextMonth(){
    if(!model){ alert('Model not loaded'); return; }
    const hotel=document.getElementById('hotel-dropdown').value;
    const year=Number(document.getElementById('input-year').value);
    const month=Number(document.getElementById('input-month').value);

    const hotelData=merged.filter(r=>r.hotel_type===hotel);
    const seq=hotelData.slice(-12).map(r=>[r.lead_time,r.avg_room_price]);
    if(seq.length<12){ alert('Not enough history'); return; }

    const input=tf.tensor([seq]);
    let pred=model.predict(input).dataSync()[0];

    const box=document.getElementById('insight-box');
    if(pred>0.7){ box.innerText="High risk — offer early payment"; box.style.backgroundColor='red'; }
    else if(pred<0.4){ box.innerText="Low risk — consider price increase"; box.style.backgroundColor='green'; }
    else{ box.innerText="Medium risk — flexible policy"; box.style.backgroundColor='yellow'; }

    document.getElementById('prediction-result').innerText=`Predicted Cancellation Rate: ${(pred*100).toFixed(1)}%`;

    const ctx=document.getElementById('prediction-chart').getContext('2d');
    new Chart(ctx,{
        type:'line',
        data:{
            labels: hotelData.slice(-12).map(r=>`${r.year}-${r.month}`).concat(`${year}-${month}`),
            datasets:[
                {label:'Historical Cancellation Rate', data: hotelData.slice(-12).map(r=>r.status), borderColor:'blue', fill:false},
                {label:'Predicted', data:[...Array(12).fill(null),pred], borderColor:'red', fill:false, pointRadius:6}
            ]
        },
        options:{}
    });
}
