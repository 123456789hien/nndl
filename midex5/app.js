let model, scaler={min:0,max:1}, currentSmoothing=1;
let lineChart,histChart,heatmapChart,predictChart;

// --- EDA ---
function generateEDA(data){
    const keys=["hotel","year","month","cancellation_rate","avg_room_price","lead_time_avg"];
    let html="<h3>Merge & Overview Table</h3><table border='1'><tr>";
    keys.forEach(k=>html+=`<th>${k}</th>`); html+="</tr>";
    data.slice(0,10).forEach(row=>{
        html+="<tr>";
        keys.forEach(k=>html+=`<td>${row[k]}</td>`);
        html+="</tr>";
    });
    html+="</table>";
    document.getElementById('merge-overview').innerHTML=html;

    // Missing Values
    let missing={}; keys.forEach(k=>missing[k]=data.filter(d=>d[k]===null||d[k]===undefined).length);
    let htmlMissing="<h3>Missing Values</h3><table border='1'><tr>";
    keys.forEach(k=>htmlMissing+=`<th>${k}</th>`); htmlMissing+="</tr><tr>";
    keys.forEach(k=>htmlMissing+=`<td>${missing[k]}</td>`); htmlMissing+="</tr></table>";
    document.getElementById('missing-values').innerHTML=htmlMissing;

    // Stats Table
    let numeric=["cancellation_rate","avg_room_price","lead_time_avg"];
    let statsHtml="<h3>Stats Table</h3><table border='1'><tr><th>Feature</th><th>Mean</th><th>Min</th><th>Max</th></tr>";
    numeric.forEach(n=>{
        const vals=data.map(d=>d[n]||0);
        statsHtml+=`<tr><td>${n}</td><td>${(vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(2)}</td><td>${Math.min(...vals)}</td><td>${Math.max(...vals)}</td></tr>`;
    });
    statsHtml+="</table>";
    document.getElementById('stats-table').innerHTML=statsHtml;

    updateLineChart(data);
    updateHistogram(data);
    updateHeatmap(data);

    // Hotel dropdown
    const hotelDropdown=document.getElementById('hotel-dropdown');
    hotelDropdown.innerHTML="";
    [...new Set(data.map(d=>d.hotel))].forEach(h=>{
        let opt=document.createElement('option'); opt.value=h; opt.text=h;
        hotelDropdown.appendChild(opt);
    });
}

// --- Smoothing Slider ---
document.getElementById('smoothing-slider').addEventListener('input',(e)=>{
    currentSmoothing=Number(e.target.value);
    document.getElementById('smoothing-value').innerText=currentSmoothing;
});

// --- Cancellation Rate Line Chart ---
function updateLineChart(data){
    const ctx=document.getElementById('cancellation-line-chart').getContext('2d');
    const sorted=data.sort((a,b)=>a.year*12+a.month - (b.year*12+b.month));
    const yData=sorted.map(d=>d.cancellation_rate);
    const xLabels=sorted.map(d=>`${d.year}-${d.month}`);
    if(lineChart) lineChart.destroy();
    lineChart=new Chart(ctx,{
        type:'line',
        data:{labels:xLabels,datasets:[{label:'Cancellation Rate',data:yData,borderColor:'#034f84',fill:true}]},
        options:{responsive:true,plugins:{tooltip:{enabled:true}}}
    });
}

// --- Histogram Avg Room Price ---
function updateHistogram(data){
    const ctx=document.getElementById('avg-price-histogram').getContext('2d');
    const yData = data.map(d=>d.avg_room_price);
    if(histChart) histChart.destroy();
    histChart=new Chart(ctx,{
        type:'bar',
        data:{labels:yData.map((v,i)=>i+1),datasets:[{label:'Avg Room Price',data:yData,backgroundColor:'#036280'}]},
        options:{responsive:true,plugins:{tooltip:{enabled:true}}}
    });
}

// --- Correlation Heatmap ---
function updateHeatmap(data){
    const ctx=document.getElementById('correlation-heatmap').getContext('2d');
    const numeric=["cancellation_rate","avg_room_price","lead_time_avg"];
    function corr(x,y){
        const mx=x.reduce((a,b)=>a+b,0)/x.length;
        const my=y.reduce((a,b)=>a+b,0)/y.length;
        const num=x.map((v,i)=> (v-mx)*(y[i]-my)).reduce((a,b)=>a+b,0);
        const den=Math.sqrt(x.map(v=>Math.pow(v-mx,2)).reduce((a,b)=>a+b,0)*y.map(v=>Math.pow(v-my,2)).reduce((a,b)=>a+b,0));
        return den===0?0:num/den;
    }
    const corrMatrix=numeric.map(xk=>numeric.map(yk=>corr(data.map(d=>d[xk]||0),data.map(d=>d[yk]||0))));
    if(heatmapChart) heatmapChart.destroy();
    heatmapChart=new Chart(ctx,{
        type:'matrix',
        data:{
            datasets:[{
                label:'Correlation Heatmap',
                data:corrMatrix.flatMap((row,i)=>row.map((v,j)=>({x:j,y:i,v}))),
                backgroundColor:ctx=>{return ctx.dataset.data[ctx.dataIndex].v>0.5?'#034f84':'#88cce3';}
            }]
        },
        options:{plugins:{tooltip:{enabled:true}}}
    });
}

// --- Train Model ---
document.getElementById('train-btn').addEventListener('click', async ()=>{
    document.getElementById('train-status').innerText="Training...";
    model=tf.sequential();
    model.add(tf.layers.lstm({units:50,inputShape:[1,3],returnSequences:true}));
    model.add(tf.layers.lstm({units:50,returnSequences:true}));
    model.add(tf.layers.lstm({units:50}));
    model.add(tf.layers.dense({units:1,activation:'sigmoid'}));
    model.compile({loss:'meanSquaredError',optimizer:'adam'});

    // Mock scaled training for demo
    const xTrain=tf.randomNormal([50,1,3]);
    const yTrain=tf.randomNormal([50,1]);
    await model.fit(xTrain,yTrain,{epochs:50});
    document.getElementById('train-status').innerText="Training done!";
    document.getElementById('download-model-btn').disabled=false;
});

// --- Download Model ---
document.getElementById('download-model-btn').addEventListener('click',async()=>{
    if(!model) return;
    await model.save('downloads://hotel_model');
});

// --- Predict ---
document.getElementById('predict-btn').addEventListener('click',async()=>{
    const hotel=document.getElementById('hotel-dropdown').value;
    const year=Number(document.getElementById('predict-year').value);
    const month=Number(document.getElementById('predict-month').value);
    if(!model){alert("Upload trained model first!"); return;}
    const input=tf.tensor2d([[0,0,month]]);
    const pred=(await model.predict(input).data())[0];
    const insightBox=document.getElementById('insight-box');
    if(pred>0.7){ insightBox.innerText="High Risk"; insightBox.style.backgroundColor='#f44336';}
    else if(pred>0.4){ insightBox.innerText="Medium Risk"; insightBox.style.backgroundColor='#ffeb3b';}
    else {insightBox.innerText="Low Risk"; insightBox.style.backgroundColor='#4caf50';}
});
