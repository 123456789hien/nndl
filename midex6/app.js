let lineChart, histChart, heatmapChart, predChart;
document.getElementById("load-data-btn").onclick = ()=>{
    const trainFile = document.getElementById("train-csv").files[0];
    const testFile = document.getElementById("test-csv").files[0];
    if(!trainFile || !testFile) { alert("Select both CSVs"); return; }
    parseCSV(trainFile, data=>{
        trainData=fillMissingMonths(data);
        parseCSV(testFile, data2=>{
            testData=fillMissingMonths(data2);
            document.getElementById("notes").innerText=`Loaded train(${trainData.length}) + test(${testData.length})`;
            updateTables();
            updateEDACharts();
            updateHotelDropdown();
        });
    });
};

function updateTables(){
    const mergeDiv=document.getElementById("merge-overview");
    let html="<h3>Merge & Overview Table</h3><table><tr><th>Hotel</th><th>Year</th><th>Month</th><th>Cancellation Rate</th><th>Avg Room Price</th><th>Lead Time</th></tr>";
    trainData.forEach(d=>{
        html+=`<tr>
        <td>${d.hotel}</td>
        <td>${d.year}</td>
        <td>${d.month}</td>
        <td>${d.cancellation_rate!==null?d.cancellation_rate:"NaN"}</td>
        <td>${d.avg_room_price!==null?d.avg_room_price:"NaN"}</td>
        <td>${d.lead_time_avg!==null?d.lead_time_avg:"NaN"}</td>
        </tr>`;
    });
    html+="</table>";
    mergeDiv.innerHTML=html;

    // Missing values
    const missingDiv=document.getElementById("missing-values");
    const missCounts = trainData.reduce((acc,d)=>{
        for(let k of ["cancellation_rate","avg_room_price","lead_time_avg"]){
            if(d[k]===null) acc[k]=(acc[k]||0)+1;
        }
        return acc;
    },{});
    missingDiv.innerHTML=`<h3>Missing Values Table</h3>
    <table><tr><th>Variable</th><th>Missing Count</th></tr>
    <tr><td>Cancellation Rate</td><td>${missCounts.cancellation_rate||0}</td></tr>
    <tr><td>Avg Room Price</td><td>${missCounts.avg_room_price||0}</td></tr>
    <tr><td>Lead Time</td><td>${missCounts.lead_time_avg||0}</td></tr>
    </table>`;

    // Stats
    const statsDiv=document.getElementById("stats-table");
    const numericCols=["cancellation_rate","avg_room_price","lead_time_avg"];
    let statsHTML="<h3>Stats Table</h3><table><tr><th>Variable</th><th>Mean</th><th>Min</th><th>Max</th></tr>";
    numericCols.forEach(col=>{
        const vals=trainData.map(d=>d[col]!==null?d[col]:0);
        statsHTML+=`<tr><td>${col}</td><td>${(vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(2)}</td><td>${Math.min(...vals)}</td><td>${Math.max(...vals)}</td></tr>`;
    });
    statsHTML+="</table>";
    statsDiv.innerHTML=statsHTML;
}

// --- EDA Charts ---
function updateEDACharts(){
    const months=trainData.map(d=>`${d.year}-${d.month}`);
    const cancelRates=trainData.map(d=>d.cancellation_rate??0);
    const prices=trainData.map(d=>d.avg_room_price??0);
    const leadTime=trainData.map(d=>d.lead_time_avg??0);

    // Line chart
    const ctx=document.getElementById("line-cancellation").getContext("2d");
    if(lineChart) lineChart.destroy();
    lineChart=new Chart(ctx,{
        type:"line",
        data:{labels:months,datasets:[{label:"Cancellation Rate",data:cancelRates,borderColor:"#01497c",fill:false}]},
        options:{responsive:true,plugins:{tooltip:{enabled:true}}}
    });

    // Histogram
    const ctx2=document.getElementById("histogram-price").getContext("2d");
    if(histChart) histChart.destroy();
    histChart=new Chart(ctx2,{
        type:"bar",
        data:{labels:months,datasets:[{label:"Avg Room Price",data:prices,backgroundColor:"#01497c"}]},
        options:{responsive:true,plugins:{tooltip:{enabled:true}}}
    });

    // Heatmap (interactive)
    const ctx3=document.getElementById("heatmap-corr").getContext("2d");
    if(heatmapChart) heatmapChart.destroy();
    const vars=["cancellation_rate","avg_room_price","lead_time_avg"];
    const corrData=[];
    for(let i=0;i<vars.length;i++){
        for(let j=0;j<vars.length;j++){
            const xi=trainData.map(d=>d[vars[i]]??0);
            const yj=trainData.map(d=>d[vars[j]]??0);
            const meanX=xi.reduce((a,b)=>a+b,0)/xi.length;
            const meanY=yj.reduce((a,b)=>a+b,0)/yj.length;
            const cov=xi.reduce((a,b,k)=>a+(b-meanX)*(yj[k]-meanY),0)/xi.length;
            const stdX=Math.sqrt(xi.reduce((a,b)=>a+Math.pow(b-meanX,2),0)/xi.length);
            const stdY=Math.sqrt(yj.reduce((a,b)=>a+Math.pow(b-meanY,2),0)/xi.length);
            const corr=stdX&&stdY?cov/(stdX*stdY):0;
            corrData.push({x:i,y:j,v:corr});
        }
    }
    heatmapChart=new Chart(ctx3,{
        type:"matrix",
        data:{datasets:[{label:"Correlation",data:corrData,backgroundColor:ctx=>{const val=ctx.dataset.data[ctx.dataIndex].v;return `rgba(1,73,124,${Math.abs(val)})`;}}]},
        options:{responsive:true,scales:{x:{type:"linear",min:0,max:2,ticks:{callback:i=>vars[i]}},y:{type:"linear",min:0,max:2,ticks:{callback:i=>vars[i]}}},plugins:{tooltip:{callbacks:{label:c=>`Corr: ${c.raw.v.toFixed(2)}`}}}}
    });
}

// Smoothing slider
document.getElementById("smoothing-slider").oninput=e=>{
    const factor=Number(e.target.value);
    const smoothed=trainData.map((d,i,arr)=>{
        const start=Math.max(0,i-factor+1);
        const vals=arr.slice(start,i+1).map(x=>x.cancellation_rate??0);
        return {...d,cancellation_rate:vals.reduce((a,b)=>a+b,0)/vals.length};
    });
    trainData=smoothed;
    updateEDACharts();
};

// Dropdown
function updateHotelDropdown(){
    const sel=document.getElementById("hotel-dropdown");
    sel.innerHTML="";
    const hotels=[...new Set(trainData.map(d=>d.hotel))];
    hotels.forEach(h=> sel.innerHTML+=`<option value="${h}">${h}</option>`);
}

// --- LSTM Training ---
let lstmModel;
document.getElementById("train-model-btn").onclick=async()=>{
    document.getElementById("train-notes").innerText="Status: Running training...";
    const xs=tf.tensor3d(trainData.map(d=>[d.avg_room_price??0,d.lead_time_avg??0,d.cancellation_rate??0]).map(x=>[x]));
    const ys=tf.tensor2d(trainData.map(d=>[d.cancellation_rate??0]));
    lstmModel=tf.sequential();
    lstmModel.add(tf.layers.lstm({units:50,returnSequences:true,inputShape:[1,3]}));
    lstmModel.add(tf.layers.lstm({units:50,returnSequences:true}));
    lstmModel.add(tf.layers.lstm({units:50}));
    lstmModel.add(tf.layers.dense({units:1}));
    lstmModel.compile({loss:"meanSquaredError",optimizer:"adam"});
    await lstmModel.fit(xs,ys,{epochs:50});
    document.getElementById("train-notes").innerText="Status: Done training";
    document.getElementById("download-model-btn").disabled=false;
};
document.getElementById("download-model-btn").onclick=async()=>{
    await lstmModel.save('downloads://hotel_model');
};

// --- Prediction ---
document.getElementById("predict-btn").onclick=async()=>{
    const hotel=document.getElementById("hotel-dropdown").value;
    const year=Number(document.getElementById("predict-year").value);
    const month=Number(document.getElementById("predict-month").value);
    if(!lstmModel){ alert("Upload model first or train"); return; }
    const filtered=trainData.filter(d=>d.hotel===hotel);
    const input=[filtered.find(d=>d.year===year && d.month===month)?.avg_room_price??0,
                 filtered.find(d=>d.year===year && d.month===month)?.lead_time_avg??0,
                 filtered.find(d=>d.year===year && d.month===month)?.cancellation_rate??0];
    const pred=(await lstmModel.predict(tf.tensor3d([input],[1,1,3]))).arraySync()[0][0];

    // Update chart + insight
    const insightBox=document.getElementById("insight-box");
    if(pred>0.6) insightBox.className="high"; else if(pred>0.3) insightBox.className="medium"; else insightBox.className="low";
    insightBox.innerText=`Predicted Cancellation Rate: ${(pred*100).toFixed(2)}%`;

    // Prediction line chart
    const ctx=document.getElementById("line-prediction").getContext("2d");
    if(predChart) predChart.destroy();
    const months=filtered.map(d=>`${d.year}-${d.month}`);
    const cancelRates=filtered.map(d=>d.cancellation_rate??0);
    const datasets=[{label:"Historical",data:cancelRates,borderColor:"#01497c",fill:false},{label:"Predicted",data:cancelRates.map((v,i)=>i===month-1?pred:v),borderColor:"#d62828",fill:false}];
    predChart=new Chart(ctx,{type:"line",data:{labels:months,datasets:datasets},options:{responsive:true,plugins:{tooltip:{enabled:true}}}});
};
