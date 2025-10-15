let chartCancel, chartPrice, chartHeatmap, chartPredict;
let model, scaler = {min:0,max:0};
let currentGroup = "room_type";
let smoothingWindow = 1;

document.getElementById("smoothing-slider").addEventListener("input", e=>{
    smoothingWindow = parseInt(e.target.value);
    document.getElementById("smoothing-val").innerText = smoothingWindow;
    renderEDA();
});
document.getElementById("group-key").addEventListener("change", e=>{
    currentGroup = e.target.value;
    renderEDA();
});

// EDA Charts
function renderEDA(){
    if(!mergedData.length) return;
    const group = currentGroup;
    const lineData = aggregateByMonth(mergedData, "cancellation_rate", group);
    const priceData = aggregateByMonth(mergedData, "avg_room_price", group);

    // Cancellation line chart
    if(chartCancel) chartCancel.destroy();
    chartCancel = new Chart(document.getElementById("chart-cancel"), {
        type:"line",
        data:{labels:lineData.labels, datasets:[{label:"Cancellation Rate", data:lineData.data, borderColor:"#0288d1", tension:0.3}]},
        options:{plugins:{tooltip:{enabled:true}}}
    });

    // Avg price histogram
    if(chartPrice) chartPrice.destroy();
    chartPrice = new Chart(document.getElementById("chart-price"), {
        type:"bar",
        data:{labels:priceData.labels, datasets:[{label:"Avg Price", data:priceData.data, backgroundColor:"#02548a"}]},
        options:{plugins:{tooltip:{enabled:true}}}
    });

    // Correlation heatmap
    if(chartHeatmap) chartHeatmap.destroy();
    const corrMatrix = computeCorrMatrix(mergedData);
    chartHeatmap = new Chart(document.getElementById("chart-heatmap"), {
        type:"matrix",
        data:{datasets:[{label:"Correlation", data:corrMatrix, backgroundColor:c=>`rgba(2,84,138,${c.v})`}]},
        options:{plugins:{tooltip:{enabled:true}}}
    });
}

function aggregateByMonth(data,col,groupKey){
    const grouped = {};
    data.forEach(d=>{
        const key = d[groupKey]+"-"+d.year+"-"+d.month;
        if(!grouped[key]) grouped[key]=[];
        grouped[key].push(d[col]);
    });
    const labels = Object.keys(grouped).sort();
    const dataArr = labels.map(k=>{
        const arr = grouped[k];
        const avg = arr.reduce((a,b)=>a+b,0)/arr.length;
        return smooth(avg);
    });
    return {labels,data:dataArr};
}

function smooth(value){ return value; } // placeholder, can apply rolling mean

function computeCorrMatrix(data){
    const keys = ["total_bookings","cancelled_bookings","cancellation_rate","avg_room_price","lead_time_avg"];
    let matrix = [];
    for(let i=0;i<keys.length;i++){
        for(let j=0;j<keys.length;j++){
            let vi = data.map(d=>d[keys[i]]), vj = data.map(d=>d[keys[j]]);
            let corr = correlation(vi,vj);
            matrix.push({x:i,y:j,v:corr});
        }
    }
    return matrix;
}

function correlation(x,y){
    const n = x.length;
    const meanX = x.reduce((a,b)=>a+b,0)/n;
    const meanY = y.reduce((a,b)=>a+b,0)/n;
    let num=0, denX=0, denY=0;
    for(let i=0;i<n;i++){
        num += (x[i]-meanX)*(y[i]-meanY);
        denX += (x[i]-meanX)**2;
        denY += (y[i]-meanY)**2;
    }
    return num/Math.sqrt(denX*denY);
}

// LSTM Train
document.getElementById("btn-train").addEventListener("click", async ()=>{
    const epochs = parseInt(document.getElementById("input-epochs").value);
    const batch = parseInt(document.getElementById("input-batch").value);
    document.getElementById("train-status").innerText="Running…";
    await trainModel(epochs,batch);
    document.getElementById("train-status").innerText="Done ✅";
    document.getElementById("btn-download").disabled=false;
});

async function trainModel(epochs,batch){
    const xs = tf.tensor2d([0,1,2,3,4,5,6,7,8,9],[10,1]); // placeholder
    const ys = tf.tensor2d([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[10,1]);
    model = tf.sequential();
    model.add(tf.layers.dense({units:50,inputShape:[1],activation:"relu"}));
    model.add(tf.layers.dense({units:50,activation:"relu"}));
    model.add(tf.layers.dense({units:50,activation:"relu"}));
    model.add(tf.layers.dense({units:1}));
    model.compile({optimizer:"adam",loss:"meanSquaredError"});
    await model.fit(xs,ys,{epochs,batchSize:batch});
}

// Download
document.getElementById("btn-download").addEventListener("click", async ()=>{
    if(!model) return alert("Model not trained yet");
    await model.save("downloads://nextstay_model");
});

// Predict
document.getElementById("btn-predict").addEventListener("click", async ()=>{
    const group = document.getElementById("sel-group").value;
    const year = parseInt(document.getElementById("inp-year").value);
    const month = parseInt(document.getElementById("inp-month").value);
    if(!model) return alert("Model not loaded or trained");

    // placeholder prediction
    const input = tf.tensor2d([month],[1,1]);
    const pred = (await model.predict(input).array())[0][0];
    const scaledPred = pred; // inverse scale
    document.getElementById("predict-result").innerText = "Prediction: "+scaledPred.toFixed(2);

    // Insight
    const insightBox = document.getElementById("insight");
    if(scaledPred>0.6) insightBox.className="insight high";
    else if(scaledPred>0.3) insightBox.className="insight medium";
    else insightBox.className="insight low";
    insightBox.innerText = `Insight: ${insightBox.className.split(" ")[1].toUpperCase()}`;
    
    renderPredictChart(scaledPred);
});

function renderPredictChart(pred){
    if(chartPredict) chartPredict.destroy();
    chartPredict = new Chart(document.getElementById("chart-predict"),{
        type:"line",
        data:{
            labels:["Month 1","Month 2","Month 3","Next Month"],
            datasets:[{
                label:"Bookings",
                data:[50,60,70,pred],
                borderColor:"#0288d1",
                fill:false,
                tension:0.3
            }]
        },
        options:{plugins:{tooltip:{enabled:true}}}
    });
}
