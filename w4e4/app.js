// app.js
import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class StockPredictionApp {
    constructor(){
        this.dataLoader=new DataLoader();
        this.model=null;
        this.currentPredictions=null;
        this.accuracyChart=null;
        this.isTraining=false;
        this.initializeEventListeners();
        this.autoLoadCSV();
    }

    async autoLoadCSV(){
        try{
            document.getElementById('status').textContent='Loading default CSV...';
            await this.dataLoader.loadCSVfromURL('data/sp500_top10_xcorr_recent3y.csv');
            this.dataLoader.createSequences();
            document.getElementById('trainBtn').disabled=false;
            document.getElementById('status').textContent='Default CSV loaded. Click Train Model.';
        }catch(err){
            console.warn('Default CSV not found. Please upload manually.');
            document.getElementById('status').textContent='Upload CSV file to begin.';
        }
    }

    initializeEventListeners(){
        const fileInput=document.getElementById('csvFile');
        const trainBtn=document.getElementById('trainBtn');
        const predictBtn=document.getElementById('predictBtn');

        fileInput.addEventListener('change',e=>this.handleFileUpload(e));
        trainBtn.addEventListener('click',()=>this.trainModel());
        predictBtn.addEventListener('click',()=>this.runPrediction());
    }

    async handleFileUpload(event){
        const file=event.target.files[0];
        if(!file) return;
        try{
            document.getElementById('status').textContent='Loading CSV...';
            await this.dataLoader.loadCSV(file);
            this.dataLoader.createSequences();
            document.getElementById('trainBtn').disabled=false;
            document.getElementById('status').textContent='Data loaded. Click Train Model.';
        }catch(err){console.error(err);}
    }

    async trainModel(){
        if(this.isTraining) return;
        this.isTraining=true;
        document.getElementById('trainBtn').disabled=true;
        document.getElementById('predictBtn').disabled=true;
        try{
            const {X_train,y_train,X_test,y_test,symbols}=this.dataLoader;
            this.model=new GRUModel([X_train.shape[1],X_train.shape[2]],symbols.length*3);
            document.getElementById('status').textContent='Training model...';
            await this.model.train(X_train,y_train,X_test,y_test,80,16);
            document.getElementById('predictBtn').disabled=false;
            document.getElementById('status').textContent='Training completed. Click Run Prediction.';
        }catch(err){console.error(err);}finally{this.isTraining=false;}
    }

    async runPrediction(){
        if(!this.model){alert('Train first'); return;}
        try{
            document.getElementById('status').textContent='Running predictions...';
            const {X_test,y_test,symbols}=this.dataLoader;
            const preds=await this.model.predict(X_test);
            const evals=this.model.evaluatePerStock(y_test,preds,symbols);
            this.currentPredictions=evals;
            this.visualizeResults(evals,symbols);
            document.getElementById('status').textContent='Prediction completed.';
            preds.dispose();
        }catch(err){console.error(err);}
    }

    visualizeResults(evaluation,symbols){
        this.createAccuracyChart(evaluation.stockAccuracies,symbols);
        this.createTimelineCharts(evaluation.stockPredictions,symbols);
    }

    createAccuracyChart(accuracies,symbols){
        const ctx=document.getElementById('accuracyChart').getContext('2d');
        const sortedEntries=Object.entries(accuracies).sort(([,a],[,b])=>b-a);
        const sortedSymbols=sortedEntries.map(([s])=>s);
        const sortedAcc=sortedEntries.map(([,a])=>a*100);

        if(this.accuracyChart) this.accuracyChart.destroy();

        this.accuracyChart=new Chart(ctx,{
            type:'bar',
            data:{labels:sortedSymbols,datasets:[{
                label:'Prediction Accuracy (%)',data:sortedAcc,
                backgroundColor:sortedAcc.map(acc=>acc>60?'rgba(75,192,192,0.8)':acc>50?'rgba(255,205,86,0.8)':'rgba(255,99,132,0.8)'),
                borderColor:sortedAcc.map(acc=>acc>60?'rgb(75,192,192)':acc>50?'rgb(255,205,86)':'rgb(255,99,132)'),borderWidth:1
            }]},
            options:{indexAxis:'y',scales:{x:{beginAtZero:true,max:100,title:{display:true,text:'Accuracy (%)'}}},
                     plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>`Accuracy: ${ctx.raw.toFixed(2)}%`}}}}
        });
    }

    createTimelineCharts(predictions,symbols){
        const container=document.getElementById('timelineContainer');
        container.innerHTML='';
        const topStocks=Object.keys(predictions).slice(0,3);
        topStocks.forEach(sym=>{
            const stockPred=predictions[sym];
            const chartContainer=document.createElement('div');
            chartContainer.className='stock-chart';
            chartContainer.innerHTML=`<h4>${sym} Prediction Timeline</h4><canvas id="timeline-${sym}"></canvas>`;
            container.appendChild(chartContainer);

            const ctx=document.getElementById(`timeline-${sym}`).getContext('2d');
            const sampleSize=Math.min(50,stockPred.length);
            const sampleData=stockPred.slice(0,sampleSize);
            const correctData=sampleData.map(p=>p.correct?1:0);
            const labels=sampleData.map((_,i)=>`Pred ${i+1}`);

            new Chart(ctx,{type:'line',data:{labels,datasets:[{
                label:'Correct Predictions',data:correctData,
                borderColor:'rgb(75,192,192)',backgroundColor:'rgba(75,192,192,0.2)',
                fill:true,tension:0.4,
                pointBackgroundColor:sampleData.map(p=>p.correct?'rgb(75,192,192)':'rgb(255,99,132)')
            }]},options:{scales:{y:{min:0,max:1,ticks:{callback:v=>v===1?'Correct':v===0?'Wrong':''}}},plugins:{tooltip:{callbacks:{label:ctx=>{const p=sampleData[ctx.dataIndex];return `Prediction: ${p.pred===1?'Up':'Down'}, Actual: ${p.true===1?'Up':'Down'}`;}}}}}});
        });
    }

    dispose(){
        if(this.dataLoader) this.dataLoader.dispose();
        if(this.model) this.model.dispose();
        if(this.accuracyChart) this.accuracyChart.destroy();
    }
}

document.addEventListener('DOMContentLoaded',()=>new StockPredictionApp());
