let model=null;
const smoothingSlider=document.getElementById('smoothingSlider');

document.getElementById('trainModelBtn').addEventListener('click',async()=>{
  if(trainData.length===0){alert('Load data first'); return;}
  document.getElementById('trainStatus').innerText='Training...';

  const hotels=[...new Set(trainData.map(d=>d.hotel))];
  const sequences=[]; const labels=[];

  hotels.forEach(hotel=>{
    let hotelRows=trainData.filter(d=>d.hotel===hotel).sort((a,b)=>a.monthIndex-b.monthIndex);
    if(smoothingSlider.value>0){
      for(let i=1;i<hotelRows.length;i++){
        hotelRows[i].cancellation_rate=(hotelRows[i].cancellation_rate*(1/(smoothingSlider.value+1)))+hotelRows[i-1].cancellation_rate*(smoothingSlider.value/(smoothingSlider.value+1));
      }
    }
    for(let i=12;i<hotelRows.length;i++){
      sequences.push(hotelRows.slice(i-12,i).map(r=>[r.cancellation_rate,r.avg_room_price,r.lead_time_avg]));
      labels.push([hotelRows[i].cancellation_rate]);
    }
  });

  const X=tf.tensor3d(sequences);
  const Y=tf.tensor2d(labels);

  model=tf.sequential();
  model.add(tf.layers.lstm({units:50,returnSequences:true,inputShape:[12,3]}));
  model.add(tf.layers.lstm({units:50,returnSequences:true}));
  model.add(tf.layers.lstm({units:50}));
  model.add(tf.layers.dense({units:1}));
  model.compile({optimizer:'adam',loss:'meanSquaredError',metrics:['mse']});

  await model.fit(X,Y,{epochs:50,batchSize:32});
  document.getElementById('trainStatus').innerText='Training complete!';
});

document.getElementById('downloadModel').addEventListener('click',async()=>{
  if(model) await model.save('downloads://hotel_model');
});

document.getElementById('predictBtn').addEventListener('click',async()=>{
  const hotel=document.getElementById('hotelDropdown').value;
  const year=parseInt(document.getElementById('inputYear').value);
  const month=parseInt(document.getElementById('inputMonth').value);
  if(!model){alert('No model loaded'); return;}

  let hotelRows=trainData.filter(d=>d.hotel===hotel).sort((a,b)=>a.monthIndex-b.monthIndex);
  if(smoothingSlider.value>0){
    for(let i=1;i<hotelRows.length;i++){
      hotelRows[i].cancellation_rate=(hotelRows[i].cancellation_rate*(1/(smoothingSlider.value+1)))+hotelRows[i-1].cancellation_rate*(smoothingSlider.value/(smoothingSlider.value+1));
    }
  }

  const last12=hotelRows.slice(-12).map(r=>[r.cancellation_rate,r.avg_room_price,r.lead_time_avg]);
  const input=tf.tensor3d([last12]);
  const pred=model.predict(input);
  const predVal=pred.dataSync()[0];

  document.getElementById('predictionResult').innerText=`Predicted Cancellation Rate: ${(predVal*100).toFixed(2)}%`;

  const insightBox=document.getElementById('insightBox');
  if(predVal>0.3) insightBox.className='high';
  else if(predVal>0.1) insightBox.className='medium';
  else insightBox.className='low';
  insightBox.innerText=insightBox.className==='high'? 'High Risk — Early Payment':
                        insightBox.className==='medium'? 'Medium Risk — Flexible Policy':
                        'Low Risk — Price Increase Consideration';

  const ctx=document.getElementById('predictionChart').getContext('2d');
  const chartData=hotelRows.map(r=>({x:r.monthIndex,y:r.cancellation_rate}));
  chartData.push({x:hotelRows[hotelRows.length-1].monthIndex+1,y:predVal});
  new Chart(ctx,{type:'line',data:{datasets:[{label:'Cancellation Rate',data:chartData,borderColor:'blue',tension:0.2,fill:false,pointBackgroundColor:chartData.map((p,i)=>i===chartData.length-1?'red':'blue')}]},options:{plugins:{tooltip:{enabled:true}}}});
});
