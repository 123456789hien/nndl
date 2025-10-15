let trainData=[], testData=[], hotelList=[];

function normalizeRow(row){
  const out={};
  Object.keys(row).forEach(k=>{
    const v=row[k];
    if(v===null||v===undefined) out[k]=null;
    else out[k]=typeof v==='string'?v.toString().trim():v;
  });
  return out;
}

function fillMissingMonths(data, hotelList){
  let filled=[];
  hotelList.forEach(hotel=>{
    const hotelRows=data.filter(d=>d.hotel===hotel);
    let monthIndex=0;
    for(let y=2018;y<=2022;y++){
      for(let m=1;m<=12;m++){
        let row=hotelRows.find(r=>parseInt(r.year)===y && parseInt(r.month)===m);
        if(!row){
          filled.push({hotel,year:y,month:m,monthIndex, total_bookings:0, cancelled_bookings:0, cancellation_rate:0, avg_room_price:0, lead_time_avg:0});
        }else{
          row.monthIndex=monthIndex;
          filled.push(row);
        }
        monthIndex++;
      }
    }
  });
  return filled;
}

document.getElementById('loadData').addEventListener('click',()=>{
  const trainFile=document.getElementById('trainCSV').files[0];
  const testFile=document.getElementById('testCSV').files[0];
  if(!trainFile||!testFile){alert('Select both train & test CSV');return;}

  Papa.parse(trainFile,{header:true,skipEmptyLines:true,complete:(res)=>{
    trainData=res.data.map(normalizeRow);
    hotelList=[...new Set(trainData.map(d=>d.hotel))];
    trainData=fillMissingMonths(trainData,hotelList);
    document.getElementById('loadStatus').innerText=`Loaded train(${trainData.length})`;
    updateHotelDropdown();
    renderMergeOverview();
    renderEDACharts();
  }});

  Papa.parse(testFile,{header:true,skipEmptyLines:true,complete:(res)=>{
    testData=res.data.map(normalizeRow);
    testData=fillMissingMonths(testData,hotelList);
    document.getElementById('loadStatus').innerText+=` + test(${testData.length})`;
  }});
});

function updateHotelDropdown(){
  const sel=document.getElementById('hotelDropdown');
  sel.innerHTML='';
  hotelList.forEach(h=>{
    const opt=document.createElement('option'); opt.value=h; opt.innerText=h;
    sel.appendChild(opt);
  });
}

function renderMergeOverview(){
  let html='<table border="1"><tr><th>Hotel</th><th>Year</th><th>Month</th><th>Cancellation Rate</th><th>Avg Room Price</th><th>Lead Time</th></tr>';
  trainData.slice(0,50).forEach(d=>{
    html+=`<tr><td>${d.hotel}</td><td>${d.year}</td><td>${d.month}</td><td>${(d.cancellation_rate*100).toFixed(2)}%</td><td>${d.avg_room_price}</td><td>${d.lead_time_avg}</td></tr>`;
  });
  html+='</table>';
  document.getElementById('mergeOverview').innerHTML=html;
}

function renderEDACharts(){
  const corrCtx=document.getElementById('correlationHeatmap').getContext('2d');
  const features=['cancellation_rate','avg_room_price','lead_time_avg'];
  const corrMatrix=features.map((f1,i)=>features.map((f2,j)=>{
    const x=trainData.map(d=>parseFloat(d[f1])); const y=trainData.map(d=>parseFloat(d[f2]));
    const meanX=x.reduce((a,b)=>a+b,0)/x.length;
    const meanY=y.reduce((a,b)=>a+b,0)/y.length;
    const cov=x.map((v,k)=> (v-meanX)*(y[k]-meanY)).reduce((a,b)=>a+b,0)/x.length;
    const stdX=Math.sqrt(x.map(v=>(v-meanX)**2).reduce((a,b)=>a+b,0)/x.length);
    const stdY=Math.sqrt(y.map(v=>(v-meanY)**2).reduce((a,b)=>a+b,0)/x.length);
    return cov/(stdX*stdY);
  }));
  new Chart(corrCtx,{type:'matrix',data:{datasets:[{label:'Correlation',data:corrMatrix.map((row,i)=>row.map((val,j)=>({x:j,y:i,v:val}))).flat(),backgroundColor:ctx=>{
    const v=ctx.dataset.data[ctx.dataIndex].v; return `rgba(255,0,0,${Math.abs(v)})`;}}]},options:{plugins:{tooltip:{callbacks:{label:ctx=>`Corr: ${ctx.dataset.data[ctx.dataIndex].v.toFixed(2)}`}}},scales:{x:{type:'category',labels:features},y:{type:'category',labels:features}}}});
}
