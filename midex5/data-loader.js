let trainData = [];
let testData = [];
let hotelList = [];

function normalizeRow(row){
    const normRow={};
    Object.keys(row).forEach(k=>{
        let val=row[k];
        if(val===null||val===undefined||val==="") normRow[k]=null;
        else if(!isNaN(val)) normRow[k]=Number(val);
        else normRow[k]=val.toString();
    });
    return normRow;
}

function fillMissingMonths(data){
    const filled=[];
    const hotels=[...new Set(data.map(d=>d.hotel))];
    hotels.forEach(h=>{
        const hotelData=data.filter(d=>d.hotel===h);
        for(let y=Math.min(...hotelData.map(d=>d.year)); y<=Math.max(...hotelData.map(d=>d.year)); y++){
            for(let m=1;m<=12;m++){
                let found=hotelData.find(d=>d.year===y && d.month===m);
                if(found) filled.push(found);
                else filled.push({hotel:h,year:y,month:m,cancellation_rate:0,avg_room_price:0,lead_time_avg:0});
            }
        }
    });
    return filled;
}

function loadCSV(file,isTrain=true){
    return new Promise((resolve,reject)=>{
        Papa.parse(file,{
            header:true, dynamicTyping:false, skipEmptyLines:true,
            complete:function(results){
                const data=results.data.map(normalizeRow);
                if(isTrain) trainData=data;
                else testData=data;
                hotelList=[...new Set(trainData.map(d=>d.hotel))];
                resolve(data);
            },
            error:function(err){ reject(err); }
        });
    });
}

document.getElementById('load-data-btn').addEventListener('click',async()=>{
    const trainFile=document.getElementById('train-file').files[0];
    const testFile=document.getElementById('test-file').files[0];
    if(!trainFile||!testFile){ alert("Upload both train & test CSV"); return; }
    document.getElementById('data-status').innerText="Loading...";
    await loadCSV(trainFile,true);
    await loadCSV(testFile,false);
    document.getElementById('data-status').innerText=`Loaded train(${trainData.length}) + test(${testData.length})`;

    // Fix Merge & Overview, fill missing
    trainData=trainData.map(d=>{
        d.hotel=d.hotel||d.ID||"Unknown Hotel";
        d.cancellation_rate=Number(d.status || 0);
        d.avg_room_price=Number(d.avg_room_price || 0);
        d.lead_time_avg=Number(d.lead_time || 0);
        return d;
    });
    trainData=fillMissingMonths(trainData);
    generateEDA(trainData);
});
