let trainData=[], testData=[];

function normalizeRow(row){
    let d = {};
    Object.keys(row).forEach(k=>{
        if(!row[k]) { d[k]=null; return; }
        if(["ID","meal_plan","room_type","market_segment"].includes(k)){
            d[k]=row[k].trim();
        } else {
            d[k]=Number(row[k]);
        }
    });
    d.cancellation_rate = d.status!==null? d.status : 0;
    d.lead_time_avg = d.lead_time;
    d.avg_room_price = d.avg_room_price;
    d.hotel = d.ID;
    return d;
}

function parseCSV(file, callback){
    Papa.parse(file,{
        header:true,
        skipEmptyLines:true,
        dynamicTyping:true,
        complete: function(results){
            const data = results.data.map(normalizeRow);
            callback(data);
        }
    });
}

function fillMissingMonths(data){
    const filled=[];
    const hotels=[...new Set(data.map(d=>d.hotel))];
    hotels.forEach(h=>{
        const hotelData=data.filter(d=>d.hotel===h);
        const years=[...new Set(hotelData.map(d=>d.year))];
        years.forEach(y=>{
            for(let m=1;m<=12;m++){
                let found=hotelData.find(d=>d.year===y && d.month===m);
                if(found){
                    filled.push(found);
                } else {
                    filled.push({hotel:h, year:y, month:m, cancellation_rate:null, avg_room_price:null, lead_time_avg:null});
                }
            }
        });
    });
    return filled;
}
