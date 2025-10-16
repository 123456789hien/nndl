// data-loader.js
// parse CSV with ; delimiter and normalize rows
let rawTrain = [];
let rawTest = [];
let monthlyAgg = []; // aggregated monthly per room_type
let groupKey = 'room_type'; // default grouping

// helper: safe trim (handles non-string values and BOM)
function safeTrim(v){
  if (v === null || v === undefined) return null;
  let s = String(v);
  // remove BOM if present
  if (s.charCodeAt(0) === 0xFEFF) s = s.slice(1);
  return s.trim();
}

// normalize one CSV row (map proper types)
function normalizeRow(row){
  // row keys come from header: e.g. ID;n_adults;n_children;...;avg_room_price;special_requests;status
  const out = {};
  out.ID = safeTrim(row['ID'] || row['Id'] || row['id']);
  out.n_adults = (row['n_adults'] === '' || row['n_adults'] === undefined) ? null : Number(row['n_adults']);
  out.n_children = (row['n_children'] === '' || row['n_children'] === undefined) ? null : Number(row['n_children']);
  out.weekend_nights = (row['weekend_nights'] === '' || row['weekend_nights'] === undefined) ? null : Number(row['weekend_nights']);
  out.week_nights = (row['week_nights'] === '' || row['week_nights'] === undefined) ? null : Number(row['week_nights']);
  out.meal_plan = safeTrim(row['meal_plan']);
  out.car_parking_space = (row['car_parking_space'] === '' || row['car_parking_space'] === undefined) ? null : Number(row['car_parking_space']);
  // try common variations for room_type column name
  out.room_type = safeTrim(row['room_type'] || row['Room_Type'] || row['room type'] || row['roomType']);
  out.lead_time = (row['lead_time'] === '' || row['lead_time'] === undefined) ? null : Number(row['lead_time']);

  // robust parsing year/month: handle strings, spaces, BOMs
  const rawYear = safeTrim(row['year'] || row['Year'] || row['arrival_year']);
  const rawMonth = safeTrim(row['month'] || row['Month'] || row['arrival_month']);
  out.year = (rawYear === null || rawYear === '' || isNaN(Number(rawYear))) ? null : Number(rawYear);
  out.month = (rawMonth === null || rawMonth === '' || isNaN(Number(rawMonth))) ? null : Number(rawMonth);

  out.date = (row['date'] === '' || row['date'] === undefined) ? null : Number(row['date']);
  out.market_segment = safeTrim(row['market_segment']);
  out.repeated_guest = (row['repeated_guest'] === '' || row['repeated_guest'] === undefined) ? null : Number(row['repeated_guest']);
  out.previous_cancellations = (row['previous_cancellations'] === '' || row['previous_cancellations'] === undefined) ? null : Number(row['previous_cancellations']);
  out.previous_bookings_not_canceled = (row['previous_bookings_not_canceled'] === '' || row['previous_bookings_not_canceled'] === undefined) ? null : Number(row['previous_bookings_not_canceled']);
  out.avg_room_price = (row['avg_room_price'] === '' || row['avg_room_price'] === undefined) ? null : Number(row['avg_room_price']);
  out.special_requests = (row['special_requests'] === '' || row['special_requests'] === undefined) ? null : Number(row['special_requests']);
  out.status = (row['status'] === '' || row['status'] === undefined) ? null : Number(row['status']); // 1 canceled, 0 not

  // convenience fields
  out.group = out.room_type || out.ID || 'unknown';
  // cancellation flag per booking
  out.is_canceled = (out.status === 1) ? 1 : 0;
  return out;
}

// parse CSV file with ; delimiter
function parseCsvFile(file){
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      delimiter: ";",      // IMPORTANT: your CSV uses ;
      dynamicTyping: false,
      skipEmptyLines: true,
      complete: results => {
        // results.data may include rows where header detection slightly off; normalize each row
        try {
          const norm = results.data.map(normalizeRow);
          resolve(norm);
        } catch (e) {
          reject(e);
        }
      },
      error: err => reject(err)
    });
  });
}

// aggregate monthly by groupKey (room_type)
function aggregateMonthly(dataset){
  const map = new Map();
  dataset.forEach(r => {
    if (!r.group) return;
    // only aggregate rows with valid year and month (non-null) into meaningful buckets;
    // if year/month missing, keep in map with year=0/month=0 (won't affect monthly trends)
    const yr = (r.year === null || isNaN(r.year)) ? 0 : r.year;
    const mo = (r.month === null || isNaN(r.month)) ? 0 : r.month;
    const key = `${r.group}||${yr}||${mo}`;
    if (!map.has(key)) {
      map.set(key, { group: r.group, year: yr, month: mo, total_bookings:0, cancelled_bookings:0, avg_room_price_sum:0, lead_time_sum:0, count_price:0, count_lead:0 });
    }
    const obj = map.get(key);
    obj.total_bookings += 1;
    obj.cancelled_bookings += (r.is_canceled || 0);
    if (r.avg_room_price !== null && !Number.isNaN(r.avg_room_price)) { obj.avg_room_price_sum += r.avg_room_price; obj.count_price += 1; }
    if (r.lead_time !== null && !Number.isNaN(r.lead_time)) { obj.lead_time_sum += r.lead_time; obj.count_lead += 1; }
  });

  const arr = [];
  map.forEach(v => {
    // compute means; if no data for price/lead then set 0 (so aggregated table has numbers not NaN)
    const avg_price = v.count_price ? v.avg_room_price_sum / v.count_price : 0;
    const lead_avg = v.count_lead ? v.lead_time_sum / v.count_lead : 0;
    const cancel_rate = v.total_bookings ? v.cancelled_bookings / v.total_bookings : 0;
    arr.push({
      group: v.group,
      year: v.year,
      month: v.month,
      total_bookings: v.total_bookings,
      cancelled_bookings: v.cancelled_bookings,
      cancellation_rate: cancel_rate,
      avg_room_price: avg_price,
      lead_time_avg: lead_avg
    });
  });

  arr.sort((a,b) => {
    if (a.group < b.group) return -1;
    if (a.group > b.group) return 1;
    if (a.year !== b.year) return a.year - b.year;
    return a.month - b.month;
  });

  const years = arr.map(d => d.year).filter(y=>y!=null && y !== 0);
  const minYear = years.length ? Math.min(...years) : new Date().getFullYear();
  arr.forEach(d => {
    // if year === 0 (invalid), we still compute month_index relative to minYear to keep ordering but those rows are peripheral
    d.month_index = (d.year - minYear) * 12 + (d.month - 1);
  });

  return arr;
}

// fill missing months (create entries with 0 values for missing months) for each group
function fillMissingMonthsAgg(arr){
  if (!arr || arr.length === 0) return [];
  const groups = [...new Set(arr.map(d => d.group))];
  const years = arr.map(d=>d.year).filter(y=>y!=null && y !== 0);
  const minY = Math.min(...years);
  const maxY = Math.max(...years);
  const out = [];
  groups.forEach(g=>{
    const byKey = new Map();
    arr.filter(x=>x.group===g).forEach(r=> byKey.set(`${r.year}-${r.month}`, r));
    for (let y=minY; y<=maxY; y++){
      for (let m=1; m<=12; m++){
        const k = `${y}-${m}`;
        if (byKey.has(k)){
          out.push(byKey.get(k));
        } else {
          // create zero rows for truly missing months so heatmap / charts can work reliably
          out.push({ group: g, year: y, month: m, total_bookings:0, cancelled_bookings:0, cancellation_rate: 0, avg_room_price: 0, lead_time_avg: 0, month_index: (y-minY)*12 + (m-1) });
        }
      }
    }
  });
  out.sort((a,b)=> (a.group < b.group ? -1 : a.group> b.group?1: a.month_index - b.month_index));
  return out;
}

// interpolation helpers
function forwardFill(arr, field){
  let last = null;
  for (let i=0;i<arr.length;i++){
    arr[i][field] = (arr[i][field] === null || arr[i][field] === undefined) ? last : arr[i][field];
    if (arr[i][field] !== null && arr[i][field] !== undefined) last = arr[i][field];
  }
  return arr;
}

function linearInterpolateSeries(arr, field){
  let i=0;
  while(i < arr.length){
    if (arr[i][field] === null || arr[i][field] === undefined){
      let j = i+1;
      while(j < arr.length && (arr[j][field] === null || arr[j][field] === undefined)) j++;
      const leftVal = (i-1 >= 0) ? arr[i-1][field] : null;
      const rightVal = (j < arr.length) ? arr[j][field] : null;
      if (leftVal === null && rightVal === null){
        for (let k = i; k<j; k++) arr[k][field] = 0;
      } else if (leftVal === null) {
        for (let k = i; k<j; k++) arr[k][field] = rightVal;
      } else if (rightVal === null) {
        for (let k = i; k<j; k++) arr[k][field] = leftVal;
      } else {
        const steps = j - (i-1);
        for (let k = i; k<j; k++){
          const t = (k - (i-1)) / steps;
          arr[k][field] = leftVal * (1 - t) + rightVal * t;
        }
      }
      i = j;
    } else {
      i++;
    }
  }
  return arr;
}

// public function to prepare aggregated monthly from uploaded raw CSVs
async function prepareMonthly(trainFile, testFile, method='none'){
  rawTrain = await parseCsvFile(trainFile);
  rawTest = await parseCsvFile(testFile);
  groupKey = document.getElementById('group-key') ? document.getElementById('group-key').value : 'room_type';

  const combined = rawTrain.concat(rawTest);
  const agg = aggregateMonthly(combined);
  // fillMissingMonthsAgg will create zero-rows for months that truly missing between min and max year present
  const filled = fillMissingMonthsAgg(agg);

  const interp = document.getElementById('interp-method') ? document.getElementById('interp-method').value : 'none';
  const groupsSet = [...new Set(filled.map(d=>d.group))];
  const final = [];
  groupsSet.forEach(g=>{
    const seq = filled.filter(d=>d.group===g).sort((a,b)=>a.month_index-b.month_index);
    if(interp==='ffill'){ forwardFill(seq,'cancellation_rate'); forwardFill(seq,'avg_room_price'); forwardFill(seq,'lead_time_avg'); }
    else if(interp==='linear'){ linearInterpolateSeries(seq,'cancellation_rate'); linearInterpolateSeries(seq,'avg_room_price'); linearInterpolateSeries(seq,'lead_time_avg'); }
    final.push(...seq);
  });

  monthlyAgg = final;
  return monthlyAgg;
}
