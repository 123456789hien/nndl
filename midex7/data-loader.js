// data-loader.js
// parse CSV with ; delimiter and normalize rows
let rawTrain = [];
let rawTest = [];
let monthlyAgg = []; // aggregated monthly per room_type
let groupKey = 'room_type'; // default grouping

// helper: safe trim
function safeTrim(v){
  return (v === null || v === undefined) ? null : String(v).trim();
}

// normalize one CSV row (map proper types)
function normalizeRow(row){
  // row keys come from header: e.g. ID;n_adults;n_children;...;avg_room_price;special_requests;status
  const out = {};
  out.ID = safeTrim(row['ID'] || row['Id'] || row['id']);
  out.n_adults = row['n_adults'] === '' || row['n_adults'] === undefined ? null : Number(row['n_adults']);
  out.n_children = row['n_children'] === '' ? null : Number(row['n_children']);
  out.weekend_nights = row['weekend_nights'] === '' ? null : Number(row['weekend_nights']);
  out.week_nights = row['week_nights'] === '' ? null : Number(row['week_nights']);
  out.meal_plan = safeTrim(row['meal_plan']);
  out.car_parking_space = row['car_parking_space'] === '' ? null : Number(row['car_parking_space']);
  out.room_type = safeTrim(row['room_type']);
  out.lead_time = row['lead_time'] === '' ? null : Number(row['lead_time']);
  out.year = row['year'] === '' ? null : Number(row['year']);
  out.month = row['month'] === '' ? null : Number(row['month']);
  out.date = row['date'] === '' ? null : Number(row['date']);
  out.market_segment = safeTrim(row['market_segment']);
  out.repeated_guest = row['repeated_guest'] === '' ? null : Number(row['repeated_guest']);
  out.previous_cancellations = row['previous_cancellations'] === '' ? null : Number(row['previous_cancellations']);
  out.previous_bookings_not_canceled = row['previous_bookings_not_canceled'] === '' ? null : Number(row['previous_bookings_not_canceled']);
  out.avg_room_price = row['avg_room_price'] === '' ? null : Number(row['avg_room_price']);
  out.special_requests = row['special_requests'] === '' ? null : Number(row['special_requests']);
  out.status = row['status'] === '' ? null : Number(row['status']); // 1 canceled, 0 not
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
        const norm = results.data.map(normalizeRow);
        resolve(norm);
      },
      error: err => reject(err)
    });
  });
}

// aggregate monthly by groupKey (room_type)
function aggregateMonthly(dataset){
  // dataset: array of normalized booking rows
  // produce rows keyed by (group, year, month)
  const map = new Map();
  dataset.forEach(r => {
    if (!r.group || !r.year || !r.month) return;
    const key = `${r.group}||${r.year}||${r.month}`;
    if (!map.has(key)) {
      map.set(key, { group: r.group, year: r.year, month: r.month, total_bookings:0, cancelled_bookings:0, avg_room_price_sum:0, lead_time_sum:0, count_price:0, count_lead:0 });
    }
    const obj = map.get(key);
    obj.total_bookings += 1;
    obj.cancelled_bookings += (r.is_canceled || 0);
    if (r.avg_room_price !== null && !Number.isNaN(r.avg_room_price)) { obj.avg_room_price_sum += r.avg_room_price; obj.count_price += 1; }
    if (r.lead_time !== null && !Number.isNaN(r.lead_time)) { obj.lead_time_sum += r.lead_time; obj.count_lead += 1; }
  });

  // convert to array with computed fields
  const arr = [];
  map.forEach(v => {
    const avg_price = v.count_price ? v.avg_room_price_sum / v.count_price : null;
    const lead_avg = v.count_lead ? v.lead_time_sum / v.count_lead : null;
    const cancel_rate = v.total_bookings ? v.cancelled_bookings / v.total_bookings : null;
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

  // sort by group -> year -> month and add month_index (continuous)
  arr.sort((a,b) => {
    if (a.group < b.group) return -1;
    if (a.group > b.group) return 1;
    if (a.year !== b.year) return a.year - b.year;
    return a.month - b.month;
  });

  // compute global min year for month_index
  const years = arr.map(d => d.year).filter(y=>y!=null);
  const minYear = years.length ? Math.min(...years) : new Date().getFullYear();
  arr.forEach(d => {
    d.month_index = (d.year - minYear) * 12 + (d.month - 1);
  });

  return arr;
}

// fill missing months (create entries with null values for missing months) for each group
function fillMissingMonthsAgg(arr){
  if (!arr || arr.length === 0) return [];
  const groups = [...new Set(arr.map(d => d.group))];
  // determine global year range
  const years = arr.map(d=>d.year).filter(y=>y!=null);
  const minY = Math.min(...years);
  const maxY = Math.max(...years);
  const out = [];
  groups.forEach(g=>{
    // build map for this group
    const byKey = new Map();
    arr.filter(x=>x.group===g).forEach(r=> byKey.set(`${r.year}-${r.month}`, r));
    for (let y=minY; y<=maxY; y++){
      for (let m=1; m<=12; m++){
        const k = `${y}-${m}`;
        if (byKey.has(k)){
          out.push(byKey.get(k));
        } else {
          out.push({ group: g, year: y, month: m, total_bookings:0, cancelled_bookings:0, cancellation_rate: null, avg_room_price: null, lead_time_avg: null, month_index: (y-minY)*12 + (m-1) });
        }
      }
    }
  });
  // sort
  out.sort((a,b)=> (a.group < b.group ? -1 : a.group> b.group?1: a.month_index - b.month_index));
  return out;
}

// interpolation helpers
function forwardFill(arr, field){
  let last = null;
  for (let i=0;i<arr.length;i++){
    if (arr[i][field] === null || arr[i][field] === undefined) {
      arr[i][field] = last;
    } else {
      last = arr[i][field];
    }
  }
  return arr;
}

function linearInterpolateSeries(arr, field){
  // arr sorted by month_index, fill null by linear interpolation between known neighbors
  let i=0;
  while(i < arr.length){
    if (arr[i][field] === null || arr[i][field] === undefined){
      // find j > i where value exists
      let j = i+1;
      while(j < arr.length && (arr[j][field] === null || arr[j][field] === undefined)) j++;
      const leftVal = (i-1 >= 0) ? arr[i-1][field] : null;
      const rightVal = (j < arr.length) ? arr[j][field] : null;
      if (leftVal === null && rightVal === null){ // nothing to fill
        for (let k = i; k<j; k++) arr[k][field] = null;
      } else if (leftVal === null) {
        // fill with rightVal
        for (let k = i; k<j; k++) arr[k][field] = rightVal;
      } else if (rightVal === null) {
        // fill with leftVal
        for (let k = i; k<j; k++) arr[k][field] = leftVal;
      } else {
        // linear between leftVal and rightVal across (j - (i-1)) steps
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
  // parse files
  rawTrain = await parseCsvFile(trainFile);
  rawTest = await parseCsvFile(testFile);
  // decide grouping key chosen by UI (room_type recommended)
  groupKey = document.getElementById('group-key') ? document.getElementById('group-key').value : 'room_type';

  // aggregate by groupKey - we'll compute group property inside normalizeRow (group)
  const combined = rawTrain.concat(rawTest);
  const agg = aggregateMonthly(combined); // returns aggregated rows for groups/months with cancellation_rate possibly null
  const filled = fillMissingMonthsAgg(agg);

  // optionally interpolate missing values according to UI selection
  const interp = document.getElementById('interp-method') ? document.getElementById('interp-method').value : 'none';
  // group sequences per group
  const groups = [...new Set(filled.map(d=>d.group))];
  const final = [];
  groups.forEach(g=>{
    const seq = filled.filter(d=>d.group===g).sort((a,b)=>a.month_index - b.month_index);
    if (interp === 'ffill') {
      forwardFill(seq, 'cancellation_rate');
      forwardFill(seq, 'avg_room_price');
      forwardFill(seq, 'lead_time_avg');
    } else if (interp === 'linear') {
      linearInterpolateSeries(seq, 'cancellation_rate');
      linearInterpolateSeries(seq, 'avg_room_price');
      linearInterpolateSeries(seq, 'lead_time_avg');
    }
    final.push(...seq);
  });

  monthlyAgg = final;
  return monthlyAgg;
}
