//
// data-loader.js
// parse CSV with semicolon delimiter, normalize, aggregate monthly per room_type
//

/*
Expect CSV with semicolon delimiter `;` and columns like:
ID;n_adults;n_children;weekend_nights;week_nights;meal_plan;car_parking_space;room_type;lead_time;year;month;date;market_segment;repeated_guest;previous_cancellations;previous_bookings_not_canceled;avg_room_price;special_requests;status
*/

let RAW_TRAIN = [];
let RAW_TEST = [];
let AGG_MONTHLY = [];    // aggregated rows: groupKey(room_type), year, month, total_bookings, cancelled, cancellation_rate, avg_room_price, lead_time_avg
let GROUP_KEY = 'room_type'; // grouping dimension for business insight (Room Type)
let MIN_YEAR = 9999, MAX_YEAR = 0;

function parseFileWithSemicolon(file, cb) {
  Papa.parse(file, {
    header: true,
    delimiter: ";",
    skipEmptyLines: true,
    dynamicTyping: true,
    complete: (res) => cb(res.data),
    error: (err) => { alert('CSV parse error: ' + err.message); console.error(err); }
  });
}

function normalizeBookingRow(raw) {
  // Map and coerce types safely
  const out = {};
  out.ID = raw['ID'] ? String(raw['ID']).trim() : null;
  out.n_adults = raw['n_adults'] != null ? Number(raw['n_adults']) : null;
  out.n_children = raw['n_children'] != null ? Number(raw['n_children']) : null;
  out.weekend_nights = raw['weekend_nights'] != null ? Number(raw['weekend_nights']) : null;
  out.week_nights = raw['week_nights'] != null ? Number(raw['week_nights']) : null;
  out.meal_plan = raw['meal_plan'] ? String(raw['meal_plan']).trim() : null;
  out.car_parking_space = raw['car_parking_space'] != null ? Number(raw['car_parking_space']) : 0;
  out.room_type = raw['room_type'] ? String(raw['room_type']).trim() : 'Unknown';
  out.lead_time = raw['lead_time'] != null ? Number(raw['lead_time']) : null;
  out.year = raw['year'] != null ? Number(raw['year']) : null;
  out.month = raw['month'] != null ? Number(raw['month']) : null;
  out.date = raw['date'] != null ? Number(raw['date']) : null;
  out.market_segment = raw['market_segment'] ? String(raw['market_segment']).trim() : null;
  out.repeated_guest = raw['repeated_guest'] != null ? Number(raw['repeated_guest']) : 0;
  out.previous_cancellations = raw['previous_cancellations'] != null ? Number(raw['previous_cancellations']) : 0;
  out.previous_bookings_not_canceled = raw['previous_bookings_not_canceled'] != null ? Number(raw['previous_bookings_not_canceled']) : 0;
  out.avg_room_price = raw['avg_room_price'] != null ? Number(raw['avg_room_price']) : null;
  out.special_requests = raw['special_requests'] != null ? Number(raw['special_requests']) : 0;
  out.status = raw['status'] != null ? Number(raw['status']) : 0; // 1 canceled, 0 not
  return out;
}

function aggregateMonthly(bookings) {
  // Build map: key = room_type|year|month
  const map = new Map();
  bookings.forEach(b => {
    if (!b[GROUP_KEY] || !b.year || !b.month) return;
    const key = `${b[GROUP_KEY]}|${b.year}|${b.month}`;
    if (!map.has(key)) {
      map.set(key, { room_type: b[GROUP_KEY], year: b.year, month: b.month, total_bookings: 0, cancelled: 0, sum_price: 0, sum_lead: 0, count_price: 0, count_lead: 0 });
    }
    const rec = map.get(key);
    rec.total_bookings += 1;
    if (Number(b.status) === 1) rec.cancelled += 1;
    if (b.avg_room_price != null && !Number.isNaN(b.avg_room_price)) { rec.sum_price += b.avg_room_price; rec.count_price += 1; }
    if (b.lead_time != null && !Number.isNaN(b.lead_time)) { rec.sum_lead += b.lead_time; rec.count_lead += 1; }
  });

  // Turn into array with cancellation_rate and averages
  const arr = [];
  for (const rec of map.values()) {
    const avg_price = rec.count_price > 0 ? rec.sum_price / rec.count_price : null;
    const avg_lead = rec.count_lead > 0 ? rec.sum_lead / rec.count_lead : null;
    arr.push({
      room_type: rec.room_type,
      year: rec.year,
      month: rec.month,
      total_bookings: rec.total_bookings,
      cancelled_bookings: rec.cancelled,
      cancellation_rate: rec.total_bookings > 0 ? rec.cancelled / rec.total_bookings : null,
      avg_room_price: avg_price,
      lead_time_avg: avg_lead
    });
  }
  return arr;
}

function expandAndFillMonths(aggArray) {
  // For each room_type, ensure continuous months from MIN_YEAR..MAX_YEAR
  const byGroup = {};
  aggArray.forEach(r => {
    const g = r.room_type;
    byGroup[g] = byGroup[g] || [];
    byGroup[g].push(r);
    if (r.year < MIN_YEAR) MIN_YEAR = r.year;
    if (r.year > MAX_YEAR) MAX_YEAR = r.year;
  });

  const expanded = [];
  Object.keys(byGroup).forEach(g => {
    // collect years present and use global MIN..MAX to ensure consistent timeline
    for (let y = MIN_YEAR; y <= MAX_YEAR; y++) {
      for (let m = 1; m <= 12; m++) {
        const found = byGroup[g].find(r => r.year === y && r.month === m);
        if (found) {
          found.month_index = (y - MIN_YEAR) * 12 + (m - 1);
          expanded.push(found);
        } else {
          expanded.push({
            room_type: g,
            year: y,
            month: m,
            month_index: (y - MIN_YEAR) * 12 + (m - 1),
            total_bookings: 0,
            cancelled_bookings: 0,
            cancellation_rate: null,
            avg_room_price: null,
            lead_time_avg: null
          });
        }
      }
    }
  });
  // sort
  expanded.sort((a,b) => a.room_type.localeCompare(b.room_type) || a.month_index - b.month_index);
  return expanded;
}

// linear interpolate nulls per group for features
function interpolateMissingPerGroup(rows, fields = ['cancellation_rate','avg_room_price','lead_time_avg']) {
  const out = [];
  const byGroup = {};
  rows.forEach(r => { byGroup[r.room_type] = byGroup[r.room_type] || []; byGroup[r.room_type].push(r); });

  Object.keys(byGroup).forEach(g => {
    const seq = byGroup[g];
    fields.forEach(field => {
      // find indices with valid numbers
      let i=0;
      while (i < seq.length) {
        if (seq[i][field] == null) {
          // locate next valid
          let j = i+1;
          while (j < seq.length && seq[j][field] == null) j++;
          // now seq[i..j-1] are nulls; seq[j] may exist or not
          const left = i-1 >= 0 ? seq[i-1][field] : null;
          const right = j < seq.length ? seq[j][field] : null;
          for (let k=i; k<j; k++) {
            if (left != null && right != null) {
              // linear interp
              seq[k][field] = left + (right - left) * ((k - (i-1)) / (j - (i-1)));
            } else if (left != null) {
              seq[k][field] = left; // forward fill
            } else if (right != null) {
              seq[k][field] = right; // backfill
            } else {
              seq[k][field] = 0;
            }
          }
          i = j;
        } else {
          i++;
        }
      }
    });
    out.push(...seq);
  });
  return out.sort((a,b)=>a.room_type.localeCompare(b.room_type) || a.month_index - b.month_index);
}

// Public loader
function prepareFromFiles(trainFile, testFile, onReady) {
  parseFileWithSemicolon(trainFile, rawTrain => {
    parseFileWithSemicolon(testFile, rawTest => {
      RAW_TRAIN = rawTrain.map(normalizeBookingRow);
      RAW_TEST = rawTest.map(normalizeBookingRow);

      // aggregate monthly (combine train+test for EDA)
      const aggTrain = aggregateMonthly(RAW_TRAIN);
      const aggTest = aggregateMonthly(RAW_TEST);
      const combined = aggTrain.concat(aggTest);

      // set min/max year from combined data
      MIN_YEAR = Infinity; MAX_YEAR = -Infinity;
      combined.forEach(r => { if(r.year < MIN_YEAR) MIN_YEAR = r.year; if(r.year > MAX_YEAR) MAX_YEAR = r.year; });

      let expanded = expandAndFillMonths(combined);
      expanded = interpolateMissingPerGroup(expanded);

      AGG_MONTHLY = expanded;

      // done
      onReady({
        rawTrain: RAW_TRAIN,
        rawTest: RAW_TEST,
        monthly: AGG_MONTHLY,
        groupKey: GROUP_KEY,
        minYear: MIN_YEAR,
        maxYear: MAX_YEAR
      });
    });
  });
}
