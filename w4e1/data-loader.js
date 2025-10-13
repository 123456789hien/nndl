// data-loader.js
// ES6 module that parses a local CSV file (uploaded via <input type="file">),
// pivots data by Symbol/Date, normalizes per-stock (MinMax on Open & Close),
// and prepares sliding-window samples:
// Input: last 12 days' [Open, Close] for all 10 symbols -> shape [samples, 12, 20]
// Output: binary up/down labels per stock for offsets 1,2,3 -> shape [samples, 30]
// Exports: DataLoader class with async loadFile(file) -> returns prepared tensors and metadata.

// NOTE: This implementation avoids external libs (PapaParse) to remain client-only.

export default class DataLoader {
  constructor(opts = {}) {
    // options: windowSize (12), forecastHorizon (3), trainSplit (0.8)
    this.windowSize = opts.windowSize || 12;
    this.forecastHorizon = opts.forecastHorizon || 3; // offsets 1..3
    this.trainSplit = opts.trainSplit || 0.8;
    this.requiredColumns = ['Date', 'Symbol', 'Open', 'Close'];
    this.MIN_SAMPLES_CHECK = 50;
  }

  // Public API: load a File object (CSV) and prepare tensors
  async loadFile(file) {
    if (!file) throw new Error('No file provided to DataLoader.loadFile');
    const text = await this._readFile(file);
    const rows = this._parseCSV(text);
    this._validateColumns(rows);
    const pivot = this._pivotBySymbol(rows);
    const symbols = Object.keys(pivot).sort();
    if (symbols.length === 0) throw new Error('No symbols found in CSV.');
    // Ensure consistent ordering of dates across symbols and that all symbols share same dates
    const commonDates = this._intersectDates(pivot);
    if (commonDates.length < this.windowSize + this.forecastHorizon + 1) {
      throw new Error(`Not enough common dates across symbols. Need at least ${this.windowSize + this.forecastHorizon + 1} days.`);
    }
    // Reduce each symbol array to only commonDates and sorted ascending
    const aligned = this._alignByDates(pivot, commonDates, symbols);

    // Build raw arrays: for each symbol, arrays of opens and closes (numbers)
    const raw = this._buildRawArrays(aligned, symbols);

    // Normalize per stock (min-max) using entire dataset for that stock
    const normResult = this._normalizePerStock(raw, symbols);

    // Sliding windows to create samples
    const samples = this._createSamples(normResult, symbols);

    if (samples.X.length < this.MIN_SAMPLES_CHECK) {
      console.warn('Warning: small number of samples:', samples.X.length);
    }

    // Chronological split
    const splitIndex = Math.floor(samples.X.length * this.trainSplit);
    const X_train_arr = samples.X.slice(0, splitIndex);
    const y_train_arr = samples.y.slice(0, splitIndex);
    const X_test_arr = samples.X.slice(splitIndex);
    const y_test_arr = samples.y.slice(splitIndex);

    // Convert to tf tensors (float32)
    // X: [samples, windowSize, features] features = symbols.length * 2
    const tf = window.tf;
    const X_train = tf.tensor3d(X_train_arr, [X_train_arr.length, this.windowSize, symbols.length * 2], 'float32');
    const X_test = tf.tensor3d(X_test_arr, [X_test_arr.length, this.windowSize, symbols.length * 2], 'float32');
    const y_train = tf.tensor2d(y_train_arr, [y_train_arr.length, symbols.length * this.forecastHorizon], 'float32');
    const y_test = tf.tensor2d(y_test_arr, [y_test_arr.length, symbols.length * this.forecastHorizon], 'float32');

    return {
      X_train, y_train, X_test, y_test,
      symbols,
      raw_close_by_symbol: raw.close, // original closes (not normalized) indexed by symbol -> useful for plotting
      normalized_meta: normResult.meta // per-stock min/max for potential inverse transforms
    };
  }

  // ------- Internal helpers --------

  _readFile(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = (e) => reject(new Error('Failed to read file: ' + e.message));
      fr.readAsText(file);
    });
  }

  _parseCSV(text) {
    // Very small and robust CSV parser (handles quoted fields)
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length === 0) throw new Error('CSV is empty.');
    const header = this._splitCSVLine(lines[0]).map(h => h.trim());
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const fields = this._splitCSVLine(lines[i]);
      if (fields.length !== header.length) {
        // skip malformed line but warn
        console.warn(`Skipping line ${i + 1} due to column mismatch.`);
        continue;
      }
      const obj = {};
      for (let j = 0; j < header.length; j++) obj[header[j]] = fields[j];
      rows.push(obj);
    }
    return { header, rows };
  }

  _splitCSVLine(line) {
    // basic CSV splitter that respects quotes
    const res = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"' || ch === "'") {
        if (inQuotes && line[i + 1] === ch) { cur += ch; i++; } // escaped quote
        else inQuotes = !inQuotes;
      } else if (ch === ',' && !inQuotes) {
        res.push(cur);
        cur = '';
      } else cur += ch;
    }
    res.push(cur);
    return res;
  }

  _validateColumns(parsed) {
    const header = parsed.header;
    for (const col of this.requiredColumns) {
      if (!header.includes(col)) {
        throw new Error(`CSV missing required column: ${col}`);
      }
    }
  }

  _pivotBySymbol(parsed) {
    // returns { SYMBOL: [ {Date, Open, Close}, ... ] }
    const map = {};
    for (const r of parsed.rows) {
      const date = r['Date'];
      const sym = r['Symbol'];
      const open = r['Open'];
      const close = r['Close'];
      if (!date || !sym) continue;
      if (!map[sym]) map[sym] = [];
      // attempt numeric parse
      const o = parseFloat(open);
      const c = parseFloat(close);
      if (Number.isNaN(o) || Number.isNaN(c)) {
        // skip rows with invalid numeric content but warn
        console.warn(`Skipping row for ${sym} on ${date} due to invalid Open/Close`);
        continue;
      }
      map[sym].push({ Date: date, Open: o, Close: c });
    }
    // sort each symbol by Date ascending (lexicographic can work for YYYY-MM-DD)
    for (const s of Object.keys(map)) {
      map[s].sort((a, b) => (a.Date < b.Date ? -1 : a.Date > b.Date ? 1 : 0));
    }
    return map;
  }

  _intersectDates(pivot) {
    // find the intersection of dates across all symbols
    const lists = Object.values(pivot).map(arr => arr.map(x => x.Date));
    if (lists.length === 0) return [];
    let common = new Set(lists[0]);
    for (let i = 1; i < lists.length; i++) {
      const s = new Set(lists[i]);
      common = new Set([...common].filter(d => s.has(d)));
    }
    // return sorted array
    const arr = [...common].sort();
    return arr;
  }

  _alignByDates(pivot, commonDates, symbols) {
    // Produce { symbol: [ {Date, Open, Close}, ... ] } with only commonDates
    const aligned = {};
    const dateSet = new Set(commonDates);
    for (const s of symbols) {
      const arr = pivot[s].filter(x => dateSet.has(x.Date)).sort((a,b)=>a.Date<b.Date?-1:1);
      if (arr.length !== commonDates.length) {
        // If some dates missing, this will try to align by date index but warn
        console.warn(`Symbol ${s} had ${arr.length} common dates vs ${commonDates.length}.`);
      }
      aligned[s] = arr;
    }
    return aligned;
  }

  _buildRawArrays(aligned, symbols) {
    // Build arrays indexed by symbol: opens[], closes[], dates[]
    const result = { open: {}, close: {}, dates: null };
    for (const s of symbols) {
      const arr = aligned[s];
      if (!arr || arr.length === 0) throw new Error(`Symbol ${s} has no data after alignment.`);
      result.open[s] = arr.map(x => x.Open);
      result.close[s] = arr.map(x => x.Close);
      // capture dates from first symbol
      if (!result.dates) result.dates = arr.map(x => x.Date);
    }
    return result;
  }

  _normalizePerStock(raw, symbols) {
    // Min-Max normalize Open and Close for each stock using the entire series
    // Return normalized arrays and meta {minOpen,maxOpen,minClose,maxClose} per symbol
    const norm = { open: {}, close: {}, meta: {} };
    for (const s of symbols) {
      const opens = raw.open[s];
      const closes = raw.close[s];
      const minO = Math.min(...opens);
      const maxO = Math.max(...opens);
      const minC = Math.min(...closes);
      const maxC = Math.max(...closes);
      const denomO = maxO - minO || 1e-8;
      const denomC = maxC - minC || 1e-8;
      norm.open[s] = opens.map(v => (v - minO) / denomO);
      norm.close[s] = closes.map(v => (v - minC) / denomC);
      norm.meta[s] = { minOpen: minO, maxOpen: maxO, minClose: minC, maxClose: maxC };
    }
    return norm;
  }

  _createSamples(normResult, symbols) {
    // For each time index t where we can take windowSize previous days ending at t (inclusive)
    // and also have forecastHorizon future days available, create:
    // X: concatenation of per-symbol features [Open, Close] for last windowSize days => shape (windowSize, symbols*2)
    // y: for each symbol and offset in 1..forecastHorizon: label = 1 if Close(t+offset) > Close(t) else 0
    const X = [];
    const y = [];
    const dates = Object.keys(normResult.open).length ? Object.keys(normResult.open).map(() => null) : []; // placeholder
    // Determine series length (common across symbols)
    const s0 = symbols[0];
    const L = normResult.open[s0].length;
    // iterate t from (windowSize - 1) to (L - forecastHorizon - 1)
    for (let t = this.windowSize - 1; t <= L - this.forecastHorizon - 1; t++) {
      try {
        // Build window data for last windowSize days: indices (t - windowSize + 1) .. t
        const windowRows = []; // will be array of length windowSize, each row is length symbols*2
        for (let w = t - this.windowSize + 1; w <= t; w++) {
          const row = [];
          for (const s of symbols) {
            row.push(normResult.open[s][w]);
            row.push(normResult.close[s][w]);
          }
          windowRows.push(row);
        }
        // Build labels: for each symbol, for offset 1..forecastHorizon compare close[t+offset] > close[t]
        const labelRow = [];
        for (const s of symbols) {
          const baseClose = normResult.close[s][t]; // normalized close at D
          for (let offset = 1; offset <= this.forecastHorizon; offset++) {
            const futureClose = normResult.close[s][t + offset];
            const bit = futureClose > baseClose ? 1 : 0;
            labelRow.push(bit);
          }
        }
        X.push(windowRows); // shape [windowSize, symbols*2]
        y.push(labelRow);   // length symbols*forecastHorizon
      } catch (err) {
        // shape mismatch; skip sample
        console.warn('Skipped sample at t=', t, err);
        continue;
      }
    }
    return { X, y };
  }
}
