// data-loader.js
// ES6 module: default export DataLoader
// Parses a local CSV (uploaded via <input type="file">), pivots by Symbol/Date,
// normalizes per-stock (MinMax on Open & Close), and prepares sliding-window samples.
//
// Outputs tensors: X_train, y_train, X_test, y_test and metadata: symbols, raw_close_by_symbol, normalized_meta
//
// Assumptions: CSV columns include at least: Date, Symbol, Open, Close
// Date format should be sortable lexicographically (e.g., YYYY-MM-DD).
//
// Usage (example):
//   const dl = new DataLoader();
//   const result = await dl.loadFile(fileInput.files[0]);

export default class DataLoader {
  constructor(opts = {}) {
    this.windowSize = opts.windowSize || 12;
    this.forecastHorizon = opts.forecastHorizon || 3; // offsets 1..3
    this.trainSplit = opts.trainSplit || 0.8;
    this.requiredColumns = ['Date', 'Symbol', 'Open', 'Close'];
    this.MIN_SAMPLES_CHECK = 10;
  }

  // Public: load CSV File object and return prepared tensors + metadata
  async loadFile(file) {
    if (!file) throw new Error('No file provided.');
    const text = await this._readFile(file);
    const parsed = this._parseCSV(text);
    this._validateColumns(parsed.header);

    const pivot = this._pivotBySymbol(parsed.rows);
    const symbols = Object.keys(pivot).sort();
    if (symbols.length === 0) throw new Error('No symbols found in CSV.');

    const commonDates = this._intersectDates(pivot);
    if (commonDates.length < this.windowSize + this.forecastHorizon + 1) {
      throw new Error(
        `Not enough common dates across symbols. Need at least ${this.windowSize + this.forecastHorizon + 1} common dates. Found ${commonDates.length}.`
      );
    }

    const aligned = this._alignByDates(pivot, commonDates, symbols);
    const raw = this._buildRawArrays(aligned, symbols);
    const norm = this._normalizePerStock(raw, symbols);
    const samples = this._createSamples(norm, symbols);

    if (samples.X.length < this.MIN_SAMPLES_CHECK) {
      console.warn('Small number of samples:', samples.X.length);
    }

    // Chronological split (keep order)
    const splitIndex = Math.floor(samples.X.length * this.trainSplit);
    const X_train_arr = samples.X.slice(0, splitIndex);
    const y_train_arr = samples.y.slice(0, splitIndex);
    const X_test_arr = samples.X.slice(splitIndex);
    const y_test_arr = samples.y.slice(splitIndex);

    // Convert to tensors
    const tf = window.tf;
    const sampleCountTrain = X_train_arr.length;
    const sampleCountTest = X_test_arr.length;
    const features = symbols.length * 2;

    const X_train = sampleCountTrain > 0
      ? tf.tensor3d(X_train_arr, [sampleCountTrain, this.windowSize, features], 'float32')
      : tf.tensor3d([], [0, this.windowSize, features], 'float32');
    const X_test = sampleCountTest > 0
      ? tf.tensor3d(X_test_arr, [sampleCountTest, this.windowSize, features], 'float32')
      : tf.tensor3d([], [0, this.windowSize, features], 'float32');

    const y_train = sampleCountTrain > 0
      ? tf.tensor2d(y_train_arr, [sampleCountTrain, symbols.length * this.forecastHorizon], 'float32')
      : tf.tensor2d([], [0, symbols.length * this.forecastHorizon], 'float32');
    const y_test = sampleCountTest > 0
      ? tf.tensor2d(y_test_arr, [sampleCountTest, symbols.length * this.forecastHorizon], 'float32')
      : tf.tensor2d([], [0, symbols.length * this.forecastHorizon], 'float32');

    return {
      X_train, y_train, X_test, y_test,
      symbols,
      raw_close_by_symbol: raw.close,
      normalized_meta: norm.meta,
      dates: raw.dates
    };
  }

  // ---------- Internal helpers ----------

  _readFile(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = (e) => reject(new Error('Failed to read file: ' + e.message));
      fr.readAsText(file);
    });
  }

  _parseCSV(text) {
    // Minimal robust CSV parser (handles quoted fields).
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length === 0) throw new Error('CSV empty.');
    const header = this._splitCSVLine(lines[0]).map(h => h.trim());
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const fields = this._splitCSVLine(lines[i]);
      if (fields.length !== header.length) {
        // ignore malformed lines but warn
        console.warn(`Skipping line ${i + 1}: column count mismatch.`);
        continue;
      }
      const obj = {};
      for (let j = 0; j < header.length; j++) obj[header[j]] = fields[j];
      rows.push(obj);
    }
    return { header, rows };
  }

  _splitCSVLine(line) {
    const res = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') { cur += '"'; i++; } else inQuotes = !inQuotes;
      } else if (ch === ',' && !inQuotes) {
        res.push(cur);
        cur = '';
      } else cur += ch;
    }
    res.push(cur);
    return res;
  }

  _validateColumns(header) {
    for (const col of this.requiredColumns) {
      if (!header.includes(col)) throw new Error(`CSV missing required column: ${col}`);
    }
  }

  _pivotBySymbol(rows) {
    const map = {};
    for (const r of rows) {
      const date = r['Date'];
      const sym = r['Symbol'];
      const open = parseFloat(r['Open']);
      const close = parseFloat(r['Close']);
      if (!date || !sym || Number.isNaN(open) || Number.isNaN(close)) {
        // skip invalid
        continue;
      }
      if (!map[sym]) map[sym] = [];
      map[sym].push({ Date: date, Open: open, Close: close });
    }
    // sort each symbol by date asc
    for (const s of Object.keys(map)) {
      map[s].sort((a, b) => (a.Date < b.Date ? -1 : a.Date > b.Date ? 1 : 0));
    }
    return map;
  }

  _intersectDates(pivot) {
    const lists = Object.values(pivot).map(arr => arr.map(x => x.Date));
    if (lists.length === 0) return [];
    let common = new Set(lists[0]);
    for (let i = 1; i < lists.length; i++) {
      const s = new Set(lists[i]);
      common = new Set([...common].filter(d => s.has(d)));
    }
    return [...common].sort();
  }

  _alignByDates(pivot, commonDates, symbols) {
    const aligned = {};
    const dateSet = new Set(commonDates);
    for (const s of symbols) {
      aligned[s] = pivot[s].filter(x => dateSet.has(x.Date)).sort((a,b)=>a.Date<b.Date?-1:1);
    }
    return aligned;
  }

  _buildRawArrays(aligned, symbols) {
    const open = {}, close = {};
    let dates = null;
    for (const s of symbols) {
      const arr = aligned[s];
      if (!arr || arr.length === 0) throw new Error(`Symbol ${s} has no aligned data.`);
      open[s] = arr.map(x => x.Open);
      close[s] = arr.map(x => x.Close);
      if (!dates) dates = arr.map(x => x.Date);
    }
    return { open, close, dates };
  }

  _normalizePerStock(raw, symbols) {
    const norm = { open: {}, close: {}, meta: {} };
    for (const s of symbols) {
      const opens = raw.open[s];
      const closes = raw.close[s];
      const minO = Math.min(...opens);
      const maxO = Math.max(...opens);
      const minC = Math.min(...closes);
      const maxC = Math.max(...closes);
      const denomO = (maxO - minO) || 1e-8;
      const denomC = (maxC - minC) || 1e-8;
      norm.open[s] = opens.map(v => (v - minO) / denomO);
      norm.close[s] = closes.map(v => (v - minC) / denomC);
      norm.meta[s] = { minOpen: minO, maxOpen: maxO, minClose: minC, maxClose: maxC };
    }
    return norm;
  }

  _createSamples(normResult, symbols) {
    const X = [];
    const y = [];
    const L = normResult.open[symbols[0]].length; // common length
    for (let t = this.windowSize - 1; t <= L - this.forecastHorizon - 1; t++) {
      const windowRows = [];
      for (let w = t - this.windowSize + 1; w <= t; w++) {
        const row = [];
        for (const s of symbols) {
          row.push(normResult.open[s][w]);
          row.push(normResult.close[s][w]);
        }
        windowRows.push(row);
      }
      const labelRow = [];
      for (const s of symbols) {
        const baseClose = normResult.close[s][t];
        for (let offset = 1; offset <= this.forecastHorizon; offset++) {
          const futureClose = normResult.close[s][t + offset];
          const bit = futureClose > baseClose ? 1 : 0;
          labelRow.push(bit);
        }
      }
      X.push(windowRows);
      y.push(labelRow);
    }
    return { X, y };
  }
}
