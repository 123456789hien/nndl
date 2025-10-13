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
    const y_test_arr = samples.y.slic
