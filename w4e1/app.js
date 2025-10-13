// app.js
// ES6 module that wires the UI (index.html) to DataLoader and GRUModel,
// manages training flow, shows progress, and renders visualizations.
// Usage:
// - index.html should import this module as a module and call App.init() or similar.
// - This file assumes tf.js is available globally as window.tf
//
// Visualizations implemented with plain DOM/CSS to avoid third-party libs.
// Clean error handling and memory disposal implemented.

import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

export default class App {
  constructor(ui) {
    // ui: object mapping element ids to DOM elements. If omitted, query by defaults.
    this.ui = ui || {};
    this._bindUI();
    this.dataLoader = new DataLoader({ windowSize: 12, forecastHorizon: 3, trainSplit: 0.8 });
    this.model = null;
    this.tensors = null;
    this.currentPreds = null;
  }

  _bindUI() {
    // Find elements by ID if not provided
    this.ui.fileInput = this.ui.fileInput || document.getElementById('csv-file');
    this.ui.loadBtn = this.ui.loadBtn || document.getElementById('btn-load');
    this.ui.trainBtn = this.ui.trainBtn || document.getElementById('btn-train');
    this.ui.progress = this.ui.progress || document.getElementById('progress');
    this.ui.status = this.ui.status || document.getElementById('status');
    this.ui.accuracyContainer = this.ui.accuracyContainer || document.getElementById('accuracy-container');
    this.ui.timelineContainer = this.ui.timelineContainer || document.getElementById('timeline-container');
    this.ui.downloadBtn = this.ui.downloadBtn || document.getElementById('btn-download');

    if (this.ui.loadBtn) this.ui.loadBtn.addEventListener('click', () => this._onLoadClick());
    if (this.ui.trainBtn) this.ui.trainBtn.addEventListener('click', () => this._onTrainClick());
    if (this.ui.downloadBtn) this.ui.downloadBtn.addEventListener('click', () => this._onSaveModel());
  }

  async _onLoadClick() {
    try {
      const fileEl = this.ui.fileInput;
      if (!fileEl || !fileEl.files || fileEl.files.length === 0) {
        this._setStatus('Please choose a CSV file first.', true);
        return;
      }
      this._setStatus('Loading CSV and preparing data...');
      const file = fileEl.files[0];
      // disable buttons while loading
      this._setBusy(true);
      // Dispose previous tensors
      this._disposeTensors();
      this.tensors = await this.dataLoader.loadFile(file);
      this._setStatus(`Loaded data. Symbols: ${this.tensors.symbols.join(', ')}`);
      this._setBusy(false);
      // Enable train button
      if (this.ui.trainBtn) this.ui.trainBtn.disabled = false;
    } catch (err) {
      this._setStatus('Error loading file: ' + err.message, true);
      this._setBusy(false);
      console.error(err);
    }
  }

  async _onTrainClick() {
    try {
      if (!this.tensors) { this._setStatus('No data loaded.', true); return; }
      this._setStatus('Building model...');
      this._setBusy(true);
      // Dispose previous model if any
      if (this.model) { this.model.dispose(); this.model = null; }
      // build model with inputShape derived from tensors
      const inputShape = [this.tensors.X_train.shape[1], this.tensors.X_train.shape[2]]; // [12,20]
      this.model = new GRUModel({
        inputShape,
        units: 96,
        convFilters: 48,
        bidirectional: true,
        dropout: 0.25,
        learningRate: 0.001
      });
      this.model.build();
      this._setStatus('Training model...');

      // Train and update progress UI via onEpochEnd
      const epochs = 25;
      const batchSize = Math.min(64, Math.max(8, Math.floor(this.tensors.X_train.shape[0] / 10)));
      const onEpochEnd = (epoch, logs) => {
        const msg = `Epoch ${epoch + 1}/${epochs} - loss:${(logs.loss||0).toFixed(4)} val_loss:${(logs.val_loss||0).toFixed(4)} acc:${((logs.binaryAccuracy||logs.binary_accuracy)||0).toFixed(4)}`;
        this._setStatus(msg);
        this._updateProgress((epoch + 1) / epochs);
      };
      await this.model.fit(this.tensors.X_train, this.tensors.y_train, {
        epochs,
        batchSize,
        validationSplit: 0.1,
        onEpochEnd
      });

      this._setStatus('Model trained. Running evaluation...');
      // Predict on test set
      await this._evaluateAndVisualize();
      this._setStatus('Evaluation complete.');
      this._setBusy(false);
    } catch (err) {
      this._setStatus('Training error: ' + err.message, true);
      console.error(err);
      this._setBusy(false);
    }
  }

  async _evaluateAndVisualize() {
    if (!this.model || !this.tensors) throw new Error('Model or data missing.');
    // Free previous prediction tensor if exists
    if (this.currentPreds) { this.currentPreds.dispose(); this.currentPreds = null; }
    const preds = await this.model.predict(this.tensors.X_test);
    this.currentPreds = preds.clone(); // keep a copy
    // compute per-stock accuracy using utility
    const result = await GRUModel.computePerStockAccuracy(preds, this.tensors.y_test, this.tensors.symbols, 0.5);
    // Sort stocks by accuracy descending
    const perStock = result.perStockAccuracy.slice().sort((a,b)=>b.accuracy - a.accuracy);
    // Render accuracies
    this._renderAccuracyBars(perStock);
    // Render timelines for each stock (sorted by accuracy)
    await this._renderTimelines(preds, this.tensors.y_test, this.tensors.symbols, perStock);
    // Dispose preds here? we keep currentPreds for potential further inspection
  }

  _renderAccuracyBars(sortedStocks) {
    // Clear container
    const container = this.ui.accuracyContainer;
    if (!container) return;
    container.innerHTML = '';
    const header = document.createElement('h3');
    header.innerText = 'Per-stock accuracy (averaged over 3 days)';
    container.appendChild(header);

    // For each stock create a horizontal bar
    for (const s of sortedStocks) {
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.alignItems = 'center';
      row.style.margin = '6px 0';
      const label = document.createElement('div');
      label.style.width = '140px';
      label.innerText = `${s.symbol}`;
      const barWrap = document.createElement('div');
      barWrap.style.flex = '1';
      barWrap.style.background = '#eee';
      barWrap.style.height = '18px';
      barWrap.style.borderRadius = '4px';
      barWrap.style.overflow = 'hidden';
      const bar = document.createElement('div');
      bar.style.height = '100%';
      bar.style.width = `${(s.accuracy*100).toFixed(2)}%`;
      bar.style.background = '#4caf50';
      bar.style.transition = 'width 0.5s';
      barWrap.appendChild(bar);
      const pct = document.createElement('div');
      pct.style.width = '70px';
      pct.style.textAlign = 'right';
      pct.innerText = `${(s.accuracy*100).toFixed(2)}%`;
      row.appendChild(label);
      row.appendChild(barWrap);
      row.appendChild(pct);
      container.appendChild(row);
    }
  }

  async _renderTimelines(predTensor, yTensor, symbols, sortedStocks) {
    // Render per-stock timeline: green for correct, red for wrong
    const container = this.ui.timelineContainer;
    if (!container) return;
    container.innerHTML = '';
    const header = document.createElement('h3');
    header.innerText = 'Per-stock prediction timelines (test set, sorted by accuracy)';
    container.appendChild(header);

    // Retrieve arrays
    const predsArr = Array.from(await predTensor.data()); // length = samples * 30
    const labelsArr = Array.from(await yTensor.data());
    const samples = yTensor.shape[0];
    const numStocks = symbols.length;
    const horizon = 3;

    // For each stock in sortedStocks order, build a timeline row with one block per sample.
    for (const sObj of sortedStocks) {
      const s = sObj.symbol;
      const idx = symbols.indexOf(s);
      const rowDiv = document.createElement('div');
      rowDiv.style.margin = '8px 0';
      const title = document.createElement('div');
      title.innerText = `${s} — accuracy ${(sObj.accuracy*100).toFixed(2)}% (tp:${sObj.tp} tn:${sObj.tn} fp:${sObj.fp} fn:${sObj.fn})`;
      title.style.fontSize = '12px';
      title.style.marginBottom = '4px';
      rowDiv.appendChild(title);

      const timeline = document.createElement('div');
      timeline.style.display = 'flex';
      timeline.style.flexWrap = 'nowrap';
      timeline.style.overflowX = 'auto';
      timeline.style.border = '1px solid #ddd';
      timeline.style.padding = '4px';
      timeline.style.background = '#fafafa';
      timeline.style.height = '28px';

      for (let i = 0; i < samples; i++) {
        // For each sample, aggregate correctness over the 3 horizon days: mark green if majority correct
        let correctCount = 0;
        for (let h = 0; h < horizon; h++) {
          const flatIdx = i * (numStocks * horizon) + idx * horizon + h;
          const p = predsArr[flatIdx] >= 0.5 ? 1 : 0;
          const l = labelsArr[flatIdx];
          if (p === l) correctCount++;
        }
        const block = document.createElement('div');
        block.style.width = '6px';
        block.style.height = '100%';
        block.style.marginRight = '1px';
        // color scale: all correct -> dark green, majority correct -> light green, tie/1 correct -> orange, 0 correct -> red
        if (correctCount === 3) block.style.background = '#166534'; // dark green
        else if (correctCount === 2) block.style.background = '#4caf50'; // green
        else if (correctCount === 1) block.style.background = '#ff9800'; // orange
        else block.style.background = '#d32f2f'; // red
        timeline.appendChild(block);
      }
      rowDiv.appendChild(timeline);
      container.appendChild(rowDiv);
    }
  }

  async _onSaveModel() {
    try {
      if (!this.model || !this.model.model) { this._setStatus('No model to save.', true); return; }
      // Save to browser download (IndexedDB or localstorage schemes); using downloads via tfjs is allowed:
      await this.model.save('downloads://gru_stock_model');
      this._setStatus('Model saved (download started).');
    } catch (err) {
      this._setStatus('Failed to save model: ' + err.message, true);
    }
  }

  _updateProgress(frac) {
    if (!this.ui.progress) return;
    const pct = Math.max(0, Math.min(1, frac)) * 100;
    this.ui.progress.value = pct;
    this.ui.progress.innerText = `${Math.round(pct)}%`;
  }

  _setStatus(msg, isError = false) {
    if (this.ui.status) {
      this.ui.status.innerText = msg;
      this.ui.status.style.color = isError ? 'red' : 'black';
    } else {
      console.log(msg);
    }
  }

  _setBusy(flag) {
    if (this.ui.loadBtn) this.ui.loadBtn.disabled = flag;
    if (this.ui.trainBtn) this.ui.trainBtn.disabled = flag;
    if (this.ui.fileInput) this.ui.fileInput.disabled = flag;
  }

  _disposeTensors() {
    if (!this.tensors) return;
    try {
      const keys = ['X_train','X_test','y_train','y_test'];
      for (const k of keys) {
        if (this.tensors[k]) { this.tensors[k].dispose(); this.tensors[k] = null; }
      }
    } catch (err) {
      console.warn('Error disposing tensors', err);
    }
    this.tensors = null;
  }

  // Call this when navigating away to clean up model and tensors
  dispose() {
    this._disposeTensors();
    if (this.model) { this.model.dispose(); this.model = null; }
    if (this.currentPreds) { this.currentPreds.dispose(); this.currentPreds = null; }
  }

  // Static helper to initialize app from default DOM ids
  static initFromDOM() {
    const app = new App();
    return app;
  }
}

// Auto init if script loaded directly as module and DOM contains expected elements
if (typeof window !== 'undefined' && document.readyState === 'complete') {
  // no automatic creation to avoid interfering; user should call App.initFromDOM() in index.html
}
