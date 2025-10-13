// app.js
// ES6 module connecting UI, DataLoader, and GRUModel.
// Default export App class. Also provides App.initFromDOM() for easy startup.
//
// Expected DOM elements (by id):
//   csv-file, btn-load, btn-train, btn-download, progress, status,
//   accuracy-container, timeline-container
//
// Index.html should include: <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.12.0/dist/tf.min.js"></script>
// and: <script type="module" src="./app.js"></script>

import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

export default class App {
  constructor(ui = {}) {
    this.ui = ui;
    this._attachElements();
    this.dataLoader = new DataLoader({ windowSize: 12, forecastHorizon: 3, trainSplit: 0.8 });
    this.model = null;
    this.tensors = null;
    this.currentPreds = null;
    this._wireActions();
  }

  _attachElements() {
    const ids = ['csv-file','btn-load','btn-train','btn-download','progress','status','accuracy-container','timeline-container'];
    for (const id of ids) {
      if (!this.ui[id]) this.ui[id] = document.getElementById(id);
    }
    // fallback simple progress element if not <progress>
    if (this.ui.progress && this.ui.progress.tagName !== 'PROGRESS') {
      this.ui.progress.value = 0;
    }
  }

  _wireActions() {
    if (this.ui['btn-load']) this.ui['btn-load'].addEventListener('click', () => this.onLoad());
    if (this.ui['btn-train']) this.ui['btn-train'].addEventListener('click', () => this.onTrain());
    if (this.ui['btn-download']) this.ui['btn-download'].addEventListener('click', () => this.onSave());
  }

  _setStatus(msg, isError = false) {
    if (this.ui.status) {
      this.ui.status.innerText = msg;
      this.ui.status.style.color = isError ? 'red' : 'black';
    } else console.log(msg);
  }

  _setProgress(frac) {
    const el = this.ui.progress;
    if (!el) return;
    const pct = Math.max(0, Math.min(1, frac)) * 100;
    if (el.tagName === 'PROGRESS') {
      el.value = pct;
    } else {
      el.style.width = pct + '%';
      el.innerText = `${Math.round(pct)}%`;
    }
  }

  _setBusy(flag) {
    if (this.ui['btn-load']) this.ui['btn-load'].disabled = flag;
    if (this.ui['btn-train']) this.ui['btn-train'].disabled = flag;
    if (this.ui['csv-file']) this.ui['csv-file'].disabled = flag;
  }

  async onLoad() {
    try {
      const fileEl = this.ui['csv-file'];
      if (!fileEl || !fileEl.files || fileEl.files.length === 0) {
        this._setStatus('Please select a CSV file first.', true);
        return;
      }
      this._setBusy(true);
      this._setStatus('Loading CSV...');
      // Dispose previous resources
      this._disposeAll();

      const file = fileEl.files[0];
      this.tensors = await this.dataLoader.loadFile(file);
      this._setStatus(`Loaded data. Symbols: ${this.tensors.symbols.join(', ')}`);
      this._setProgress(0);
      if (this.ui['btn-train']) this.ui['btn-train'].disabled = false;
      this._setBusy(false);
    } catch (err) {
      console.error(err);
      this._setStatus('Error loading CSV: ' + err.message, true);
      this._setBusy(false);
    }
  }

  async onTrain() {
    try {
      if (!this.tensors) { this._setStatus('No data loaded.', true); return; }
      this._setBusy(true);
      this._setStatus('Building model...');
      // Dispose old model
      if (this.model) { this.model.dispose(); this.model = null; }
      const inputShape = [this.tensors.X_train.shape[1], this.tensors.X_train.shape[2]];
      this.model = new GRUModel({
        inputShape,
        units: 96,
        convFilters: 48,
        bidirectional: true,
        dropout: 0.25,
        learningRate: 0.001
      });
      this.model.build();

      this._setStatus('Training...');
      const epochs = 25;
      const batchSize = Math.max(8, Math.min(64, Math.floor(this.tensors.X_train.shape[0] / 10) || 32));
      const onEpochEnd = (epoch, logs) => {
        const loss = logs.loss ? logs.loss.toFixed(4) : '-';
        const valLoss = logs.val_loss ? logs.val_loss.toFixed(4) : '-';
        const acc = (logs.binaryAccuracy || logs.binary_accuracy) ? (logs.binaryAccuracy || logs.binary_accuracy).toFixed(4) : '-';
        this._setStatus(`Epoch ${epoch+1}/${epochs} loss:${loss} val_loss:${valLoss} acc:${acc}`);
        this._setProgress((epoch + 1) / epochs);
      };

      await this.model.fit(this.tensors.X_train, this.tensors.y_train, {
        epochs,
        batchSize,
        validationSplit: 0.1,
        onEpochEnd
      });

      this._setStatus('Training finished. Evaluating on test set...');
      await this._evaluate();
      this._setBusy(false);
    } catch (err) {
      console.error(err);
      this._setStatus('Training error: ' + err.message, true);
      this._setBusy(false);
    }
  }

  async _evaluate() {
    try {
      if (!this.model || !this.tensors) throw new Error('Model or data missing.');
      // Predict (synchronous)
      if (this.currentPreds) { this.currentPreds.dispose(); this.currentPreds = null; }
      const preds = this.model.predict(this.tensors.X_test);
      this.currentPreds = preds.clone ? preds.clone() : preds; // keep a copy
      // Compute per-stock metrics
      const { perStock, overallAccuracy } = await GRUModel.computePerStockAccuracy(this.currentPreds, this.tensors.y_test, this.tensors.symbols, 0.5);
      // Sort by accuracy desc
      const sorted = perStock.slice().sort((a,b) => b.accuracy - a.accuracy);
      this._renderAccuracy(sorted);
      await this._renderTimelines(this.currentPreds, this.tensors.y_test, this.tensors.symbols, sorted);
      this._setStatus(`Evaluation done. Overall avg per-stock accuracy ${(overallAccuracy*100).toFixed(2)}%`);
      this._setProgress(1);
    } catch (err) {
      console.error(err);
      this._setStatus('Evaluation error: ' + err.message, true);
    }
  }

  _renderAccuracy(sorted) {
    const container = this.ui['accuracy-container'];
    if (!container) return;
    container.innerHTML = '';
    const title = document.createElement('h3'); title.innerText = 'Per-stock accuracy (avg over 3 days)'; container.appendChild(title);
    for (const s of sorted) {
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.alignItems = 'center';
      row.style.margin = '6px 0';
      const label = document.createElement('div');
      label.style.width = '140px';
      label.innerText = s.symbol;
      const barWrap = document.createElement('div');
      barWrap.style.flex = '1';
      barWrap.style.background = '#eee';
      barWrap.style.height = '18px';
      barWrap.style.borderRadius = '4px';
      barWrap.style.overflow = 'hidden';
      const bar = document.createElement('div');
      bar.style.height = '100%';
      bar.style.width = `${(s.accuracy*100).toFixed(2)}%`;
      bar.style.background = '#2196f3';
      bar.style.transition = 'width 0.4s';
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
    const container = this.ui['timeline-container'];
    if (!container) return;
    container.innerHTML = '';
    const title = document.createElement('h3'); title.innerText = 'Per-stock timelines (test set, green=good)'; container.appendChild(title);
    const predsData = await predTensor.data();
    const labelsData = await yTensor.data();
    const samples = yTensor.shape[0];
    const numStocks = symbols.length;
    const horizon = 3;

    for (const sObj of sortedStocks) {
      const s = sObj.symbol;
      const idx = symbols.indexOf(s);
      const wrapper = document.createElement('div');
      wrapper.style.margin = '8px 0';
      const header = document.createElement('div');
      header.innerText = `${s} — ${(sObj.accuracy*100).toFixed(2)}% (tp:${sObj.tp} tn:${sObj.tn} fp:${sObj.fp} fn:${sObj.fn})`;
      header.style.fontSize = '12px';
      header.style.marginBottom = '4px';
      wrapper.appendChild(header);

      const timeline = document.createElement('div');
      timeline.style.display = 'flex';
      timeline.style.overflowX = 'auto';
      timeline.style.border = '1px solid #ddd';
      timeline.style.padding = '4px';
      timeline.style.background = '#fafafa';
      timeline.style.height = '28px';

      for (let i = 0; i < samples; i++) {
        let correctCount = 0;
        for (let h = 0; h < horizon; h++) {
          const flatIdx = i * (numStocks * horizon) + idx * horizon + h;
          const p = predsData[flatIdx] >= 0.5 ? 1 : 0;
          const l = labelsData[flatIdx] >= 0.5 ? 1 : 0;
          if (p === l) correctCount++;
        }
        const block = document.createElement('div');
        block.style.width = '6px';
        block.style.height = '100%';
        block.style.marginRight = '1px';
        if (correctCount === 3) block.style.background = '#1b5e20';
        else if (correctCount === 2) block.style.background = '#4caf50';
        else if (correctCount === 1) block.style.background = '#ffb74d';
        else block.style.background = '#c62828';
        timeline.appendChild(block);
      }

      wrapper.appendChild(timeline);
      container.appendChild(wrapper);
    }
  }

  async onSave() {
    try {
      if (!this.model) { this._setStatus('No model to save.', true); return; }
      await this.model.save('downloads://gru_stock_model');
      this._setStatus('Model download started.');
    } catch (err) {
      console.error(err);
      this._setStatus('Save failed: ' + err.message, true);
    }
  }

  _disposeAll() {
    try {
      if (this.tensors) {
        ['X_train','X_test','y_train','y_test'].forEach(k => {
          if (this.tensors[k]) { this.tensors[k].dispose(); this.tensors[k] = null; }
        });
        this.tensors = null;
      }
      if (this.currentPreds) { this.currentPreds.dispose(); this.currentPreds = null; }
      if (this.model) { this.model.dispose(); this.model = null; }
    } catch (e) { console.warn('Dispose error', e); }
  }

  dispose() { this._disposeAll(); }

  // Convenience initializer when DOM is ready
  static initFromDOM() {
    const app = new App();
    return app;
  }
}

// Auto-init is not performed; index.html should import and call App.initFromDOM()
