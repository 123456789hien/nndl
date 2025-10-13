// gru.js
// ES6 module: default export GRUModel
// Builds and compiles a GRU-based TF.js model for multi-output binary classification.
// Output size: numStocks * horizon (assumed 10*3 = 30 for the user's dataset).
//
// Exports: GRUModel class with methods: build(), fit(), predict(), evaluate(), save(), load(), dispose()
// Also includes static async computePerStockAccuracy(predicted, yTrue, symbols, threshold)

export default class GRUModel {
  constructor(opts = {}) {
    this.inputShape = opts.inputShape || [12, 20]; // [timeSteps, features]
    this.units = opts.units || 64;
    this.convFilters = opts.convFilters || 32;
    this.convKernel = opts.convKernel || 3;
    this.bidirectional = opts.bidirectional !== undefined ? opts.bidirectional : true;
    this.dropout = opts.dropout || 0.2;
    this.learningRate = opts.learningRate || 0.001;
    this.model = null;
  }

  build() {
    const tf = window.tf;
    const [timeSteps, features] = this.inputShape;
    const input = tf.input({ shape: [timeSteps, features] });

    // Conv1D temporal extractor to boost local pattern learning
    let x = tf.layers.conv1d({
      filters: this.convFilters,
      kernelSize: this.convKernel,
      padding: 'causal',
      activation: 'relu',
      strides: 1
    }).apply(input);

    x = tf.layers.batchNormalization().apply(x);

    const gru1 = tf.layers.gru({
      units: this.units,
      returnSequences: true,
      dropout: this.dropout,
      recurrentActivation: 'sigmoid',
      resetAfter: true
    });

    const gru2 = tf.layers.gru({
      units: Math.max(32, Math.floor(this.units / 2)),
      returnSequences: false,
      dropout: Math.max(0.1, this.dropout / 2),
      recurrentActivation: 'sigmoid',
      resetAfter: true
    });

    if (this.bidirectional) {
      x = tf.layers.bidirectional({ layer: gru1, mergeMode: 'concat' }).apply(x);
      x = tf.layers.bidirectional({ layer: gru2, mergeMode: 'concat' }).apply(x);
    } else {
      x = gru1.apply(x);
      x = gru2.apply(x);
    }

    x = tf.layers.dropout({ rate: this.dropout }).apply(x);

    // Dense head: output 30 sigmoids (for 10 stocks * 3 days)
    const outputUnits = 30;
    const outputs = tf.layers.dense({ units: outputUnits, activation: 'sigmoid' }).apply(x);

    this.model = tf.model({ inputs: input, outputs: outputs });

    const optimizer = tf.train.adam(this.learningRate);
    this.model.compile({
      optimizer,
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });

    return this.model;
  }

  // fitParams: { epochs, batchSize, validationSplit, onEpochEnd }
  async fit(X_train, y_train, fitParams = {}) {
    if (!this.model) this.build();
    const epochs = fitParams.epochs || 20;
    const batchSize = fitParams.batchSize || 32;
    const validationSplit = fitParams.validationSplit || 0.1;
    const onEpochEnd = fitParams.onEpochEnd || (() => {});
    const history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationSplit,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          try { await onEpochEnd(epoch, logs); } catch (e) { console.warn(e); }
          await window.tf.nextFrame();
        },
        onBatchEnd: async () => { await window.tf.nextFrame(); }
      }
    });
    return history;
  }

  // Predict returns tf.Tensor [samples, 30]
  predict(X) {
    if (!this.model) throw new Error('Model not built.');
    const preds = this.model.predict(X);
    return preds;
  }

  async evaluate(X, y) {
    if (!this.model) throw new Error('Model not built.');
    const evalOut = await this.model.evaluate(X, y, { batchSize: 32 });
    // evalOut can be scalar or array; convert to numbers
    if (Array.isArray(evalOut)) {
      return evalOut.map(t => (t && t.dataSync ? t.dataSync()[0] : t));
    } else {
      return evalOut.dataSync ? evalOut.dataSync()[0] : evalOut;
    }
  }

  async save(path = 'downloads://gru_stock_model') {
    if (!this.model) throw new Error('Model not built.');
    return await this.model.save(path);
  }

  async load(path) {
    this.model = await window.tf.loadLayersModel(path);
    return this.model;
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }

  // Compute per-stock accuracy and confusion metrics
  // predicted: tf.Tensor [samples, 30] (probabilities)
  // yTrue: tf.Tensor [samples, 30] (0/1 ints)
  // symbols: array of stock symbols length 10
  // returns: { perStock: [ {symbol, accuracy, tp,tn,fp,fn} ], overallAccuracy }
  static async computePerStockAccuracy(predicted, yTrue, symbols, threshold = 0.5) {
    const tf = window.tf;
    // Get arrays asynchronously (non-blocking)
    const predsData = await predicted.data(); // Float32Array
    const labelsData = await yTrue.data();    // Float32Array
    const samples = predicted.shape[0];
    const numStocks = symbols.length;
    const horizon = 3;
    const perStock = [];
    for (let s = 0; s < numStocks; s++) {
      let correct = 0;
      let total = 0;
      let tp = 0, tn = 0, fp = 0, fn = 0;
      for (let i = 0; i < samples; i++) {
        for (let h = 0; h < horizon; h++) {
          const idx = i * (numStocks * horizon) + s * horizon + h;
          const p = predsData[idx] >= threshold ? 1 : 0;
          const l = labelsData[idx] >= 0.5 ? 1 : 0;
          if (p === l) correct++;
          if (l === 1 && p === 1) tp++;
          if (l === 0 && p === 0) tn++;
          if (l === 0 && p === 1) fp++;
          if (l === 1 && p === 0) fn++;
          total++;
        }
      }
      const acc = total > 0 ? correct / total : 0;
      perStock.push({ symbol: symbols[s], accuracy: acc, tp, tn, fp, fn });
    }
    const overallAccuracy = perStock.reduce((sum, p) => sum + p.accuracy, 0) / perStock.length;
    return { perStock, overallAccuracy };
  }
}
