// gru.js
// ES6 module defining GRUModel class: builds, compiles, trains, predicts, evaluates a multi-output GRU model in TensorFlow.js
// Architecture (designed to improve performance vs a vanilla GRU):
// - Optional Conv1D temporal extractor (1D convolution over time) to learn local patterns
// - Bidirectional GRU stack (can be toggled)
// - Dense head -> output 30 units with sigmoid (binary for each stock/day)
// - Loss: binaryCrossentropy, metric: binaryAccuracy
//
// Exports: GRUModel class

export default class GRUModel {
  constructor(opts = {}) {
    // opts: inputShape: [timeSteps, features], units, convFilters, lr, bidirectional, dropout
    this.inputShape = opts.inputShape || [12, 20]; // default to (12,20)
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
    tf.util.assert(this.inputShape.length === 2, 'inputShape must be [timeSteps, features]');
    const timeSteps = this.inputShape[0];
    const features = this.inputShape[1];
    const input = tf.input({ shape: [timeSteps, features] });

    // Conv1D (temporal filtering) - help extract short-term patterns (improves performance vs raw GRU)
    // tf.layers.conv1d expects dataFormat 'channelsLast' with shape [timeSteps, features]
    let x = tf.layers.conv1d({
      filters: this.convFilters,
      kernelSize: this.convKernel,
      padding: 'causal', // causal to respect temporal order
      activation: 'relu',
      strides: 1
    }).apply(input);

    // Optional layer normalization-ish: batchNormalization
    x = tf.layers.batchNormalization().apply(x);

    // GRU stack
    const gru1 = tf.layers.gru({
      units: this.units,
      returnSequences: true,
      recurrentActivation: 'sigmoid',
      dropout: this.dropout,
      recurrentDropout: 0.0,
      resetAfter: true,
    });
    const gru2 = tf.layers.gru({
      units: Math.max(32, Math.floor(this.units / 2)),
      returnSequences: false,
      recurrentActivation: 'sigmoid',
      dropout: this.dropout / 2,
      recurrentDropout: 0.0,
      resetAfter: true,
    });

    if (this.bidirectional) {
      const bidi1 = tf.layers.bidirectional({ layer: gru1, mergeMode: 'concat' }).apply(x);
      // if bidi returns sequences, ensure any following GRU sees sequences
      // Need to expand dims to sequence -> convert bidi1 back to a sequence if necessary
      // Here, bidi1 returns sequences (returnSequences true), okay.
      x = bidi1;
      x = tf.layers.bidirectional({ layer: gru2, mergeMode: 'concat' }).apply(x);
    } else {
      x = gru1.apply(x);
      x = gru2.apply(x);
    }

    x = tf.layers.dropout({ rate: this.dropout }).apply(x);

    // Dense head
    // We must output 10 stocks * 3 days = 30 binary outputs
    const outputs = tf.layers.dense({ units: 30, activation: 'sigmoid' }).apply(x);

    this.model = tf.model({ inputs: input, outputs: outputs });

    // Optimizer with a small learning rate and Adam
    const optimizer = tf.train.adam(this.learningRate);

    this.model.compile({
      optimizer,
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });

    return this.model;
  }

  // Train with callbacks to report progress.
  // fitParams: { epochs, batchSize, onEpochEnd } where onEpochEnd(epoch, logs) will be called.
  async fit(X_train, y_train, fitParams = {}) {
    if (!this.model) this.build();
    const tf = window.tf;
    const epochs = fitParams.epochs || 20;
    const batchSize = fitParams.batchSize || 32;
    const validationSplit = fitParams.validationSplit || 0.1;
    const onEpochEnd = fitParams.onEpochEnd || (()=>{});
    const callbacks = {
      onEpochEnd: async (epoch, logs) => {
        try { await onEpochEnd(epoch, logs); } catch(e) { console.warn(e); }
        // allow TF.js to free memory used by logs
        await tf.nextFrame();
      },
      onBatchEnd: async (batch, logs) => { await tf.nextFrame(); }
    };
    const history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationSplit,
      callbacks
    });
    return history;
  }

  // Predict returns a tf.Tensor [samples, 30] of probabilities
  predict(X) {
    if (!this.model) throw new Error('Model not built. Call build() first or load a saved model.');
    // No grads required
    return tf.tidy(() => {
      const preds = this.model.predict(X);
      return preds; // Tensor left to caller to manage
    });
  }

  // Evaluate returns loss and binaryAccuracy using model.evaluate
  async evaluate(X, y) {
    if (!this.model) throw new Error('Model not built.');
    const evalOut = await this.model.evaluate(X, y, { batchSize: 32 });
    // evalOut can be scalar or array depending on metrics; convert to numbers
    const res = Array.isArray(evalOut) ? evalOut.map(r => (Array.isArray(r.dataSync) ? r.dataSync() : [r.dataSync ? r.dataSync() : r])) : evalOut;
    return res;
  }

  // Save model weights to local browser download (tfjs format)
  async save(name = 'localstorage://gru-stock-model') {
    if (!this.model) throw new Error('Model not built.');
    return await this.model.save(name);
  }

  // Load model from URL or localstorage path
  async load(path) {
    this.model = await tf.loadLayersModel(path);
    return this.model;
  }

  // Dispose model to free memory
  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }

  // Utility: compute per-stock accuracy from predicted tensor and y_true tensor
  // predicted: tf.Tensor [samples, 30] probabilities
  // yTrue: tf.Tensor [samples, 30] binary labels
  // returns JS object { perStockAccuracy: [10], perStockConfusion: [ {tp,tn,fp,fn}*10 ] , overallAccuracy }
  static async computePerStockAccuracy(predicted, yTrue, symbols, threshold = 0.5) {
    // predicted and yTrue are tf.Tensors
    const tf = window.tf;
    return tf.tidy(() => {
      const predsBin = predicted.greater(tf.scalar(threshold)).toInt(); // [samples,30]
      const labels = yTrue.toInt(); // [samples,30]
      const samples = predsBin.shape[0];
      const numStocks = symbols.length;
      const horizon = 3;
      const perStock = [];
      const predsArr = predsBin.dataSync();
      const labelsArr = labels.dataSync();
      // note: data is row-major: sample0:[30], sample1:[30], ...
      for (let s = 0; s < numStocks; s++) {
        let correct = 0;
        let total = 0;
        let tp=0, tn=0, fp=0, fn=0;
        for (let i = 0; i < samples; i++) {
          for (let h = 0; h < horizon; h++) {
            const idx = i * (numStocks * horizon) + s * horizon + h;
            const p = predsArr[idx];
            const l = labelsArr[idx];
            if (p === l) correct++;
            if (l === 1 && p === 1) tp++;
            if (l === 0 && p === 0) tn++;
            if (l === 0 && p === 1) fp++;
            if (l === 1 && p === 0) fn++;
            total++;
          }
        }
        const acc = total === 0 ? 0 : correct / total;
        perStock.push({ symbol: symbols[s], accuracy: acc, tp, tn, fp, fn });
      }
      const overallCorrect = perStock.reduce((sum, s) => sum + s.accuracy, 0) / numStocks;
      return { perStockAccuracy: perStock, overallAccuracy: overallCorrect };
    });
  }
}
