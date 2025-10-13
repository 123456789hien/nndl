// gru.js
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

class GRUModel {
    constructor(inputShape, outputSize) {
        this.model = null;
        this.inputShape = inputShape;
        this.outputSize = outputSize;
        this.history = null;
    }

    buildModel() {
        const input = tf.input({ shape: this.inputShape });

        // CNN layer to improve feature extraction
        const conv = tf.layers.conv1d({
            filters: 32,
            kernelSize: 3,
            padding: 'causal',
            activation: 'relu'
        }).apply(input);

        // GRU layer with resetAfter=false
        const gru1 = tf.layers.gru({
            units: 64,
            returnSequences: false,
            resetAfter: false,
            dropout: 0.2,
            recurrentDropout: 0.1
        }).apply(conv);

        const dense1 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(gru1);
        const output = tf.layers.dense({ units: this.outputSize, activation: 'sigmoid' }).apply(dense1);

        this.model = tf.model({ inputs: input, outputs: output });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });

        console.log('✅ GRU + CNN model built');
        this.model.summary();

        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 50, batchSize = 32) {
        if (!this.model) this.buildModel();

        this.history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    const status = `Epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(4)}, acc: ${(logs.binaryAccuracy*100).toFixed(2)}%, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${(logs.val_binaryAccuracy*100).toFixed(2)}%`;
                    console.log(status);
                    const progressElement = document.getElementById('trainingProgress');
                    const statusElement = document.getElementById('status');
                    if (progressElement) progressElement.value = ((epoch+1)/epochs)*100;
                    if (statusElement) statusElement.textContent = status;
                    await tf.nextFrame();
                }
            }
        });

        return this.history;
    }

    async predict(X) {
        if (!this.model) throw new Error('Model not trained');
        return this.model.predict(X);
    }

    evaluatePerStock(yTrue, yPred, symbols, horizon = 3) {
        const yTrueArray = yTrue.arraySync();
        const yPredArray = yPred.arraySync();
        const numStocks = symbols.length;

        const stockAccuracies = {};
        const stockPredictions = {};

        symbols.forEach((symbol, stockIdx) => {
            let correct = 0;
            let total = 0;
            const predictions = [];

            for (let i = 0; i < yTrueArray.length; i++) {
                for (let offset = 0; offset < horizon; offset++) {
                    const targetIdx = stockIdx * horizon + offset;
                    const trueVal = yTrueArray[i][targetIdx];
                    const predVal = yPredArray[i][targetIdx] > 0.5 ? 1 : 0;

                    if (trueVal === predVal) correct++;
                    total++;

                    predictions.push({ true: trueVal, pred: predVal, correct: trueVal===predVal });
                }
            }

            stockAccuracies[symbol] = correct / total;
            stockPredictions[symbol] = predictions;
        });

        return { stockAccuracies, stockPredictions };
    }

    dispose() {
        if (this.model) this.model.dispose();
    }
}

export default GRUModel;
