let model;
let lossChart, accuracyChart, confusionChart, perClassChart;

document.getElementById('load-data').addEventListener('click', async () => {
    await data.load();
    alert('Data Loaded Successfully!');
});

document.getElementById('train-model').addEventListener('click', async () => {
    if (!data.trainImages) { alert('Please load data first'); return; }

    if (!model) {
        model = tf.sequential();

        model.add(tf.layers.conv2d({
            inputShape: [28, 28, 1],
            filters: 32,
            kernelSize: 3,
            activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
    }

    const { xs, ys } = data.getTrainData();
    const { xs: testXs, ys: testYs } = data.getTestData();

    lossChart = new Chart(document.getElementById('lossChart'), { type: 'line', data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: 'red', fill: false }] }});
    accuracyChart = new Chart(document.getElementById('accuracyChart'), { type: 'line', data: { labels: [], datasets: [{ label: 'Accuracy', data: [], borderColor: 'blue', fill: false }] }});

    await model.fit(xs, ys, {
        epochs: 5,
        batchSize: 64,
        validationData: [testXs, testYs],
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                const loss = logs.loss ?? 0;
                const acc = logs.acc ?? logs.acc ?? 0;

                lossChart.data.labels.push(epoch + 1);
                lossChart.data.datasets[0].data.push(loss.toFixed ? Number(loss.toFixed(4)) : loss);
                lossChart.update();

                accuracyChart.data.labels.push(epoch + 1);
                accuracyChart.data.datasets[0].data.push(acc.toFixed ? Number(acc.toFixed(4)) : acc);
                accuracyChart.update();

                await tf.nextFrame();
            }
        }
    });

    updateConfusionMatrix(testXs, testYs);
    updatePerClassAccuracy(testXs, testYs);
});

// --- Prediction ---
const inputCanvas = document.getElementById('inputCanvas');
const ctx = inputCanvas.getContext('2d');

document.getElementById('predict').addEventListener('click', async () => {
    if (!model) { alert('Train model first'); return; }

    const imageData = ctx.getImageData(0, 0, 28, 28);
    const input = tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imageData, 1)
            .reshape([1, 28, 28, 1])
            .div(255.0);
        return tensor;
    });

    const prediction = model.predict(input);
    const predictedValue = prediction.argMax(1).dataSync()[0];
    document.getElementById('predictionResult').innerText = `Predicted: ${predictedValue}`;
    input.dispose();
});

// --- Confusion Matrix ---
function updateConfusionMatrix(xs, ys) {
    const preds = model.predict(xs).argMax(-1);
    const labels = ys.argMax(-1);

    const cm = Array.from({ length: 10 }, () => Array(10).fill(0));
    const predVals = preds.dataSync();
    const labelVals = labels.dataSync();

    for (let i = 0; i < labelVals.length; i++) {
        cm[labelVals[i]][predVals[i]]++;
    }

    if (confusionChart) confusionChart.destroy();
    confusionChart = new Chart(document.getElementById('confusionChart'), {
        type: 'bar',
        data: {
            labels: [...Array(10).keys()],
            datasets: cm.map((row, i) => ({ label: `Label ${i}`, data: row }))
        }
    });

    preds.dispose();
    labels.dispose();
}

// --- Per-class Accuracy ---
function updatePerClassAccuracy(xs, ys) {
    const preds = model.predict(xs).argMax(-1);
    const labels = ys.argMax(-1);

    const predVals = preds.dataSync();
    const labelVals = labels.dataSync();

    const correct = Array(10).fill(0);
    const total = Array(10).fill(0);

    for (let i = 0; i < labelVals.length; i++) {
        total[labelVals[i]]++;
        if (predVals[i] === labelVals[i]) correct[labelVals[i]]++;
    }

    const perClassAcc = correct.map((c, i) => total[i] ? c / total[i] : 0);

    if (perClassChart) perClassChart.destroy();
    perClassChart = new Chart(document.getElementById('perClassChart'), {
        type: 'bar',
        data: {
            labels: [...Array(10).keys()],
            datasets: [{ label: 'Per-class Accuracy', data: perClassAcc.map(a => a.toFixed(4)), backgroundColor: 'green' }]
        }
    });

    preds.dispose();
    labels.dispose();
}
