// =======================
// Global Variables
// =======================
let trainData = [];
let testData = [];
let processedTrain = null;
let processedTest = null;
let model = null;
let threshold = 0.5;
let addFamily = false;

// =======================
// Load Data
// =======================
function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    if (!trainFile || !testFile) {
        alert("Please choose both train.csv and test.csv files!");
        return;
    }

    Papa.parse(trainFile, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        quoteChar: '"',          // ✅ Fix CSV comma-escape issue
        escapeChar: '"',
        complete: res => {
            trainData = res.data;
            document.getElementById('data-status').innerText = `Train data loaded: ${trainData.length} rows.`;
            enableButtons();
        },
        error: err => alert("Train CSV load error: " + err)
    });

    Papa.parse(testFile, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        quoteChar: '"',          // ✅ Fix CSV comma-escape issue
        escapeChar: '"',
        complete: res => {
            testData = res.data;
            document.getElementById('data-status').innerText += ` Test data loaded: ${testData.length} rows.`;
            enableButtons();
        },
        error: err => alert("Test CSV load error: " + err)
    });
}

function enableButtons() {
    if (trainData.length && testData.length) {
        document.getElementById('inspect-btn').disabled = false;
    }
}

// =======================
// Inspect Data
// =======================
function inspectData() {
    // Preview head
    const previewHead = trainData.slice(0, 5);
    let html = "<table><tr>";
    Object.keys(previewHead[0]).forEach(col => html += `<th>${col}</th>`);
    html += "</tr>";
    previewHead.forEach(row => {
        html += "<tr>";
        Object.values(row).forEach(v => html += `<td>${v}</td>`);
        html += "</tr>";
    });
    html += "</table>";
    document.getElementById('data-preview').innerHTML = html;

    // Show chart survival by Sex
    const maleSurvived = trainData.filter(r => r.Sex === 'male' && r.Survived === 1).length;
    const maleDied = trainData.filter(r => r.Sex === 'male' && r.Survived === 0).length;
    const femaleSurvived = trainData.filter(r => r.Sex === 'female' && r.Survived === 1).length;
    const femaleDied = trainData.filter(r => r.Sex === 'female' && r.Survived === 0).length;

    const chartData = [
        { index: 'Male Survived', value: maleSurvived },
        { index: 'Male Died', value: maleDied },
        { index: 'Female Survived', value: femaleSurvived },
        { index: 'Female Died', value: femaleDied }
    ];

    tfvis.render.barchart(
        { name: 'Survival by Sex', tab: 'Charts' },
        chartData,
        { xLabel: 'Group', yLabel: 'Count' }
    );

    document.getElementById('preprocess-btn').disabled = false;
}

// =======================
// Preprocessing
// =======================
function preprocessData() {
    addFamily = document.getElementById('add-family-features').checked;

    // Compute median Age, Fare
    const ages = trainData.map(r => r.Age).filter(a => !isNaN(a));
    const fares = trainData.map(r => r.Fare).filter(f => !isNaN(f));
    const median = arr => {
        const s = arr.slice().sort((a, b) => a - b);
        return s[Math.floor(s.length / 2)];
    };
    const medianAge = median(ages);
    const medianFare = median(fares);

    const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const std = (arr, m) => Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
    const meanAge = mean(ages), stdAge = std(ages, meanAge);
    const meanFare = mean(fares), stdFare = std(fares, meanFare);

    function transform(row, isTrain) {
        const age = isNaN(row.Age) ? medianAge : row.Age;
        const fare = isNaN(row.Fare) ? medianFare : row.Fare;
        const features = [
            row.Pclass,
            row.Sex === 'female' ? 1 : 0,
            (age - meanAge) / stdAge,
            row.SibSp,
            row.Parch,
            (fare - meanFare) / stdFare,
            row.Embarked === 'C' ? 1 : (row.Embarked === 'Q' ? 2 : 0)
        ];
        if (addFamily) {
            const familySize = row.SibSp + row.Parch + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            features.push(familySize);
            features.push(isAlone);
        }
        return {
            x: features,
            y: isTrain ? row.Survived : null
        };
    }

    processedTrain = trainData.map(r => transform(r, true));
    processedTest = testData.map(r => transform(r, false));

    document.getElementById('preprocessing-output').innerText = `Preprocessing done. Features per sample: ${processedTrain[0].x.length}`;
    document.getElementById('create-model-btn').disabled = false;
}

// =======================
// Model
// =======================
function createModel() {
    const inputDim = processedTrain[0].x.length;
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputDim] }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    tfvis.show.modelSummary({ name: 'Model Summary', tab: 'Model' }, model);
    document.getElementById('train-btn').disabled = false;
}

// =======================
// Training
// =======================
async function trainModel() {
    const xs = tf.tensor2d(processedTrain.map(r => r.x));
    const ys = tf.tensor2d(processedTrain.map(r => [r.y]));

    const callbacks = tfvis.fitCallbacks(
        { name: 'Training Performance', tab: 'Model' },
        ['loss', 'val_loss', 'acc', 'val_acc'],
        { callbacks: ['onEpochEnd'] } // ✅ Fix error this.getMonitorValue
    );

    try {
        await model.fit(xs, ys, {
            epochs: 50,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks
        });
        document.getElementById('training-status').innerText = "Training complete!";
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('predict-btn').disabled = false;
    } catch (err) {
        document.getElementById('training-status').innerText = "Training stopped or error: " + err.message;
    }
}

// =======================
// Evaluation
// =======================
function updateMetrics() {
    threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').innerText = threshold.toFixed(2);
    // Compute metrics (confusion matrix, etc.) if needed
}

// =======================
// Prediction
// =======================
function predict() {
    if (!model || !processedTest) {
        alert("Model not trained or test data not loaded.");
        return;
    }

    const xs = tf.tensor2d(processedTest.map(r => r.x));
    const probs = model.predict(xs).dataSync();
    testData.forEach((r, i) => {
        r.Survived = probs[i] > threshold ? 1 : 0;
    });

    document.getElementById('prediction-output').innerText = "Prediction done!";
    document.getElementById('export-btn').disabled = false;
}

// =======================
// Export Results
// =======================
function exportResults() {
    let csv = "PassengerId,Survived\n";
    testData.forEach(r => {
        csv += `${r.PassengerId},${r.Survived}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'submission.csv';
    a.click();
    document.getElementById('export-status').innerText = "Results exported!";
}
