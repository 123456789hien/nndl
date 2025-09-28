/* app.js
   Integrated fixes and improvements based on instructor's original code.
   - Uses PapaParse for robust CSV parsing (handles quoted commas in Name).
   - Keeps original workflow and UI element IDs.
   - Fixes evaluation table rendering and ensures metrics update after training.
   - Implements stratified 80/20 split, early stopping, ROC/AUC plotting.
   - Exports submission.csv, probabilities.csv and saves the TFJS model.
*/

/* ========= Global variables ========= */
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;            // tf.tensor2d for validation features
let validationLabels = null;          // tf.tensor1d for validation labels
let validationProbsArray = null;      // Float32Array of predicted probabilities on validation set
let validationLabelsArray = null;     // numeric array of true labels (0/1)
let testPredictions = null;           // tensor (probabilities) for test set
let stopRequested = false;            // allow user to stop training

/* ========= Schema configuration (swap for other datasets if needed) ========= */
const TARGET_FEATURE = 'Survived';    // Binary target
const ID_FEATURE = 'PassengerId';     // Identifier
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];
// Note: If you change the dataset, update the lists above accordingly.

/* ========= Helper utilities ========= */

// Simple DOM getter
const $ = id => document.getElementById(id);

// Robust CSV parsing: accepts File or csvText string
function parseCSVInput(input) {
    // Returns Promise resolving to array of row objects
    return new Promise((resolve, reject) => {
        const papaOptions = {
            header: true,
            dynamicTyping: true,   // convert numeric-looking fields to numbers
            skipEmptyLines: true,
            quoteChar: '"',        // important: handle quoted fields with commas
            escapeChar: '"',
            transformHeader: h => h.trim()
        };

        if (input instanceof File) {
            Papa.parse(input, {
                ...papaOptions,
                complete: results => {
                    if (results.errors && results.errors.length) {
                        console.warn('PapaParse warnings/errors:', results.errors);
                    }
                    resolve(results.data);
                },
                error: err => reject(err)
            });
        } else if (typeof input === 'string') {
            Papa.parse(input, {
                ...papaOptions,
                complete: results => resolve(results.data),
                error: err => reject(err)
            });
        } else {
            reject(new Error('Unsupported input type for parseCSVInput'));
        }
    });
}

// Show message in data-status
function setDataStatus(msg) { $('data-status').textContent = msg; }

// Utility: compute median
function calculateMedian(values) {
    if (!values || !values.length) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return (sorted.length % 2 !== 0) ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Utility: compute mode (most frequent), returns first mode if tie
function calculateMode(values) {
    if (!values || !values.length) return null;
    const freq = {};
    let maxCount = 0;
    let mode = null;
    values.forEach(v => {
        if (v === null || v === undefined || v === '') return;
        freq[v] = (freq[v] || 0) + 1;
        if (freq[v] > maxCount) {
            maxCount = freq[v];
            mode = v;
        }
    });
    return mode;
}

// Standard deviation
function calculateStdDev(values) {
    if (!values || !values.length) return 1;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const sq = values.map(v => Math.pow(v - mean, 2));
    const varr = sq.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(varr);
}

// One-hot encode helper: returns array of 0/1 for categories
function oneHotEncode(value, categories) {
    return categories.map(c => (value === c ? 1 : 0));
}

// Download helper
function downloadBlob(filename, content, mimeType = 'text/csv;charset=utf-8') {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ========= Data loading & inspection ========= */

async function loadData() {
    const trainFile = $('train-file').files[0];
    const testFile = $('test-file').files[0];

    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }

    setDataStatus('Loading and parsing CSV files (PapaParse)...');

    try {
        // Parse using PapaParse (handles quoted commas correctly)
        const [trainRows, testRows] = await Promise.all([
            parseCSVInput(trainFile),
            parseCSVInput(testFile)
        ]);

        // Basic validation / conversion: ensure numeric fields are numbers (Papa's dynamicTyping should handle most)
        trainData = trainRows.map(r => normalizeRowTypes(r));
        testData = testRows.map(r => normalizeRowTypes(r));

        setDataStatus(`Data loaded: Train ${trainData.length} rows, Test ${testData.length} rows.`);

        // Enable Inspect button
        $('inspect-btn').disabled = false;
    } catch (err) {
        console.error(err);
        setDataStatus('Error parsing CSV files: ' + err.message);
        alert('Error parsing CSV files. Check console for details.');
    }
}

// Ensure numeric-like strings are numbers (extra safety)
function normalizeRowTypes(row) {
    const out = {};
    Object.keys(row).forEach(k => {
        let v = row[k];
        // Trim strings
        if (typeof v === 'string') v = v.trim();
        // Convert empty strings to null
        if (v === '') v = null;
        // If it's numeric-like (and dynamicTyping failed), convert
        if (v !== null && typeof v === 'string' && !isNaN(v) && v !== '') {
            const num = Number(v);
            if (!Number.isNaN(num)) v = num;
        }
        out[k] = v;
    });
    return out;
}

function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }

    // Show preview (first 10 rows)
    const previewDiv = $('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));

    // Stats
    const statsDiv = $('data-stats');
    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;

    // missing percentage
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li>${feature}: ${missingPercent}%</li>`;
    });
    missingInfo += '</ul>';

    statsDiv.innerHTML = `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;

    // Visualizations (survival by Sex, Pclass)
    createVisualizations();

    // Enable preprocess button
    $('preprocess-btn').disabled = false;
}

function createPreviewTable(data) {
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.keys(row).forEach(k => {
            const td = document.createElement('td');
            td.textContent = (row[k] === null || row[k] === undefined) ? 'NULL' : String(row[k]);
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}

/* ========= Visualizations (tfjs-vis) ========= */

function createVisualizations() {
    // Survival by Sex
    const survivalBySex = {};
    trainData.forEach(row => {
        const sex = row.Sex || 'Missing';
        if (!survivalBySex[sex]) survivalBySex[sex] = { survived: 0, total: 0 };
        survivalBySex[sex].total++;
        if (row.Survived === 1) survivalBySex[sex].survived++;
    });
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({ x: sex, y: (stats.survived / stats.total) * 100 }));
    tfvis.render.barchart({ name: 'Survival Rate by Sex', tab: 'Charts' }, sexData, { xLabel: 'Sex', yLabel: 'Survival Rate (%)' });

    // Survival by Pclass
    const survivalByPclass = {};
    trainData.forEach(row => {
        const p = row.Pclass === undefined ? 'Missing' : String(row.Pclass);
        if (!survivalByPclass[p]) survivalByPclass[p] = { survived: 0, total: 0 };
        survivalByPclass[p].total++;
        if (row.Survived === 1) survivalByPclass[p].survived++;
    });
    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({ x: `Class ${pclass}`, y: (stats.survived / stats.total) * 100 }));
    tfvis.render.barchart({ name: 'Survival Rate by Passenger Class', tab: 'Charts' }, pclassData, { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' });

    // Inform the user about tfvis visor
    const chartsDiv = $('charts');
    chartsDiv.innerHTML = '<p class="muted">Charts rendered to the <strong>tfjs-vis</strong> visor. Click the small visor button at the bottom-right to open the visualization pane.</p>';
}

/* ========= Preprocessing ========= */

function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }

    const outputDiv = $('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';

    try {
        // Compute imputation values from training data
        const ageVals = trainData.map(r => r.Age).filter(v => v !== null && v !== undefined && !Number.isNaN(v));
        const fareVals = trainData.map(r => r.Fare).filter(v => v !== null && v !== undefined && !Number.isNaN(v));
        const embarkedVals = trainData.map(r => r.Embarked).filter(v => v !== null && v !== undefined && v !== '');

        const ageMedian = calculateMedian(ageVals);
        const fareMedian = calculateMedian(fareVals);
        const embarkedMode = calculateMode(embarkedVals);

        // Precompute std dev for standardization
        const ageStd = calculateStdDev(ageVals) || 1;
        const fareStd = calculateStdDev(fareVals) || 1;

        // Determine categories for one-hot encoding
        const pclassCategories = Array.from(new Set(trainData.map(r => r.Pclass).filter(v => v !== null && v !== undefined))).sort();
        const sexCategories = Array.from(new Set(trainData.map(r => r.Sex).filter(v => v !== null && v !== undefined)));
        const embarkedCategories = Array.from(new Set(embarkedVals)).sort();

        // Save preprocessing config into closure for later use
        const config = {
            ageMedian, fareMedian, embarkedMode, ageStd, fareStd,
            pclassCategories, sexCategories, embarkedCategories,
            addFamily: $('add-family-features').checked
        };

        // Extract features from row
        const extractRowFeatures = (row) => {
            // Impute
            const age = (row.Age !== null && row.Age !== undefined && !Number.isNaN(row.Age)) ? row.Age : config.ageMedian;
            const fare = (row.Fare !== null && row.Fare !== undefined && !Number.isNaN(row.Fare)) ? row.Fare : config.fareMedian;
            const embarked = (row.Embarked !== null && row.Embarked !== undefined && row.Embarked !== '') ? row.Embarked : config.embarkedMode;

            // Standardize
            const standardizedAge = (age - config.ageMedian) / config.ageStd;
            const standardizedFare = (fare - config.fareMedian) / config.fareStd;

            // Basic numeric features (SibSp, Parch)
            const sibsp = (row.SibSp !== null && row.SibSp !== undefined && !Number.isNaN(row.SibSp)) ? row.SibSp : 0;
            const parch = (row.Parch !== null && row.Parch !== undefined && !Number.isNaN(row.Parch)) ? row.Parch : 0;

            // One-hot encodings
            const pclassOH = oneHotEncode(row.Pclass, config.pclassCategories);
            const sexOH = oneHotEncode(row.Sex, config.sexCategories);
            const embarkedOH = oneHotEncode(embarked, config.embarkedCategories);

            // Base features
            let features = [standardizedAge, standardizedFare, sibsp, parch];
            features = features.concat(pclassOH, sexOH, embarkedOH);

            // Optional family features
            if (config.addFamily) {
                const familySize = sibsp + parch + 1;
                const isAlone = familySize === 1 ? 1 : 0;
                features.push(familySize, isAlone);
            }

            return features;
        };

        // Preprocess train
        const trainFeaturesArr = [];
        const trainLabelsArr = [];
        trainData.forEach(r => {
            trainFeaturesArr.push(extractRowFeatures(r));
            trainLabelsArr.push(Number(r[TARGET_FEATURE] === undefined || r[TARGET_FEATURE] === null ? 0 : r[TARGET_FEATURE]));
        });

        // Preprocess test
        const testFeaturesArr = [];
        const testIds = [];
        testData.forEach(r => {
            testFeaturesArr.push(extractRowFeatures(r));
            testIds.push(r[ID_FEATURE]);
        });

        // Convert to tensors (training)
        preprocessedTrainData = {
            features: tf.tensor2d(trainFeaturesArr),
            labels: tf.tensor1d(trainLabelsArr, 'int32')
        };

        preprocessedTestData = {
            features: testFeaturesArr,   // keep as array for later conversion (predict)
            passengerIds: testIds
        };

        outputDiv.innerHTML = `
            <p>Preprocessing completed!</p>
            <p>Training features shape: [${preprocessedTrainData.features.shape}]</p>
            <p>Training labels shape: [${preprocessedTrainData.labels.shape}]</p>
            <p>Test features: ${preprocessedTestData.features.length} rows, feature length ${preprocessedTestData.features[0].length}</p>
        `;

        // Enable create model button
        $('create-model-btn').disabled = false;

        // Save config for later if needed (not global here)
        preprocessedTrainData._preprocConfig = config;
        preprocessedTestData._preprocConfig = config;
    } catch (err) {
        console.error(err);
        alert('Error during preprocessing: ' + err.message);
    }
}

/* ========= Model setup ========= */

function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    const inputShape = preprocessedTrainData.features.shape[1];

    model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputShape] }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Show lightweight summary in the page and console
    const summaryDiv = $('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    let summaryText = `<p>Sequential model with ${model.layers.length} layers. Total params: ${model.countParams()}</p>`;
    summaryText += '<ul>';
    model.layers.forEach((layer, i) => summaryText += `<li>Layer ${i + 1}: ${layer.getClassName()} - output shape ${JSON.stringify(layer.outputShape)}</li>`);
    summaryText += '</ul>';
    summaryDiv.innerHTML += summaryText;

    $('create-model-btn').disabled = false;
    $('summary-btn').disabled = false;
    $('train-btn').disabled = false;
    // Allow summary-button to print detailed summary to console
    $('summary-btn').onclick = () => {
        console.log('Model summary (layers):', model.layers);
        model.summary(); // prints to console in tfjs
    };
}

/* ========= Stratified split helper ========= */

// Performs a stratified split on features tensor and labels tensor (80/20 default)
function stratifiedSplit(featuresTensor, labelsTensor, testFraction = 0.2) {
    // Convert tensors to arrays for indices
    const labels = labelsTensor.arraySync();
    const indicesByLabel = {};
    labels.forEach((lbl, idx) => {
        const key = String(lbl);
        if (!indicesByLabel[key]) indicesByLabel[key] = [];
        indicesByLabel[key].push(idx);
    });

    const trainIndices = [];
    const valIndices = [];

    Object.keys(indicesByLabel).forEach(label => {
        const indices = indicesByLabel[label];
        // shuffle
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        const valCount = Math.max(1, Math.floor(indices.length * testFraction));
        valIndices.push(...indices.slice(0, valCount));
        trainIndices.push(...indices.slice(valCount));
    });

    // Convert index arrays to tensors
    const xTrain = featuresTensor.gather(tf.tensor1d(trainIndices, 'int32'));
    const yTrain = labelsTensor.gather(tf.tensor1d(trainIndices, 'int32'));
    const xVal = featuresTensor.gather(tf.tensor1d(valIndices, 'int32'));
    const yVal = labelsTensor.gather(tf.tensor1d(valIndices, 'int32'));

    return { xTrain, yTrain, xVal, yVal };
}

/* ========= Training ========= */

async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model and preprocess data first.');
        return;
    }

    $('train-btn').disabled = true;
    $('stop-train-btn').disabled = false;
    stopRequested = false;
    $('training-status').innerHTML = 'Preparing data and starting training...';

    try {
        // Stratified split 80/20
        const split = stratifiedSplit(preprocessedTrainData.features, preprocessedTrainData.labels, 0.2);
        const xTrain = split.xTrain;
        const yTrain = split.yTrain;
        const xVal = split.xVal;
        const yVal = split.yVal;

        // Save validation tensors globally for metrics
        validationData = xVal;
        validationLabels = yVal;

        // Convert val tensors to arrays for metric functions later
        validationLabelsArray = validationLabels.arraySync();

        // Fit model with callbacks: tfjs-vis and early stopping
        const fitCallbacks = tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Training' },
            ['loss', 'acc', 'val_loss', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        );

        const earlyStop = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 });

        // We provide an onEpochEnd to update status
        const onEpochEndCb = {
            onEpochEnd: async (epoch, logs) => {
                $('training-status').innerHTML = `Epoch ${epoch + 1}: loss=${(logs.loss||0).toFixed(4)}, acc=${(logs.acc||0).toFixed(4)}, val_loss=${(logs.val_loss||0).toFixed(4)}, val_acc=${(logs.val_acc||0).toFixed(4)}`;
                // allow manual stop
                if (stopRequested) {
                    $('training-status').innerHTML += ' | Stop requested. Halting...';
                    // Note: tfjs does not provide a built-in way to cancel training from here other than throwing
                    throw new Error('Training stopped by user');
                }
            }
        };

        // Train
        trainingHistory = await model.fit(xTrain, yTrain, {
            epochs: 50,
            batchSize: 32,
            validationData: [xVal, yVal],
            callbacks: [fitCallbacks, earlyStop, onEpochEndCb]
        });

        $('training-status').innerHTML += '<p>Training completed.</p>';
        $('stop-train-btn').disabled = true;

        // Compute validation probabilities and store for metrics
        const valProbsTensor = model.predict(validationData);
        validationProbsArray = Array.from((await valProbsTensor.data()));
        // Keep numeric arrays of labels already saved before
        // Plot ROC and compute AUC
        await plotROC(validationLabelsArray, validationProbsArray);

        // Enable threshold slider and prediction/export buttons
        $('threshold-slider').disabled = false;
        $('predict-btn').disabled = false;

        // Attach slider event
        $('threshold-slider').addEventListener('input', () => {
            const thr = parseFloat($('threshold-slider').value);
            $('threshold-value').textContent = thr.toFixed(2);
            updateMetricsTable(validationProbsArray, validationLabelsArray, thr);
        });

        // Initialize metrics display at default threshold
        const defaultThr = parseFloat($('threshold-slider').value);
        updateMetricsTable(validationProbsArray, validationLabelsArray, defaultThr);

    } catch (err) {
        console.error('Training error:', err);
        $('training-status').innerHTML = 'Training stopped or error: ' + err.message;
        // re-enable train button so user can try again
        $('train-btn').disabled = false;
        $('stop-train-btn').disabled = true;
    }
}

// Stop training request handler
$('stop-train-btn').addEventListener('click', () => {
    stopRequested = true;
});

/* ========= Metrics: confusion matrix, precision, recall, F1, ROC/AUC ========= */

// Update confusion matrix and performance metrics table in DOM
function updateMetricsTable(probArray, trueLabels, threshold = 0.5) {
    if (!probArray || !trueLabels) return;

    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let i = 0; i < probArray.length; i++) {
        const pred = probArray[i] >= threshold ? 1 : 0;
        const actual = Number(trueLabels[i]);
        if (pred === 1 && actual === 1) tp++;
        else if (pred === 0 && actual === 0) tn++;
        else if (pred === 1 && actual === 0) fp++;
        else if (pred === 0 && actual === 1) fn++;
    }

    // Avoid division by zero
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : 2 * (precision * recall) / (precision + recall);
    const accuracy = (tp + tn) / (tp + tn + fp + fn);

    // Update confusion matrix HTML
    const cmHtml = `
        <table>
            <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
            <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
        </table>
    `;
    $('confusion-matrix').innerHTML = cmHtml;

    // Update performance metrics HTML
    const perfHtml = `
        <p>Accuracy: ${(accuracy * 100).toFixed(2)}%</p>
        <p>Precision: ${precision.toFixed(4)}</p>
        <p>Recall: ${recall.toFixed(4)}</p>
        <p>F1 Score: ${f1.toFixed(4)}</p>
    `;
    $('performance-metrics').innerHTML = perfHtml;
}

// Compute ROC points and plot ROC curve + AUC using trapezoidal rule
async function plotROC(trueLabels, probArray) {
    if (!trueLabels || !probArray) return;
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const rocPoints = [];

    for (let t of thresholds) {
        let tp = 0, tn = 0, fp = 0, fn = 0;
        for (let i = 0; i < probArray.length; i++) {
            const pred = probArray[i] >= t ? 1 : 0;
            const actual = Number(trueLabels[i]);
            if (pred === 1 && actual === 1) tp++;
            else if (pred === 0 && actual === 0) tn++;
            else if (pred === 1 && actual === 0) fp++;
            else if (pred === 0 && actual === 1) fn++;
        }
        const tpr = tp + fn === 0 ? 0 : tp / (tp + fn); // recall
        const fpr = fp + tn === 0 ? 0 : fp / (fp + tn);
        rocPoints.push({ fpr, tpr });
    }

    // Sort by fpr (should already be sorted by decreasing threshold but ensure)
    const sorted = rocPoints.slice().sort((a, b) => a.fpr - b.fpr);

    // Compute AUC via trapezoidal rule
    let auc = 0;
    for (let i = 1; i < sorted.length; i++) {
        const x1 = sorted[i - 1].fpr, x2 = sorted[i].fpr;
        const y1 = sorted[i - 1].tpr, y2 = sorted[i].tpr;
        auc += (x2 - x1) * (y1 + y2) / 2;
    }

    // Prepare data for tfjs-vis linechart
    const values = sorted.map(p => ({ x: p.fpr, y: p.tpr }));
    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values },
        { xLabel: 'False Positive Rate', yLabel: 'True Positive Rate', width: 400, height: 400 }
    );

    // Display AUC in performance metrics (append)
    const perfDiv = $('performance-metrics');
    const existing = perfDiv.innerHTML;
    perfDiv.innerHTML = existing + `<p>AUC: ${auc.toFixed(4)}</p>`;
}

/* ========= Prediction & Export ========= */

async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }

    $('prediction-output').innerHTML = 'Making predictions on test set...';

    try {
        // Convert test features array to tensor2d
        const testX = tf.tensor2d(preprocessedTestData.features);
        const probsTensor = model.predict(testX);
        testPredictions = probsTensor;
        const probsArray = Array.from(await probsTensor.data());

        // Build results (PassengerId, Survived based on threshold 0.5 by default)
        const thr = parseFloat($('threshold-slider').value);
        const results = preprocessedTestData.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: probsArray[i] >= thr ? 1 : 0,
            Probability: probsArray[i]
        }));

        // Show first 10 results
        const first10 = results.slice(0, 10);
        $('prediction-output').innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        $('prediction-output').appendChild(createPredictionTable(first10));
        $('prediction-output').innerHTML += `<p>Total predictions: ${results.length}. Threshold used: ${thr.toFixed(2)}</p>`;

        // Enable export button
        $('export-btn').disabled = false;
    } catch (err) {
        console.error(err);
        $('prediction-output').innerHTML = 'Prediction error: ' + err.message;
    }
}

function createPredictionTable(data) {
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    data.forEach(r => {
        const tr = document.createElement('tr');
        const tdId = document.createElement('td'); tdId.textContent = r.PassengerId; tr.appendChild(tdId);
        const tdS = document.createElement('td'); tdS.textContent = r.Survived; tr.appendChild(tdS);
        const tdP = document.createElement('td'); tdP.textContent = r.Probability.toFixed(4); tr.appendChild(tdP);
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}

async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }

    $('export-status').textContent = 'Preparing export files...';

    try {
        const probsArray = Array.from(await testPredictions.data());
        // submission.csv (PassengerId,Survived)
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            const surv = probsArray[i] >= parseFloat($('threshold-slider').value) ? 1 : 0;
            submissionCSV += `${id},${surv}\n`;
        });
        downloadBlob('submission.csv', submissionCSV, 'text/csv');

        // probabilities.csv
        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            probabilitiesCSV += `${id},${probsArray[i].toFixed(6)}\n`;
        });
        downloadBlob('probabilities.csv', probabilitiesCSV, 'text/csv');

        // Save model to downloads
        await model.save('downloads://titanic-tfjs-model');

        $('export-status').textContent = 'Export completed: submission.csv, probabilities.csv downloaded; model saved to downloads.';
    } catch (err) {
        console.error(err);
        $('export-status').textContent = 'Export error: ' + err.message;
    }
}

/* ========= Wire UI buttons ========= */

document.addEventListener('DOMContentLoaded', () => {
    $('load-data-btn').addEventListener('click', loadData);
    $('inspect-btn').addEventListener('click', inspectData);
    $('preprocess-btn').addEventListener('click', preprocessData);
    $('create-model-btn').addEventListener('click', createModel);
    $('train-btn').addEventListener('click', trainModel);
    $('predict-btn').addEventListener('click', predict);
    $('export-btn').addEventListener('click', exportResults);
});
