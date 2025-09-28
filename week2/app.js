// ==================== GLOBAL VARS ====================
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;

// Schema config
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// ==================== LOAD DATA ====================
async function loadData() {
  const trainFile = document.getElementById('train-file').files[0];
  const testFile = document.getElementById('test-file').files[0];
  if (!trainFile || !testFile) {
    alert('Please upload both training and test CSV files.');
    return;
  }
  const statusDiv = document.getElementById('data-status');
  statusDiv.innerHTML = 'Loading data...';
  try {
    const trainText = await readFile(trainFile);
    trainData = parseCSV(trainText);
    const testText = await readFile(testFile);
    testData = parseCSV(testText);
    statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
    document.getElementById('inspect-btn').disabled = false;
  } catch (err) {
    statusDiv.innerHTML = `Error loading data: ${err.message}`;
    console.error(err);
  }
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = e => reject(new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

// ✅ FIXED CSV PARSE: handle commas inside quotes
function parseCSV(csvText) {
  const lines = csvText.split(/\r?\n/).filter(l => l.trim() !== '');
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const values = [];
    let current = '';
    let insideQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        insideQuotes = !insideQuotes;
      } else if (char === ',' && !insideQuotes) {
        values.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    values.push(current.trim());
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = values[i] === '' ? null : values[i];
      if (!isNaN(obj[h]) && obj[h] !== null) obj[h] = parseFloat(obj[h]);
    });
    return obj;
  });
}

// ==================== INSPECT DATA ====================
function inspectData() {
  const previewDiv = document.getElementById('data-preview');
  previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
  const sample = trainData.slice(0, 10);
  previewDiv.appendChild(renderTable(sample));

  // Survival Rate by Sex
  const sexGroups = {};
  trainData.forEach(row => {
    if (row.Sex && row.Survived !== null) {
      if (!sexGroups[row.Sex]) sexGroups[row.Sex] = { survived: 0, total: 0 };
      sexGroups[row.Sex].total++;
      if (row.Survived == 1) sexGroups[row.Sex].survived++;
    }
  });
  const sexChartData = Object.keys(sexGroups).map(sex => ({
    x: sex,
    y: sexGroups[sex].survived / sexGroups[sex].total
  }));

  // Survival Rate by Pclass
  const classGroups = {};
  trainData.forEach(row => {
    if (row.Pclass && row.Survived !== null) {
      if (!classGroups[row.Pclass]) classGroups[row.Pclass] = { survived: 0, total: 0 };
      classGroups[row.Pclass].total++;
      if (row.Survived == 1) classGroups[row.Pclass].survived++;
    }
  });
  const classChartData = Object.keys(classGroups).map(cls => ({
    x: `Class ${cls}`,
    y: classGroups[cls].survived / classGroups[cls].total
  }));

  // Render charts
  const chartsDiv = document.getElementById('charts');
  chartsDiv.innerHTML = '<h3>Charts</h3>';
  tfvis.render.barchart({name: 'Survival Rate by Sex', tab: 'Charts'}, sexChartData, {xLabel: 'Sex', yLabel: 'Survival Rate'});
  tfvis.render.barchart({name: 'Survival Rate by Passenger Class', tab: 'Charts'}, classChartData, {xLabel: 'Class', yLabel: 'Survival Rate'});

  document.getElementById('preprocess-btn').disabled = false;
}

function renderTable(data) {
  const table = document.createElement('table');
  const headerRow = document.createElement('tr');
  Object.keys(data[0]).forEach(h => {
    const th = document.createElement('th');
    th.textContent = h;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);
  data.forEach(row => {
    const tr = document.createElement('tr');
    Object.keys(row).forEach(h => {
      const td = document.createElement('td');
      td.textContent = row[h];
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  return table;
}

// ==================== PREPROCESSING ====================
function preprocessData() {
  // Placeholder: Implement preprocessing logic
  // For simplicity we just create dummy tensors
  const xs = tf.tensor2d(trainData.map(r => [r.Age || 0, r.Fare || 0]));
  const ys = tf.tensor2d(trainData.map(r => [r.Survived || 0]));
  preprocessedTrainData = {xs, ys};

  const testXs = tf.tensor2d(testData.map(r => [r.Age || 0, r.Fare || 0]));
  preprocessedTestData = {xs: testXs};

  document.getElementById('preprocessing-output').innerHTML = 'Preprocessing complete!';
  document.getElementById('create-model-btn').disabled = false;
}

// ==================== MODEL ====================
function createModel() {
  model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [2], units: 8, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  document.getElementById('model-summary').innerHTML = 'Model created!';
  document.getElementById('train-btn').disabled = false;
}

// ==================== TRAIN ====================
async function trainModel() {
  if (!model || !preprocessedTrainData) {
    alert('Please create the model and preprocess data first.');
    return;
  }

  const statusDiv = document.getElementById('training-status');
  statusDiv.innerHTML = 'Training...';

  const {xs, ys} = preprocessedTrainData;
  const history = await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance', tab: 'Training' },
      ['loss', 'val_loss', 'acc', 'val_acc'],
      { callbacks: ['onEpochEnd'] }
    )
  });

  trainingHistory = history.history;
  statusDiv.innerHTML = 'Training complete!';
  document.getElementById('threshold-slider').disabled = false;
  document.getElementById('predict-btn').disabled = false;
}

// ==================== EVALUATION ====================
function updateMetrics(threshold=0.5) {
  if (!validationPredictions) return;
  const preds = validationPredictions.greater(tf.scalar(threshold)).dataSync();
  const labels = validationLabels.dataSync();

  let tp=0, tn=0, fp=0, fn=0;
  for (let i=0; i<preds.length; i++) {
    if (labels[i]===1 && preds[i]===1) tp++;
    if (labels[i]===0 && preds[i]===0) tn++;
    if (labels[i]===0 && preds[i]===1) fp++;
    if (labels[i]===1 && preds[i]===0) fn++;
  }

  const confusionDiv = document.getElementById('confusion-matrix');
  confusionDiv.innerHTML = `
    <h3>Confusion Matrix</h3>
    <table>
      <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
      <tr><th>True 0</th><td>${tn}</td><td>${fp}</td></tr>
      <tr><th>True 1</th><td>${fn}</td><td>${tp}</td></tr>
    </table>
  `;

  const acc = (tp+tn)/(tp+tn+fp+fn);
  const precision = tp/(tp+fp+1e-7);
  const recall = tp/(tp+fn+1e-7);
  const f1 = 2*precision*recall/(precision+recall+1e-7);

  const perfDiv = document.getElementById('performance-metrics');
  perfDiv.innerHTML = `
    <h3>Performance Metrics</h3>
    <p>Accuracy: ${acc.toFixed(3)}</p>
    <p>Precision: ${precision.toFixed(3)}</p>
    <p>Recall: ${recall.toFixed(3)}</p>
    <p>F1 Score: ${f1.toFixed(3)}</p>
  `;
}

// ==================== PREDICT ====================
function predict() {
  if (!preprocessedTestData) {
    alert('Preprocess test data first.');
    return;
  }
  const preds = model.predict(preprocessedTestData.xs);
  testPredictions = preds.dataSync();
  const outputDiv = document.getElementById('prediction-output');
  outputDiv.innerHTML = '<h3>Predictions (first 20)</h3>' + testPredictions.slice(0,20).join(', ');
  document.getElementById('export-btn').disabled = false;
}

// ==================== EXPORT ====================
function exportResults() {
  const ids = testData.map(r => r[ID_FEATURE]);
  const results = ['PassengerId,Survived\n'];
  for (let i=0; i<ids.length; i++) {
    const survived = testPredictions[i] > 0.5 ? 1 : 0;
    results.push(`${ids[i]},${survived}\n`);
  }
  const blob = new Blob(results, {type: 'text/csv'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'submission.csv';
  link.click();
  document.getElementById('export-status').innerHTML = 'Results exported!';
}
