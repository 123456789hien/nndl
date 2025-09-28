// ===== Global variables =====
let trainData = [];
let testData = [];
let model;
let threshold = 0.5;

// ===== CSV Loader =====
document.getElementById('loadBtn').addEventListener('click', () => {
  const trainFile = document.getElementById('trainFile').files[0];
  const testFile = document.getElementById('testFile').files[0];
  if (!trainFile || !testFile) {
    alert('Please select both train.csv and test.csv');
    return;
  }
  loadCSV(trainFile, true);
  loadCSV(testFile, false);
});

function loadCSV(file, isTrain) {
  Papa.parse(file, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    quoteChar: '"',       // fix comma escape issue
    escapeChar: '"',
    complete: (results) => {
      if (isTrain) {
        trainData = results.data;
        console.log('Train loaded', trainData.length);
      } else {
        testData = results.data;
        console.log('Test loaded', testData.length);
      }
      if (trainData.length && testData.length) {
        previewData();
        drawCharts();
      }
    },
    error: (err) => {
      console.error('CSV parse error', err);
      alert('Failed to parse CSV: ' + err.message);
    }
  });
}

// ===== Preview Table =====
function previewData() {
  const head = trainData.slice(0, 5);
  let html = '<table border="1"><tr>';
  Object.keys(head[0]).forEach(k => html += `<th>${k}</th>`);
  html += '</tr>';
  head.forEach(row => {
    html += '<tr>';
    Object.values(row).forEach(v => html += `<td>${v}</td>`);
    html += '</tr>';
  });
  html += '</table>';
  document.getElementById('preview').innerHTML = html;
}

// ===== Simple Charts =====
function drawCharts() {
  // Example: Survival by Sex
  const maleSurvived = trainData.filter(r => r.Sex === 'male' && r.Survived === 1).length;
  const maleDied = trainData.filter(r => r.Sex === 'male' && r.Survived === 0).length;
  const femaleSurvived = trainData.filter(r => r.Sex === 'female' && r.Survived === 1).length;
  const femaleDied = trainData.filter(r => r.Sex === 'female' && r.Survived === 0).length;

  const surface = { name: 'Survival by Sex', tab: 'Charts', styles: { height: 300 } };
  tfvis.render.barchart(
    surface,
    [
      { index: 'Male Survived', value: maleSurvived },
      { index: 'Male Died', value: maleDied },
      { index: 'Female Survived', value: femaleSurvived },
      { index: 'Female Died', value: femaleDied }
    ],
    { xLabel: 'Group', yLabel: 'Count' }
  );
}

// ===== Data Preprocessing =====
function preprocess(data, isTrain) {
  // drop PassengerId
  return data.map(r => {
    const age = isNaN(r.Age) ? medianAge : r.Age;
    const fare = isNaN(r.Fare) ? medianFare : r.Fare;
    return {
      Survived: isTrain ? r.Survived : undefined,
      Pclass: r.Pclass,
      Sex: r.Sex === 'female' ? 1 : 0,
      Age: (age - meanAge) / stdAge,
      SibSp: r.SibSp,
      Parch: r.Parch,
      Fare: (fare - meanFare) / stdFare,
      Embarked: r.Embarked === 'C' ? 1 : r.Embarked === 'Q' ? 2 : 0
    };
  });
}

// compute median/mean/std from training
let medianAge, medianFare, meanAge, stdAge, meanFare, stdFare;

function computeStats() {
  const ages = trainData.map(r => r.Age).filter(v => !isNaN(v));
  const fares = trainData.map(r => r.Fare).filter(v => !isNaN(v));
  const median = arr => arr.sort((a,b)=>a-b)[Math.floor(arr.length/2)];
  const mean = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
  const std = (arr,m) => Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/arr.length);

  medianAge = median(ages);
  medianFare = median(fares);
  meanAge = mean(ages);
  stdAge = std(ages, meanAge);
  meanFare = mean(fares);
  stdFare = std(fares, meanFare);
}

// ===== Model =====
function createModel(inputDim) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputDim] }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  m.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return m;
}

// ===== Training =====
document.getElementById('trainBtn').addEventListener('click', async () => {
  if (!trainData.length) { alert('Load data first'); return; }
  computeStats();
  const proc = preprocess(trainData, true);

  const xs = tf.tensor2d(proc.map(r => [r.Pclass, r.Sex, r.Age, r.SibSp, r.Parch, r.Fare, r.Embarked]));
  const ys = tf.tensor2d(proc.map(r => [r.Survived]));

  model = createModel(xs.shape[1]);
  const surface = { name: 'Training', tab: 'Model' };
  const history = await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: tfvis.show.fitCallbacks(surface, ['loss','val_loss','acc','val_acc'], { callbacks:['onEpochEnd'] })
  });
  document.getElementById('modelStatus').innerText = 'Training complete!';
});

// ===== Threshold Slider =====
document.getElementById('thresholdSlider').addEventListener('input', (e) => {
  threshold = parseFloat(e.target.value);
  document.getElementById('thresholdVal').innerText = threshold.toFixed(2);
});

// ===== Prediction =====
document.getElementById('predictBtn').addEventListener('click', () => {
  if (!model || !testData.length) { alert('Train model and load test first'); return; }
  const proc = preprocess(testData, false);
  const xs = tf.tensor2d(proc.map(r => [r.Pclass, r.Sex, r.Age, r.SibSp, r.Parch, r.Fare, r.Embarked]));
  const probs = model.predict(xs).dataSync();
  testData.forEach((r,i) => { r.Survived = probs[i] > threshold ? 1 : 0; r.Prob = probs[i]; });
  alert('Prediction complete');
});

// ===== Export =====
document.getElementById('exportBtn').addEventListener('click', () => {
  if (!testData.length) { alert('No predictions yet'); return; }
  let csv = 'PassengerId,Survived\n';
  testData.forEach(r => {
    csv += `${r.PassengerId},${r.Survived}\n`;
  });
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'submission.csv';
  a.click();
});
