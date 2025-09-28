// Load CSV an toàn với PapaParse
async function loadCSV(filePath) {
  return new Promise((resolve, reject) => {
    Papa.parse(filePath, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        resolve(results.data);
      },
      error: (err) => reject(err),
    });
  });
}

// Preprocess Titanic data
function preprocessData(data) {
  const features = [];
  const labels = [];

  data.forEach(row => {
    if (!row.Age || !row.Sex || !row.Pclass) return;

    const sex = row.Sex === "male" ? 1 : 0;
    const age = parseFloat(row.Age);
    const pclass = parseInt(row.Pclass);
    const sibsp = parseInt(row.SibSp);
    const parch = parseInt(row.Parch);

    features.push([sex, age, pclass, sibsp, parch]);
    if (row.Survived !== undefined) {
      labels.push(parseInt(row.Survived));
    }
  });

  return {
    features: tf.tensor2d(features),
    labels: labels.length > 0 ? tf.tensor1d(labels, "int32") : null,
  };
}

// Build model
function createModel(inputShape) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [inputShape] }));
  model.add(tf.layers.dense({ units: 8, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

// Metrics
function calculateMetrics(predictions, labels, threshold) {
  let tp = 0, fp = 0, tn = 0, fn = 0;

  predictions.forEach((pred, i) => {
    const predicted = pred >= threshold ? 1 : 0;
    const actual = labels[i];
    if (predicted === 1 && actual === 1) tp++;
    if (predicted === 1 && actual === 0) fp++;
    if (predicted === 0 && actual === 0) tn++;
    if (predicted === 0 && actual === 1) fn++;
  });

  const accuracy = (tp + tn) / (tp + tn + fp + fn);
  const precision = tp / (tp + fp + 1e-7);
  const recall = tp / (tp + fn + 1e-7);
  const f1 = 2 * (precision * recall) / (precision + recall + 1e-7);

  return { accuracy, precision, recall, f1 };
}

// Main run
async function run() {
  const trainData = await loadCSV("train.csv");
  const testData = await loadCSV("test.csv");

  const preTrain = preprocessData(trainData);
  const split = Math.floor(preTrain.features.shape[0] * 0.8);

  const trainFeatures = preTrain.features.slice([0, 0], [split, preTrain.features.shape[1]]);
  const trainLabels = preTrain.labels.slice([0], [split]);
  const valFeatures = preTrain.features.slice([split, 0], [preTrain.features.shape[0] - split, preTrain.features.shape[1]]);
  const valLabels = preTrain.labels.slice([split], [preTrain.features.shape[0] - split]);

  const model = createModel(trainFeatures.shape[1]);

  await model.fit(trainFeatures, trainLabels, {
    epochs: 50,
    validationData: [valFeatures, valLabels],
    callbacks: [
      tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ["loss", "acc"],
        { height: 200, callbacks: ["onEpochEnd"] }
      ),
      {
        onEpochEnd: (epoch, logs) => {
          const row = document.createElement("tr");
          row.innerHTML = `<td>${epoch + 1}</td>
                           <td>${logs.loss.toFixed(4)}</td>
                           <td>${logs.acc.toFixed(4)}</td>
                           <td>${logs.val_loss.toFixed(4)}</td>
                           <td>${logs.val_acc.toFixed(4)}</td>`;
          document.querySelector("#evaluation-table tbody").appendChild(row);
        },
      },
    ],
  });

  // Predictions
  const preds = model.predict(valFeatures).dataSync();
  const labels = valLabels.dataSync();
  const thresholdSlider = document.getElementById("threshold");
  const thresholdValue = document.getElementById("threshold-value");

  function updateMetrics() {
    const threshold = parseFloat(thresholdSlider.value);
    thresholdValue.textContent = threshold.toFixed(2);
    const metrics = calculateMetrics(Array.from(preds), Array.from(labels), threshold);
    document.getElementById("accuracy").textContent = metrics.accuracy.toFixed(2);
    document.getElementById("precision").textContent = metrics.precision.toFixed(2);
    document.getElementById("recall").textContent = metrics.recall.toFixed(2);
    document.getElementById("f1-score").textContent = metrics.f1.toFixed(2);
  }

  thresholdSlider.oninput = updateMetrics;
  updateMetrics();
}
