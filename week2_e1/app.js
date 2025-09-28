// =========================
// Global variables
// =========================
let trainData = [];
let testData = [];
let model;
let stopFlag = false;

// =========================
// CSV Loader (FileReader)
// =========================
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map(line => {
    const cols = line.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/); // handle commas inside quotes
    const row = {};
    headers.forEach((h, i) => {
      row[h.trim()] = cols[i] ? cols[i].replace(/"/g, "").trim() : "";
    });
    return row;
  });
}

function loadData() {
  const trainFile = document.getElementById("trainFile").files[0];
  const testFile = document.getElementById("testFile").files[0];

  if (!trainFile || !testFile) {
    alert("Please upload both train.csv and test.csv files!");
    return;
  }

  const readerTrain = new FileReader();
  readerTrain.onload = e => {
    trainData = parseCSV(e.target.result).map(cleanRow);
    console.log("Train loaded:", trainData.length);
    alert("Train data loaded successfully!");
  };
  readerTrain.readAsText(trainFile);

  const readerTest = new FileReader();
  readerTest.onload = e => {
    testData = parseCSV(e.target.result).map(cleanRow);
    console.log("Test loaded:", testData.length);
    alert("Test data loaded successfully!");
  };
  readerTest.readAsText(testFile);
}

// =========================
// Data Cleaning
// =========================
function cleanRow(row) {
  return {
    PassengerId: +row.PassengerId,
    Survived: row.Survived !== "" ? +row.Survived : null,
    Pclass: row.Pclass !== "" ? +row.Pclass : null,
    Sex: row.Sex ? row.Sex.trim().toLowerCase() : null,
    Age: row.Age !== "" ? +row.Age : null,
    SibSp: row.SibSp !== "" ? +row.SibSp : 0,
    Parch: row.Parch !== "" ? +row.Parch : 0,
    Fare: row.Fare !== "" ? +row.Fare : null,
    Embarked: row.Embarked ? row.Embarked.trim().toUpperCase() : null
  };
}

// =========================
// Inspect Data
// =========================
function inspectData() {
  if (trainData.length === 0) {
    alert("Load data first!");
    return;
  }

  const preview = document.getElementById("dataPreview");
  preview.innerHTML = "";

  // Show first 5 rows
  let html = "<table><thead><tr>";
  Object.keys(trainData[0]).forEach(k => html += `<th>${k}</th>`);
  html += "</tr></thead><tbody>";
  trainData.slice(0, 5).forEach(r => {
    html += "<tr>" + Object.values(r).map(v => `<td>${v}</td>`).join("") + "</tr>";
  });
  html += "</tbody></table>";
  preview.innerHTML = html;

  // Missing values count
  const missing = {};
  Object.keys(trainData[0]).forEach(k => {
    missing[k] = trainData.filter(r => r[k] === null || r[k] === "").length;
  });

  let missText = "<p class='mt-3 text-pink-700'><b>Missing values:</b><br>";
  for (let [k, v] of Object.entries(missing)) {
    missText += `${k}: ${v} · `;
  }
  missText += "</p>";
  preview.innerHTML += missText;

  // Simple charts with tfjs-vis
  const valid = trainData.filter(r => r.Survived !== null && r.Sex && r.Pclass !== null);

  const sexCounts = {};
  valid.forEach(r => {
    if (!sexCounts[r.Sex]) sexCounts[r.Sex] = { Survived: 0, Total: 0 };
    sexCounts[r.Sex].Total++;
    if (r.Survived === 1) sexCounts[r.Sex].Survived++;
  });
  const sexData = Object.entries(sexCounts).map(([sex, d]) => ({
    index: sex,
    value: d.Survived / d.Total
  }));
  tfvis.render.barchart({ name: "Survival Rate by Sex", tab: "Charts" }, sexData);

  const classCounts = {};
  valid.forEach(r => {
    if (!classCounts[r.Pclass]) classCounts[r.Pclass] = { Survived: 0, Total: 0 };
    classCounts[r.Pclass].Total++;
    if (r.Survived === 1) classCounts[r.Pclass].Survived++;
  });
  const classData = Object.entries(classCounts).map(([cls, d]) => ({
    index: "Class " + cls,
    value: d.Survived / d.Total
  }));
  tfvis.render.barchart({ name: "Survival Rate by Pclass", tab: "Charts" }, classData);
}

// =========================
// Preprocessing (simplified for demo)
// =========================
function preprocess() {
  if (trainData.length === 0) {
    alert("Load data first!");
    return;
  }

  // Impute missing Age, Fare, Embarked
  const ages = trainData.map(d => d.Age).filter(v => v != null);
  const medianAge = ages.sort((a,b)=>a-b)[Math.floor(ages.length/2)];
  trainData.forEach(d => { if (d.Age == null) d.Age = medianAge; });

  const fares = trainData.map(d => d.Fare).filter(v => v != null);
  const medianFare = fares.sort((a,b)=>a-b)[Math.floor(fares.length/2)];
  trainData.forEach(d => { if (d.Fare == null) d.Fare = medianFare; });

  const embarkCounts = {};
  trainData.forEach(d => { if (d.Embarked) embarkCounts[d.Embarked] = (embarkCounts[d.Embarked]||0)+1; });
  const modeEmbarked = Object.entries(embarkCounts).sort((a,b)=>b[1]-a[1])[0][0];
  trainData.forEach(d => { if (!d.Embarked) d.Embarked = modeEmbarked; });

  document.getElementById("preprocessInfo").innerText = 
    "✅ Preprocessing done: Age/Fare imputed, Embarked mode filled.";
}

// =========================
// Model Creation
// =========================
function createModel() {
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [6] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  model.summary();
  const summary = [];
  model.layers.forEach(l => summary.push(l.getConfig()));
  document.getElementById("modelSummary").innerText = JSON.stringify(summary, null, 2);
}

// =========================
// Training
// =========================
async function trainModel() {
  if (!model) { alert("Create model first!"); return; }

  stopFlag = false;

  // Dummy input tensors (you will replace with real features later)
  const xs = tf.randomNormal([100, 6]);
  const ys = tf.randomUniform([100, 1]).round();

  const history = await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: [
      tfvis.show.fitCallbacks({ name: 'Training Performance', tab: 'Training' }, ['loss', 'acc'], { height: 200 }),
      {
        onEpochEnd: async (epoch, logs) => {
          if (stopFlag) { model.stopTraining = true; }
        }
      }
    ]
  });
}

function stopTraining() {
  stopFlag = true;
  alert("⛔ Training will stop after current epoch!");
}

// =========================
// Metrics & Evaluation (dummy)
// =========================
function computeMetrics() {
  document.getElementById("rocCurve").innerText = "ROC Curve computed (placeholder).";
  document.getElementById("confMatrix").innerText = "Confusion Matrix will show here (placeholder).";
}

function updateThreshold(val) {
  document.getElementById("thresholdVal").innerText = val;
}

// =========================
// Prediction & Export (dummy)
// =========================
function predictTest() {
  if (!model || testData.length === 0) {
    alert("Load model and test data first!");
    return;
  }
  document.getElementById("predictionPreview").innerText = "Predictions done (placeholder).";
}

function exportResults() {
  alert("Export CSV (placeholder).");
}
