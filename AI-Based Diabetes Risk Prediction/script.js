// ==================== GLOBAL VARIABLES ====================
let csvData = [];
let models = { logistic: null, neuralNet: null };
let classStats = null;
let logisticWeights = null;
let charts = { class: null, feature: null, edaAge: null, edaBmi: null };
let edaStats = null;

const featureNames = [
    'gender',
    'age',
    'hypertension',
    'heart_disease',
    'smoking_history',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level'
];

const smokingMap = { 
    'Never': 0, 
    'No Info': 1, 
    'Current': 2, 
    'Former': 3, 
    'Ever': 4, 
    'Not Current': 5 
};

// ==================== FILE UPLOAD (MANUAL) ====================
function handleFileUpload() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file');
        return;
    }

    document.getElementById('uploadStatus').innerHTML =
        '<div class="loading"></div> Parsing uploaded CSV...';

    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
            processParsedData(results.data, 'Uploaded dataset loaded successfully!');
        },
        error: (error) => {
            alert('Error parsing CSV: ' + error.message);
        }
    });
}

// ==================== FILE UPLOAD (AUTO-LOAD FROM GITHUB) ====================
async function autoLoadDataset() {
    // Đường dẫn tương đối tới file trong repo GitHub Pages
    // Nếu bạn đặt file trong thư mục /data thì đổi lại thành 'data/diabetes_raw_cleaned_25k.csv'
    const githubUrl = 'diabetes_raw_cleaned_25k.csv';

    document.getElementById('uploadStatus').innerHTML =
        '<div class="loading"></div> Loading default dataset...';

    try {
        const response = await fetch(githubUrl);
        if (!response.ok) throw new Error('Dataset not found at ' + githubUrl);

        const csvText = await response.text();

        Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true,
            complete: (results) => {
                processParsedData(results.data, 'Default dataset loaded successfully!');
            },
            error: (error) => {
                document.getElementById('uploadStatus').innerHTML =
                    '<strong style="color: var(--danger);">✗ Failed to parse dataset</strong>';
                console.error(error);
            }
        });

    } catch (err) {
        document.getElementById('uploadStatus').innerHTML =
            '<strong style="color: var(--danger);">✗ Failed to load dataset</strong>';
        console.error(err);
    }
}

// Common post-processing for uploaded or auto-loaded data
function processParsedData(rawData, successMessage) {
    csvData = rawData.filter(row => Object.values(row).some(val => val));

    if (csvData.length === 0) {
        alert('Dataset is empty');
        return;
    }

    const positives = csvData.filter(row => parseInt(row.diabetes) === 1).length;
    const negatives = csvData.length - positives;
    classStats = { positives, negatives, total: csvData.length };

    document.getElementById('uploadStatus').innerHTML = `
        <strong>✓ ${successMessage}</strong><br>
        Total Rows: <strong style="color: var(--primary);">${classStats.total}</strong><br>
        Diabetes Cases: <strong style="color: var(--accent);">${classStats.positives}</strong><br>
        Distribution: ${((negatives/classStats.total)*100).toFixed(1)}% negative and
        ${((positives/classStats.total)*100).toFixed(1)}% positive
    `;
    
    document.getElementById('trainBtn').disabled = false;
    document.getElementById('chartsContainer').classList.remove('hidden');

    drawClassChart();
    computeEdaStats();
    drawEdaCharts();
    document.getElementById('edaContainer').classList.remove('hidden');
}

// ==================== CHARTS ====================
function drawClassChart() {
    if (!classStats) return;

    const ctx = document.getElementById('classChart').getContext('2d');
    
    if (charts.class) charts.class.destroy();

    charts.class = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['No Diabetes (0)', 'Diabetes (1)'],
            datasets: [{
                label: 'Count',
                data: [classStats.negatives, classStats.positives],
                backgroundColor: ['rgba(34, 197, 94, 0.7)', 'rgba(99, 102, 241, 0.7)'],
                borderColor: ['rgba(34, 197, 94, 1)', 'rgba(99, 102, 241, 1)'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(148, 163, 184, 0.15)' },
                    ticks: { color: '#e5e7eb' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#e5e7eb' }
                }
            }
        }
    });
}

function drawFeatureChart() {
    if (!logisticWeights) return;

    const ctx = document.getElementById('featureChart').getContext('2d');
    
    if (charts.feature) charts.feature.destroy();

    const backgroundColors = logisticWeights.map(w => 
        w >= 0 ? 'rgba(99, 102, 241, 0.75)' : 'rgba(34, 197, 94, 0.75)'
    );
    const borderColors = logisticWeights.map(w => 
        w >= 0 ? 'rgba(99, 102, 241, 1)' : 'rgba(34, 197, 94, 1)'
    );

    charts.feature = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: featureNames,
            datasets: [{
                label: 'Weight',
                data: logisticWeights,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: { color: 'rgba(148, 163, 184, 0.15)' },
                    ticks: { color: '#e5e7eb' }
                },
                x: {
                    grid: { display: false },
                    ticks: { 
                        color: '#e5e7eb',
                        maxRotation: 45,
                        minRotation: 45,
                        autoSkip: false
                    }
                }
            }
        }
    });
}

// ==================== EDA (Age & BMI) ====================
function computeEdaStats() {
    const ageBins = {
        '<30': 0,
        '30-39': 0,
        '40-49': 0,
        '50-59': 0,
        '60+': 0
    };

    const bmiBins = {
        '<18.5': 0,
        '18.5-24.9': 0,
        '25-29.9': 0,
        '30-34.9': 0,
        '35+': 0
    };

    csvData.forEach(row => {
        const age = parseFloat(row.age);
        if (!isNaN(age)) {
            if (age < 30) ageBins['<30']++;
            else if (age < 40) ageBins['30-39']++;
            else if (age < 50) ageBins['40-49']++;
            else if (age < 60) ageBins['50-59']++;
            else ageBins['60+']++;
        }

        const bmi = parseFloat(row.bmi);
        if (!isNaN(bmi)) {
            if (bmi < 18.5) bmiBins['<18.5']++;
            else if (bmi < 25) bmiBins['18.5-24.9']++;
            else if (bmi < 30) bmiBins['25-29.9']++;
            else if (bmi < 35) bmiBins['30-34.9']++;
            else bmiBins['35+']++;
        }
    });

    edaStats = { ageBins, bmiBins };
}

function drawEdaCharts() {
    if (!edaStats) return;

    // Age chart
    const ageCtx = document.getElementById('edaAgeChart').getContext('2d');
    const ageLabels = Object.keys(edaStats.ageBins);
    const ageValues = ageLabels.map(l => edaStats.ageBins[l]);

    if (charts.edaAge) charts.edaAge.destroy();

    charts.edaAge = new Chart(ageCtx, {
        type: 'bar',
        data: {
            labels: ageLabels,
            datasets: [{
                label: 'Count by Age Group',
                data: ageValues,
                backgroundColor: 'rgba(99, 102, 241, 0.75)',
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(148, 163, 184, 0.15)' },
                    ticks: { color: '#e5e7eb' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#e5e7eb' }
                }
            }
        }
    });

    // BMI chart
    const bmiCtx = document.getElementById('edaBmiChart').getContext('2d');
    const bmiLabels = Object.keys(edaStats.bmiBins);
    const bmiValues = bmiLabels.map(l => edaStats.bmiBins[l]);

    if (charts.edaBmi) charts.edaBmi.destroy();

    charts.edaBmi = new Chart(bmiCtx, {
        type: 'bar',
        data: {
            labels: bmiLabels,
            datasets: [{
                label: 'Count by BMI Category',
                data: bmiValues,
                backgroundColor: 'rgba(34, 197, 94, 0.75)',
                borderColor: 'rgba(34, 197, 94, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(148, 163, 184, 0.15)' },
                    ticks: { color: '#e5e7eb' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#e5e7eb' }
                }
            }
        }
    });
}

// ==================== TRAINING ====================
async function trainModels() {
    if (csvData.length === 0) {
        alert('Please upload a CSV file first');
        return;
    }

    document.getElementById('trainBtn').disabled = true;
    document.getElementById('trainingStatus').innerHTML =
        '<div class="loading" style="display: inline-block;"></div> Training models...';
    document.getElementById('trainingProgress').classList.remove('hidden');

    try {
        const sampleSize = Math.min(5000, csvData.length);
        const sampledData = csvData.slice(0, sampleSize);
        
        const features = sampledData.map(row => 
            featureNames.map(f => {
                const val = parseFloat(row[f]);
                return isNaN(val) ? 0 : val;
            })
        );
        const labels = sampledData.map(row => parseInt(row.diabetes) || 0);

        const splitIdx = Math.floor(features.length * 0.8);
        const xTrain = tf.tensor2d(features.slice(0, splitIdx));
        const yTrain = tf.tensor2d(labels.slice(0, splitIdx), [splitIdx, 1]);
        const xTest = tf.tensor2d(features.slice(splitIdx));
        const yTest = tf.tensor2d(labels.slice(splitIdx), [labels.length - splitIdx, 1]);

        document.getElementById('trainingProgress').innerHTML = '📊 Training Logistic Regression...';

        const logModel = tf.sequential({
            layers: [
                tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [8] })
            ]
        });
        logModel.compile({ 
            optimizer: 'adam', 
            loss: 'binaryCrossentropy', 
            metrics: ['accuracy'] 
        });
        
        await logModel.fit(xTrain, yTrain, { 
            epochs: 20,
            verbose: 0,
            batchSize: 32
        });

        document.getElementById('trainingProgress').innerHTML = '🧠 Training Neural Network...';

        const nnModel = tf.sequential({
            layers: [
                tf.layers.dense({ units: 16, activation: 'relu', inputShape: [8] }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 8, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 1, activation: 'sigmoid' })
            ]
        });
        nnModel.compile({ 
            optimizer: 'adam', 
            loss: 'binaryCrossentropy', 
            metrics: ['accuracy'] 
        });
        
        await nnModel.fit(xTrain, yTrain, { 
            epochs: 20,
            verbose: 0,
            batchSize: 32
        });

        document.getElementById('trainingProgress').innerHTML = '📈 Evaluating models...';

        const logEval = logModel.evaluate(xTest, yTest);
        const nnEval = nnModel.evaluate(xTest, yTest);

        const logLoss = (await logEval[0].data())[0];
        const logAcc = (await logEval[1].data())[0];
        const nnLoss = (await nnEval[0].data())[0];
        const nnAcc = (await nnEval[1].data())[0];

        const weights = logModel.getWeights()[0];
        logisticWeights = Array.from(await weights.data());

        models.logistic = logModel;
        models.neuralNet = nnModel;

        document.getElementById('logAccuracy').textContent = (logAcc * 100).toFixed(2) + '%';
        document.getElementById('logLoss').textContent = logLoss.toFixed(4);
        document.getElementById('nnAccuracy').textContent = (nnAcc * 100).toFixed(2) + '%';
        document.getElementById('metricsContainer').classList.remove('hidden');
        document.getElementById('featureChartContainer').classList.remove('hidden');
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('trainingStatus').innerHTML =
            '<strong style="color: var(--accent);">✓ Models trained successfully!</strong>';
        document.getElementById('trainingProgress').classList.add('hidden');

        drawFeatureChart();

        xTrain.dispose();
        yTrain.dispose();
        xTest.dispose();
        yTest.dispose();
        logEval[0].dispose();
        logEval[1].dispose();
        nnEval[0].dispose();
        nnEval[1].dispose();
        weights.dispose();

    } catch (error) {
        console.error('Training error:', error);
        alert('Error training models: ' + error.message);
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('trainingStatus').innerHTML =
            '<strong style="color: var(--danger);">✗ Training failed</strong>';
        document.getElementById('trainingProgress').classList.add('hidden');
    }
}

// ==================== BMI HELPER ====================
function calculateBMI() {
    const weight = parseFloat(document.getElementById('weightKg').value);
    const height = parseFloat(document.getElementById('heightCm').value);

    if (isNaN(weight) || isNaN(height) || weight <= 0 || height <= 0) {
        alert('Please enter valid weight and height');
        return;
    }

    // Nếu height > 10 coi như cm → đổi sang m
    const heightMeters = height > 10 ? height / 100 : height;
    const bmi = weight / (heightMeters * heightMeters);
    const bmiInput = document.getElementById('bmi');
    bmiInput.value = bmi.toFixed(2);
}

// ==================== PREDICTION ====================
function handlePredict(event) {
    event.preventDefault();

    if (!models.neuralNet) {
        alert('Please train models first');
        return;
    }

    try {
        const hbaRaw = document.getElementById('hba1c').value;
        const bmiRaw = document.getElementById('bmi').value;

        const input = {
            gender: document.getElementById('gender').value === 'Female' ? 0 : 1,
            age: parseFloat(document.getElementById('age').value),
            hypertension: document.getElementById('hypertension').value === 'Yes' ? 1 : 0,
            heart_disease: document.getElementById('heartDisease').value === 'Yes' ? 1 : 0,
            smoking_history: smokingMap[document.getElementById('smokingHistory').value],
            bmi: bmiRaw === '' ? NaN : parseFloat(bmiRaw),
            HbA1c_level: hbaRaw === '' ? NaN : parseFloat(hbaRaw),
            blood_glucose_level: parseFloat(document.getElementById('bloodGlucose').value)
        };

        const riskProbability = predictRiskFromInput(input);

        displayResult(riskProbability, input);

    } catch (error) {
        alert('Error making prediction: ' + error.message);
    }
}

// helper để predict từ input (dùng lại cho What-if)
function predictRiskFromInput(inputObj) {
    // Bảo vệ khi thiếu BMI hoặc HbA1c: thay NaN = 0 để model vẫn chạy
    const safeInput = { ...inputObj };
    featureNames.forEach(f => {
        if (typeof safeInput[f] === 'number' && isNaN(safeInput[f])) {
            safeInput[f] = 0;
        }
    });

    const inputArray = featureNames.map(f => safeInput[f]);
    const inputTensor = tf.tensor2d([inputArray]);
    const prediction = models.neuralNet.predict(inputTensor);
    const riskProbability = Array.from(prediction.dataSync())[0];
    inputTensor.dispose();
    prediction.dispose();
    return riskProbability;
}

// ==================== RESULT DISPLAY ====================
function displayResult(probability, input) {
    const riskLevel = probability > 0.5 ? 'high' : probability > 0.3 ? 'moderate' : 'low';
    const riskPercentage = (probability * 100).toFixed(2);

    document.getElementById('riskProbability').textContent = riskPercentage + '%';
    document.getElementById('progressFill').style.setProperty('--progress-width', riskPercentage + '%');
    document.getElementById('riskLevel').textContent = riskLevel;

    const badge = document.getElementById('resultBadge');
    badge.textContent = riskLevel.toUpperCase();
    badge.className = 'result-badge ' + (riskLevel === 'high' ? 'high' : 'negative');

    const riskFactors = [];

    // HbA1c (optional)
    if (!isNaN(input.HbA1c_level)) {
        if (input.HbA1c_level > 6.5) {
            riskFactors.push({
                name: 'High HbA1c Level',
                value: input.HbA1c_level,
                type: 'risk',
                comment: 'High (≥ 6.5, diabetes range)'
            });
        } else if (input.HbA1c_level < 5.7) {
            riskFactors.push({
                name: 'Normal HbA1c',
                value: input.HbA1c_level,
                type: 'protective',
                comment: 'Normal (< 5.7)'
            });
        } else {
            riskFactors.push({
                name: 'Borderline HbA1c',
                value: input.HbA1c_level,
                type: 'risk',
                comment: 'Elevated (5.7–6.4, pre-diabetes range)'
            });
        }
    } else {
        riskFactors.push({
            name: 'HbA1c',
            value: 'Not provided',
            type: 'protective',
            comment: 'No HbA1c value entered, model used other features only'
        });
    }

    // Blood glucose
    if (input.blood_glucose_level > 125) {
        riskFactors.push({
            name: 'High Blood Glucose',
            value: input.blood_glucose_level,
            type: 'risk',
            comment: 'High (≥ 126 mg/dL)'
        });
    } else if (input.blood_glucose_level < 100) {
        riskFactors.push({
            name: 'Normal Blood Glucose',
            value: input.blood_glucose_level,
            type: 'protective',
            comment: 'Normal (< 100 mg/dL)'
        });
    } else {
        riskFactors.push({
            name: 'Borderline Blood Glucose',
            value: input.blood_glucose_level,
            type: 'risk',
            comment: 'Elevated (100–125 mg/dL)'
        });
    }

    // BMI (optional)
    if (!isNaN(input.bmi)) {
        if (input.bmi >= 30) {
            const bmiComment = input.bmi >= 35 ? 'Severely obese (≥ 35)' : 'Obese (30–34.9)';
            riskFactors.push({
                name: 'High BMI',
                value: input.bmi,
                type: 'risk',
                comment: bmiComment
            });
        } else if (input.bmi >= 25) {
            riskFactors.push({
                name: 'Overweight BMI',
                value: input.bmi,
                type: 'risk',
                comment: 'Overweight (25–29.9)'
            });
        } else if (input.bmi >= 18.5) {
            riskFactors.push({
                name: 'Healthy BMI',
                value: input.bmi,
                type: 'protective',
                comment: 'Healthy range (18.5–24.9)'
            });
        } else {
            riskFactors.push({
                name: 'Low BMI',
                value: input.bmi,
                type: 'risk',
                comment: 'Underweight (< 18.5)'
            });
        }
    } else {
        riskFactors.push({
            name: 'BMI',
            value: 'Not provided',
            type: 'protective',
            comment: 'No BMI value entered, you can calculate it using weight and height'
        });
    }

    // Hypertension
    if (input.hypertension === 1) {
        riskFactors.push({
            name: 'Hypertension Present',
            value: 'Yes',
            type: 'risk',
            comment: 'Known cardiovascular risk factor'
        });
    } else {
        riskFactors.push({
            name: 'No Hypertension',
            value: 'No',
            type: 'protective',
            comment: 'No diagnosed high blood pressure'
        });
    }

    // Heart disease
    if (input.heart_disease === 1) {
        riskFactors.push({
            name: 'Heart Disease Present',
            value: 'Yes',
            type: 'risk',
            comment: 'Significant cardiovascular risk factor'
        });
    } else {
        riskFactors.push({
            name: 'No Heart Disease',
            value: 'No',
            type: 'protective',
            comment: 'No diagnosed heart disease reported'
        });
    }

    // Age
    let ageComment;
    if (input.age < 35) ageComment = 'Younger adult (lower age-related risk)';
    else if (input.age < 45) ageComment = 'Mid-age adult (moderate age-related risk)';
    else if (input.age < 60) ageComment = 'Higher age-related risk (45–59)';
    else ageComment = 'Older age group (≥ 60, higher risk)';

    riskFactors.push({
        name: 'Age',
        value: input.age,
        type: input.age >= 45 ? 'risk' : 'protective',
        comment: ageComment
    });

    // Render risk factors list
    const factorsList = document.getElementById('riskFactorsList');
    factorsList.innerHTML = riskFactors.map(factor => `
        <li class="feature-item">
            <div class="feature-left">
                <span class="feature-name">${factor.name}</span>
                <span class="feature-comment">${factor.comment}</span>
            </div>
            <span class="feature-value ${factor.type === 'risk' ? 'risk-factor' : 'protective-factor'}">
                ${typeof factor.value === 'number'
                    ? factor.value.toFixed(2)
                    : factor.value}
            </span>
        </li>
    `).join('');

    document.getElementById('resultContainer').classList.remove('hidden');

    // What-if scenarios
    renderWhatIfScenarios(probability, input);
}

// ==================== WHAT-IF SCENARIOS ====================
function renderWhatIfScenarios(baseProb, input) {
    if (!models.neuralNet) return;

    const whatIfList = document.getElementById('whatIfList');
    const whatIfContainer = document.getElementById('whatIfContainer');

    const scenarios = [];

    // Scenario 1: BMI = 25 (chỉ nếu user đã nhập hoặc có thể suy ra từ dataset, nhưng logic ở đây chỉ cần giả định)
    const targetBmi = 25;
    const scenarioInputBmi = { ...input, bmi: targetBmi };
    const scenarioProbBmi = predictRiskFromInput(scenarioInputBmi);
    const deltaBmi = (baseProb - scenarioProbBmi) * 100;
    const scenarioPctBmi = (scenarioProbBmi * 100).toFixed(2);

    scenarios.push({
        title: `If your BMI were ${targetBmi.toFixed(1)}`,
        text: `Risk ≈ ${scenarioPctBmi}%. Change compared to now: ${deltaBmi >= 0 ? '↓' : '↑'} ${Math.abs(deltaBmi).toFixed(2)} percentage points.`
    });

    // Scenario 2: HbA1c = 6.0 (chỉ meaningful nếu có HbA1c trong model, kể cả khi user không nhập)
    const targetHb = 6.0;
    const scenarioInputHb = { ...input, HbA1c_level: targetHb };
    const scenarioProbHb = predictRiskFromInput(scenarioInputHb);
    const deltaHb = (baseProb - scenarioProbHb) * 100;
    const scenarioPctHb = (scenarioProbHb * 100).toFixed(2);

    scenarios.push({
        title: 'If your HbA1c were 6.0',
        text: `Risk ≈ ${scenarioPctHb}%. Change compared to now: ${deltaHb >= 0 ? '↓' : '↑'} ${Math.abs(deltaHb).toFixed(2)} percentage points.`
    });

    whatIfList.innerHTML = scenarios.map(sc => `
        <li class="whatif-item">
            <span class="whatif-item-strong">${sc.title}</span><br>
            ${sc.text}
        </li>
    `).join('');

    whatIfContainer.classList.remove('hidden');
}

// ==================== INIT ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('✓ Diabetes Dashboard initialized');
});
