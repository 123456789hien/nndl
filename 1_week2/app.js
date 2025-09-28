// ========================
//  app.js
//  (Giữ format gốc của Thầy – chỉ sửa lỗi load data & chart)
// ========================

// Biến toàn cục
let trainData = [];
let testData = [];
let model = null;

// ========================
// 1. Đọc và load dữ liệu
// ========================
function parseCSVFile(file, callback) {
    const reader = new FileReader();
    reader.onload = e => {
        const text = e.target.result;
        // tách các dòng, bỏ dòng trống
        const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);

        // lấy header
        const headers = lines[0]
            .split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/) // regex giữ dấu phẩy trong dấu nháy
            .map(h => h.trim().replace(/"/g, ''));

        // lấy dữ liệu
        const data = lines.slice(1).map(line => {
            const cols = line
                .split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/)
                .map(c => c.trim().replace(/^"|"$/g, ''));
            const obj = {};
            headers.forEach((h, i) => obj[h] = cols[i]);
            return obj;
        });

        console.log("Loaded rows:", data.length);
        callback(data);
    };
    reader.onerror = () => alert("Error reading file");
    reader.readAsText(file);
}

function loadData() {
    const trainFile = document.getElementById("trainFile").files[0];
    const testFile = document.getElementById("testFile").files[0];

    if (!trainFile || !testFile) {
        alert("Please upload both train and test CSV files!");
        return;
    }

    parseCSVFile(trainFile, data => {
        trainData = data;
        alert("Train data loaded!");
        console.log("First rows of train:", trainData.slice(0, 3));
    });

    parseCSVFile(testFile, data => {
        testData = data;
        alert("Test data loaded!");
    });
}

// ========================
// 2. Xem trước dữ liệu và chart
// ========================
function inspectData() {
    if (trainData.length === 0) {
        alert("Please load data first!");
        return;
    }

    const previewDiv = document.getElementById("dataPreview");

    const headers = Object.keys(trainData[0]);
    let html = `<table border="1" cellspacing="0" cellpadding="4"><tr>`;
    headers.forEach(h => html += `<th>${h}</th>`);
    html += `</tr>`;

    trainData.slice(0, 5).forEach(row => {
        html += `<tr>`;
        headers.forEach(h => html += `<td>${row[h]}</td>`);
        html += `</tr>`;
    });
    html += `</table>`;

    previewDiv.innerHTML = html;

    // Chart: Survival by Sex
    const bySex = { male: 0, female: 0 };
    const byPclass = { 1: 0, 2: 0, 3: 0 };

    trainData.forEach(r => {
        if (r.Survived === '1') {
            if (r.Sex === 'male') bySex.male++;
            if (r.Sex === 'female') bySex.female++;
            byPclass[r.Pclass] = (byPclass[r.Pclass] || 0) + 1;
        }
    });

    tfvis.render.barchart(
        { name: 'Survival by Sex', tab: 'Charts' },
        [
            { name: 'male', value: bySex.male },
            { name: 'female', value: bySex.female }
        ]
    );

    tfvis.render.barchart(
        { name: 'Survival by Pclass', tab: 'Charts' },
        Object.keys(byPclass).map(k => ({ name: k, value: byPclass[k] }))
    );
}

// ========================
// 3. Tiền xử lý dữ liệu
// ========================
function preprocessData(rawData) {
    const xs = [];
    const ys = [];

    rawData.forEach(r => {
        if (r.Age && r.Sex && r.Pclass) {
            const age = parseFloat(r.Age);
            const sex = r.Sex === 'male' ? 0 : 1;
            const pclass = parseInt(r.Pclass);
            const survived = parseInt(r.Survived);

            xs.push([age, sex, pclass]);
            ys.push(survived);
        }
    });

    const xsTensor = tf.tensor2d(xs);
    const ysTensor = tf.tensor2d(ys, [ys.length, 1]);

    return { xs: xsTensor, ys: ysTensor };
}

// ========================
// 4. Tạo model
// ========================
function createModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [3], units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    alert("Model created!");
}

// ========================
// 5. Huấn luyện model
// ========================
async function trainModel() {
    if (!model) {
        alert("Please create the model first!");
        return;
    }
    if (trainData.length === 0) {
        alert("Please load and preprocess data first!");
        return;
    }

    const { xs, ys } = preprocessData(trainData);

    await model.fit(xs, ys, {
        epochs: 20,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Training' },
            ['loss', 'acc'],
            { height: 300, callbacks: ['onEpochEnd'] }
        )
    });

    alert("Training completed!");
}
