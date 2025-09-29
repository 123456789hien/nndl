// app.js
let model;
let trainXs, trainYs, valXs, valYs, testXs, testYs;

async function onLoadData() {
  const trainFile = document.getElementById('train-file').files[0];
  const testFile = document.getElementById('test-file').files[0];

  if (!trainFile || !testFile) {
    alert("Please select both training and test CSV files.");
    return;
  }

  document.getElementById('status').textContent = "⏳ Loading data...";

  try {
    const trainData = await loadTrainFromFiles(trainFile);
    const testData = await loadTestFromFiles(testFile);

    const split = splitTrainVal(trainData.xs, trainData.ys, 0.1);
    trainXs = split.trainXs;
    trainYs = split.trainYs;
    valXs = split.valXs;
    valYs = split.valYs;
    testXs = testData.xs;
    testYs = testData.ys;

    console.log("✅ Train:", trainXs.shape, trainYs.shape);
    console.log("✅ Val:", valXs.shape, valYs.shape);
    console.log("✅ Test:", testXs.shape, testYs.shape);

    document.getElementById('status').textContent = 
      `✅ Data loaded: ${trainXs.shape[0]} train, ${valXs.shape[0]} val, ${testXs.shape[0]} test`;

    // preview ảnh đầu tiên
    const canvas = document.getElementById('preview');
    draw28x28ToCanvas(trainXs.slice([0,0,0,0],[1,28,28,1]).reshape([28,28]), canvas, 4);
  } catch (err) {
    console.error(err);
    document.getElementById('status').textContent = "❌ Failed to load data.";
  }
}

function createModel() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({
    inputShape: [28,28,1],
    filters: 16, kernelSize: 3, activation: 'relu'
  }));
  m.add(tf.layers.maxPooling2d({poolSize: 2}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({units: 64, activation: 'relu'}));
  m.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  m.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return m;
}

async function onTrain() {
  if (!trainXs) {
    alert("Please load data first.");
    return;
  }

  model = createModel();
  document.getElementById('status').textContent = "⏳ Training...";

  await model.fit(trainXs, trainYs, {
    epochs: 5,
    validationData: [valXs, valYs],
    batchSize: 64,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc}`);
      }
    }
  });

  document.getElementById('status').textContent = "✅ Training complete!";
}

async function onTest() {
  if (!model) {
    alert("Train model first.");
    return;
  }

  const evalResult = model.evaluate(testXs, testYs);
  const loss = evalResult[0].dataSync()[0].toFixed(4);
  const acc = (evalResult[1].dataSync()[0]*100).toFixed(2);

  document.getElementById('status').textContent = `📊 Test Accuracy: ${acc}% (loss=${loss})`;
}
