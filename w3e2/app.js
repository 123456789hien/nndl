let modelCNN, modelDenoiser;
const logs = document.getElementById('logs');

document.getElementById('loadDataBtn').onclick = async () => {
  logs.innerText += "Loading MNIST data...\n";
  await loadMNIST();
  logs.innerText += "Data loaded.\n";
};

document.getElementById('trainBtn').onclick = async () => {
  if (!trainData) return alert("Load data first.");
  logs.innerText += "Training CNN...\n";

  modelCNN = tf.sequential();
  modelCNN.add(tf.layers.conv2d({ inputShape:[28,28,1], filters:32, kernelSize:3, activation:'relu' }));
  modelCNN.add(tf.layers.maxPooling2d({ poolSize:2 }));
  modelCNN.add(tf.layers.conv2d({ filters:64, kernelSize:3, activation:'relu' }));
  modelCNN.add(tf.layers.maxPooling2d({ poolSize:2 }));
  modelCNN.add(tf.layers.flatten());
  modelCNN.add(tf.layers.dense({ units:128, activation:'relu' }));
  modelCNN.add(tf.layers.dense({ units:10, activation:'softmax' }));

  modelCNN.compile({ optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy'] });

  await modelCNN.fit(trainData.xs, trainData.labels, {
    epochs:3,
    batchSize:64,
    validationSplit:0.1,
    callbacks: {
      onEpochEnd: async (epoch, logs_) => {
        const acc = logs_['accuracy'] ?? logs_['acc'] ?? 0;
        logs.innerText += `Epoch ${epoch+1}: loss=${logs_.loss.toFixed(4)}, acc=${acc.toFixed(4)}\n`;
        logs.scrollTop = logs.scrollHeight;
      }
    }
  });

  logs.innerText += "CNN Training complete.\n";
};

function addNoise(tensor, noiseLevel=0.25) {
  return tf.tidy(() => tensor.add(tf.randomNormal(tensor.shape, 0, noiseLevel)).clipByValue(0,1));
}

document.getElementById('trainDenoiserBtn').onclick = async () => {
  if (!trainData) return alert("Load data first.");

  logs.innerText += "Training Denoiser...\n";

  const nSamples = Math.min(1000, trainData.xs.shape[0]);
  const noisy = tf.tidy(() => addNoise(trainData.xs.slice([0,0,0,0],[nSamples,28,28,1]),0.25));
  const clean = trainData.xs.slice([0,0,0,0],[nSamples,28,28,1]);

  modelDenoiser = tf.sequential();
  modelDenoiser.add(tf.layers.conv2d({ inputShape:[28,28,1], filters:32, kernelSize:3, activation:'relu', padding:'same' }));
  modelDenoiser.add(tf.layers.conv2d({ filters:32, kernelSize:3, activation:'relu', padding:'same' }));
  modelDenoiser.add(tf.layers.conv2d({ filters:1, kernelSize:3, activation:'sigmoid', padding:'same' }));

  modelDenoiser.compile({ optimizer:'adam', loss:'meanSquaredError' });

  await modelDenoiser.fit(noisy, clean, {
    epochs:3,
    batchSize:64,
    validationSplit:0.1,
    callbacks: tfvis.show.fitCallbacks({name:'Denoiser Training', tab:'Training'}, ['loss','val_loss'])
  });

  logs.innerText += "Denoiser Training complete.\n";
  tf.dispose([noisy, clean]);
};

document.getElementById('predictBtn').onclick = async () => {
  if (!modelCNN) return alert("Train CNN first.");
  const sample = testData.xs.slice([0,0,0,0],[1,28,28,1]);
  const pred = modelCNN.predict(sample);
  pred.array().then(array => {
    logs.innerText += `Prediction: ${array[0].map(a=>a.toFixed(3)).join(', ')}\n`;
  });
};
