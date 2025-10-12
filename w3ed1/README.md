You are a senior front‑end engineer and ML instructor building a browser‑only TensorFlow.js MNIST + Denoising Autoencoder demo for students.

Context:

Build a GitHub Pages–deployable web app that TRAINS and RUNS entirely client‑side with TensorFlow.js and tfjs‑vis.

MNIST data will be provided by the user as two local CSV files via file inputs: mnist_train.csv and mnist_test.csv.

CSV format: each row = label (0–9) followed by 784 pixel values (0–255), no header.

Do NOT fetch data over the network; parse the uploaded files in the browser, normalize pixels to [0,1], reshape to [N,28,28,1], and one-hot encode labels to depth 10.

Implement FILE-BASED model Save/Load only: download model.json + weights.bin, and reload from user-selected files. Do not clear model info on load.

Include CNN training, Denoising Autoencoder training, evaluation with Confusion Matrix + Per-class Accuracy charts, and a random 5-image preview showing original, noisy, denoised images with predicted labels.

Instruction:
Output exactly three fenced code blocks, in this order, labeled index.html, data-loader.js, and app.js, implementing all features below without extra prose.

index.html

Include CDNs:

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>

Minimal CSS for two-column layout, horizontal preview strip, professional and responsive pink-themed UI.

Controls:

Upload inputs: train-csv, test-csv

Buttons: Load Data, Train CNN, Train Denoiser, Evaluate, Test 5 Random, Save Model, Load Model, Reset, Toggle Visor

Model load inputs: upload-json, upload-weights

Sections:

Data Status, Training Logs

Metrics (overall accuracy + charts)

Random 5 Preview (row of canvases + predicted labels)

Model Info (layers/params)

Defer-load data-loader.js then app.js.

data-loader.js

Implement file-based CSV parsing with FileReader/TextDecoder (no external libraries).

Parse each row: first value → label int, remaining 784 → pixels; ignore empty lines.

Normalize pixels /255, reshape [N,28,28,1], one-hot labels depth 10.

Provide:

async function loadTrainFromFiles(file) → {xs, ys}

async function loadTestFromFiles(file) → {xs, ys}

function splitTrainVal(xs, ys, valRatio=0.1) → {trainXs, trainYs, valXs, valYs}

function getRandomTestBatch(xs, ys, k=5) → tensors for preview

function addNoise(xs, std=0.25) → noisy images for denoising

function draw28x28ToCanvas(tensor, canvas, scale=4)

Dispose intermediate tensors to avoid memory leaks.

app.js

Wire UI:

onLoadData: read CSVs, build tensors, show counts

onTrainCNN: build CNN, train with tfjs-vis fitCallbacks

onTrainDenoiser: build CNN Autoencoder for denoising, train with noisy inputs

onEvaluate: compute test accuracy; render Confusion Matrix + Per-class Accuracy bar chart; print overall accuracy

onTestFive: sample 5 random test images; display original, noisy, denoised images with predicted labels

onSaveDownload: download model JSON + weights

onLoadFromFiles: load JSON + BIN; replace current model but do not erase model info, call model.summary()

onReset: dispose tensors/model, clear UI

onToggleVisor: open/close tfjs-vis visor

Model:

CNN

tf.sequential([
  Conv2D(32, 3, activation='relu', padding='same', inputShape:[28,28,1]),
  Conv2D(64, 3, activation='relu', padding='same'),
  MaxPool2D(2),
  Dropout(0.25),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(10, activation='softmax')
])


Compile: optimizer='adam', loss='categoricalCrossentropy', metrics=['accuracy']

Training defaults: epochs 5–10, batchSize 64–128, shuffle true

Denoising Autoencoder

Conv2D encoder + Conv2DTranspose decoder

Loss: meanSquaredError

Input: noisy images → Output: original images

Charts (tfjs-vis):

Live loss/val_loss and acc/val_acc during fit

Confusion matrix + per-class accuracy on evaluation

Performance & safety:

Use tf.tidy where appropriate; dispose old models/tensors

Try/catch around file handling and training

Ensure UI responsive with await and requestAnimationFrame for long operations

Formatting

Produce only three fenced code blocks labeled exactly index.html, data-loader.js, and app.js.

Browser-only JavaScript; no Node or extra libraries.

Include clear English comments explaining logic, models, and interactions.

Keep all UI/UX styling, layout, and functionality consistent with previous version.
