<!-- index.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>TF.js GRU Stock Prediction Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; }
    #controls { display:flex; gap:12px; align-items:center; margin-bottom:12px; flex-wrap:wrap; }
    #accuracy-container, #timeline-container { margin-top: 16px; }
    progress { width: 300px; height: 18px; }
    button { padding: 6px 12px; }
  </style>
</head>
<body>
  <h2>Browser-based GRU Stock Prediction (TF.js)</h2>
  <div id="controls">
    <input id="csv-file" type="file" accept=".csv" />
    <button id="btn-load">Load CSV</button>
    <button id="btn-train" disabled>Train Model</button>
    <button id="btn-download">Save Model (download)</button>
    <label><progress id="progress" value="0" max="100"></progress></label>
    <div id="status">Idle</div>
  </div>

  <div id="accuracy-container"></div>
  <div id="timeline-container"></div>

  <!-- TF.js from CDN -->
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.12.0/dist/tf.min.js"></script>

  <!-- Import modules -->
  <script type="module">
    import App from './app.js';
    // auto-init
    const app = App.initFromDOM();
    // expose for console debugging
    window._APP = app;
    // Optional: warn user to use modern browser
    if (!window.crypto || !window.crypto.subtle) {
      document.getElementById('status').innerText = 'Warning: For best results use a modern browser (Chrome/Edge/Firefox).';
    }
  </script>
</body>
</html>
