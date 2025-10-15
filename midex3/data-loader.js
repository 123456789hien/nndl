let trainData = null;
let testData = null;
let mergedData = null;

function setStatus(msg) { document.getElementById('load-status').innerText = msg; }

function normalizeRow(row) {
  const out = {};
  Object.keys(row).forEach(k => {
    let v = row[k];
    if (v === null || v === undefined) out[k] = null;
    else if (typeof v === 'string') out[k] = v.trim();
    else out[k] = v;
  });
  return out;
}

function loadData() {
  const trainFile = document.getElementById('train-file').files[0];
  const testFile = document.getElementById('test-file').files[0];
  if (!trainFile || !testFile) { alert('Please select both train & test CSV'); return; }

  setStatus('Parsing CSV files...');
  
  Papa.parse(trainFile, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: (results) => {
      trainData = results.data.map(normalizeRow);
      tryBuildMerged();
    }
  });

  Papa.parse(testFile, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: (results) => {
      testData = results.data.map(normalizeRow);
      tryBuildMerged();
    }
  });
}

function tryBuildMerged() {
  if (trainData && testData) {
    mergedData = trainData.concat(testData);
    setStatus(`Loaded train(${trainData.length}) + test(${testData.length})`);
    document.getElementById('run-eda-btn').disabled = false;
  }
}

document.getElementById('load-data-btn').addEventListener('click', loadData);
