// data-loader.js
'use strict';

function readFileAsText(file) { /* unchanged */ ... }
async function loadCSVFile(file) { /* unchanged */ ... }
async function loadTrainFromFiles(file) { return loadCSVFile(file); }
async function loadTestFromFiles(file) { return loadCSVFile(file); }

function splitTrainVal(xs, ys, valRatio = 0.1) { /* unchanged */ ... }
function addNoise(xs, noiseStd = 0.25) { /* unchanged */ ... }
function getRandomTestBatch(xs, ys, k = 5) { /* unchanged */ ... }
function draw28x28ToCanvas(tensor, canvas, scale = 4) { /* unchanged */ ... }

window.loadTrainFromFiles = loadTrainFromFiles;
window.loadTestFromFiles = loadTestFromFiles;
window.splitTrainVal = splitTrainVal;
window.getRandomTestBatch = getRandomTestBatch;
window.draw28x28ToCanvas = draw28x28ToCanvas;
window.addNoise = addNoise;
