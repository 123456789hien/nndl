// app.js
import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.currentPredictions = null;
        this.accuracyChart = null;
        this.isTraining = false;

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const fileInput = document.getElementById('csvFile');
        const trainBtn = document.getElementById('trainBtn');
        const predictBtn = document.getElementById('predictBtn');

        fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        trainBtn.addEventListener('click', () => this.trainModel());
        predictBtn.addEventListener('click', () => this.runPrediction());
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            document.getElementById('status').textContent = 'Loading CSV...';
            await this.dataLoader.loadCSV(file);

            document.getElementById('status').textContent = 'Preprocessing data...';
            this.dataLoader.createSequences();

            document.getElementById('trainBtn').disabled = false;
            document.getElementById('status').textContent = 'Data loaded. Click Train Model to begin training.';
        } catch (error) {
            document.getElementById('status').textContent = `Error: ${error.message}`;
            console.error(error);
        }
    }

    async trainModel() {
        if (this.isTraining) return;

        this.isTraining = true;
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('predictBtn').disabled = true;

        try {
            const { X_train, y_train, X_test, y_test, symbols } = this.dataLoader;

            this.model = new GRUModel([12, symbols.length*2], symbols.length*3);

            document.getElementById('status').textContent = 'Training model...';
            await this.model.train(X_train, y_train, X_test, y_test, 50, 32);

            document.getElementById('predictBtn').disabled = false;
            document.getElementById('status').textContent = 'Training completed. Click Run Prediction to evaluate.';
        } catch (error) {
            document.getElementById('status').textContent = `Training error: ${error.message}`;
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    async runPrediction() {
        if (!this.model) {
            alert('Please train the model first');
            return;
        }

        try {
            document.getElementById('status').textContent = 'Running predictions...';
            const { X_test, y_test, symbols } = this.dataLoader;

            const predictions = await this.model.predict(X_test);
            const evaluation = this.model.evaluatePerStock(y_test, predictions, symbols);

            this.currentPredictions = evaluation;
            this.visualizeResults(evaluation, symbols);

            document.getElementById('status').
