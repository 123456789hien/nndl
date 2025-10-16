// Main JS
let rawData = []; // assume data loaded via fetch/ajax
let monthlyData = [];

// Load dropdown filter
const dataRangeSelect = document.getElementById('dataRange');
dataRangeSelect.addEventListener('change', () => {
    updateCharts();
});

// Update charts based on selected filter
function updateCharts() {
    let filteredData = [...monthlyData];
    if (dataRangeSelect.value === '2018') {
        filteredData = filteredData.filter(d => d.month.startsWith('2018'));
    }

    drawCancellationLine(filteredData);
    drawPriceHist(filteredData);
    drawCorrHeatmap(filteredData);

    // Predict next month
    const nextMonth = predictNextMonth(filteredData);
    const predDiv = document.getElementById('predictNextMonth');
    predDiv.innerText = `Predicted next month: ${nextMonth ? nextMonth.month : 'N/A'}`;
}

// Draw line chart for cancellation
function drawCancellationLine(data) {
    const trace = {
        x: data.map(d => d.month),
        y: data.map(d => d.cancellation_rate),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Cancellation Rate',
        text: data.map(d => `Month: ${d.month}<br>Cancellation Rate: ${d.cancellation_rate}%`),
        hoverinfo: 'text'
    };
    Plotly.newPlot('cancellationLine', [trace], { margin: { t: 30 } });
}

// Draw histogram for avg room price
function drawPriceHist(data) {
    const trace = {
        x: data.map(d => d.avg_room_price),
        type: 'histogram',
        text: data.map(d => `Avg Room Price: $${d.avg_room_price}`),
        hoverinfo: 'text'
    };
    Plotly.newPlot('priceHist', [trace], { margin: { t: 30 } });
}

// Draw correlation heatmap
function drawCorrHeatmap(data) {
    const corr = [
        ['cancellation_rate', 'avg_room_price', 'lead_time_avg'],
        ['cancellation_rate', 'avg_room_price', 'lead_time_avg']
    ]; // placeholder: real correlation calculation

    const trace = {
        z: [
            [1, 0.2, 0.5],
            [0.2, 1, 0.3],
            [0.5, 0.3, 1]
        ],
        x: ['Cancellation', 'Price', 'LeadTime'],
        y: ['Cancellation', 'Price', 'LeadTime'],
        type: 'heatmap',
        text: [
            ['100%', '20%', '50%'],
            ['20%', '100%', '30%'],
            ['50%', '30%', '100%']
        ],
        hoverinfo: 'text'
    };
    Plotly.newPlot('corrHeatmap', [trace], { margin: { t: 30 } });
}

// On load
window.onload = () => {
    // Simulate data load
    rawData = loadData(); // assume loadData() returns full dataset
    monthlyData = prepareMonthly(rawData);
    updateCharts();
};
