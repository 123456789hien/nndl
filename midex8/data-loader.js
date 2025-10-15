let trainData = [];
let testData = [];
let mergedData = [];

function parseCSV(file, callback) {
  Papa.parse(file, {
    header: true,
    delimiter: ";",
    skipEmptyLines: true,
    complete: results => {
      callback(results.data);
    }
  });
}

function loadCSVFiles(trainFile, testFile, onComplete) {
  parseCSV(trainFile, data => {
    trainData = data.map(d => ({
      hotel: d.ID,
      year: +d.year,
      month: +d.month,
      avg_room_price: +d.avg_room_price,
      lead_time: +d.lead_time,
      cancellation_rate: +d.status, 
    }));

    parseCSV(testFile, tData => {
      testData = tData.map(d => ({
        hotel: d.ID,
        year: +d.year,
        month: +d.month,
        avg_room_price: +d.avg_room_price,
        lead_time: +d.lead_time,
        cancellation_rate: +d.status,
      }));

      mergedData = [...trainData, ...testData].filter(d => !isNaN(d.month) && !isNaN(d.avg_room_price));
      onComplete();
    });
  });
}
