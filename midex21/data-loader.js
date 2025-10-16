// Helper: prepare monthly aggregated data
function prepareMonthly(rawData) {
    const monthly = {};

    rawData.forEach(row => {
        const month = row.month;
        if (!monthly[month]) {
            monthly[month] = {
                cancellation_rate: 0,
                avg_room_price: 0,
                lead_time_avg: 0,
                count: 0
            };
        }
        monthly[month].cancellation_rate += isNaN(row.cancellation_rate) ? 0 : row.cancellation_rate;
        monthly[month].avg_room_price += isNaN(row.avg_room_price) ? 0 : row.avg_room_price;
        monthly[month].lead_time_avg += isNaN(row.lead_time_avg) ? 0 : row.lead_time_avg;
        monthly[month].count += 1;
    });

    const result = [];
    for (const month in monthly) {
        const data = monthly[month];
        result.push({
            month,
            cancellation_rate: +(data.cancellation_rate / data.count).toFixed(2),
            avg_room_price: +(data.avg_room_price / data.count).toFixed(2),
            lead_time_avg: +(data.lead_time_avg / data.count).toFixed(2)
        });
    }

    // Sort by month ascending
    result.sort((a, b) => new Date(a.month) - new Date(b.month));
    return result;
}

// Helper: predict next month
function predictNextMonth(data) {
    if (data.length === 0) return null;
    const last = data[data.length - 1];
    let [year, month] = last.month.split('-').map(Number);
    month += 1;
    if (month > 12) {
        month = 1;
        year += 1;
    }
    return { month: `${year}-${String(month).padStart(2,'0')}` };
}
