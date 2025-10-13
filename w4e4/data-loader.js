// data-loader.js
class DataLoader {
    constructor() {
        this.stocksData = null;
        this.normalizedData = null;
        this.symbols = [];
        this.dates = [];
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
        this.testDates = [];
        this.features = ['Open','Close','High','Low','Volume','Return'];
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    this.parseCSV(csv);
                    resolve(this.stocksData);
                } catch (error) { reject(error); }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    async loadCSVfromURL(url) {
        try {
            const response = await fetch(url);
            if(!response.ok) throw new Error('CSV not found');
            const csvText = await response.text();
            this.parseCSV(csvText);
            return this.stocksData;
        } catch(err) { console.error(err); throw err; }
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');

        const data = {};
        const symbols = new Set();
        const dates = new Set();

        for(let i=1;i<lines.length;i++){
            const values = lines[i].split(',');
            if(values.length !== headers.length) continue;
            const row = {};
            headers.forEach((h, idx)=>row[h.trim()]=values[idx].trim());

            const symbol = row.Symbol;
            const date = row.Date;
            symbols.add(symbol);
            dates.add(date);

            if(!data[symbol]) data[symbol]={};
            data[symbol][date]={
                Open: parseFloat(row.Open),
                Close: parseFloat(row.Close),
                High: parseFloat(row.High),
                Low: parseFloat(row.Low),
                Volume: parseFloat(row.Volume)
            };
        }

        this.symbols = Array.from(symbols).sort();
        this.dates = Array.from(dates).sort();
        this.stocksData = data;
        console.log(`Loaded ${this.symbols.length} stocks, ${this.dates.length} days`);

        // Compute daily return
        this.symbols.forEach(sym=>{
            this.dates.forEach(date=>{
                const d = data[sym][date];
                if(d) d.Return=(d.Close-d.Open)/d.Open;
            });
        });
    }

    normalizeData() {
        if(!this.stocksData) throw new Error('No data loaded');
        this.normalizedData={};
        const minMax={};

        this.symbols.forEach(sym=>{
            minMax[sym]={};
            this.features.forEach(f=>{
                minMax[sym][f]={min:Infinity,max:-Infinity};
            });
            this.dates.forEach(date=>{
                const d=this.stocksData[sym][date];
                if(!d) return;
                this.features.forEach(f=>{
                    minMax[sym][f].min=Math.min(minMax[sym][f].min,d[f]);
                    minMax[sym][f].max=Math.max(minMax[sym][f].max,d[f]);
                });
            });
        });

        this.symbols.forEach(sym=>{
            this.normalizedData[sym]={};
            this.dates.forEach(date=>{
                const d=this.stocksData[sym][date];
                if(!d) return;
                this.normalizedData[sym][date]={};
                this.features.forEach(f=>{
                    const range=minMax[sym][f].max-minMax[sym][f].min;
                    this.normalizedData[sym][date][f]=range>0?(d[f]-minMax[sym][f].min)/range:0.5;
                });
            });
        });
        return this.normalizedData;
    }

    createSequences(sequenceLength=12,predictionHorizon=3){
        if(!this.normalizedData) this.normalizeData();

        const sequences=[]; const targets=[]; const validDates=[];
        for(let i=sequenceLength;i<this.dates.length-predictionHorizon;i++){
            const seqDate=this.dates[i]; let valid=true;
            const seqData=[];
            for(let j=sequenceLength-1;j>=0;j--){
                const d=this.dates[i-j]; const stepData=[];
                this.symbols.forEach(sym=>{
                    const point=this.normalizedData[sym][d];
                    if(!point) valid=false;
                    else this.features.forEach(f=>stepData.push(point[f]));
                });
                if(valid) seqData.push(stepData);
            }

            if(valid){
                const target=[];
                const baseClose=this.symbols.map(s=>this.stocksData[s][seqDate].Close);
                for(let offset=1;offset<=predictionHorizon;offset++){
                    const fDate=this.dates[i+offset];
                    this.symbols.forEach((sym,idx)=>{
                        const fClose=this.stocksData[sym][fDate].Close;
                        target.push(fClose>baseClose[idx]?1:0);
                    });
                }
                sequences.push(seqData);
                targets.push(target);
                validDates.push(seqDate);
            }
        }

        const split=Math.floor(sequences.length*0.8);
        this.X_train=tf.tensor3d(sequences.slice(0,split));
        this.y_train=tf.tensor2d(targets.slice(0,split));
        this.X_test=tf.tensor3d(sequences.slice(split));
        this.y_test=tf.tensor2d(targets.slice(split));
        this.testDates=validDates.slice(split);
        console.log(`Sequences: ${sequences.length}, Train: ${this.X_train.shape[0]}, Test: ${this.X_test.shape[0]}`);
        return {X_train:this.X_train,y_train:this.y_train,X_test:this.X_test,y_test:this.y_test,symbols:this.symbols,testDates:this.testDates};
    }

    dispose(){
        if(this.X_train) this.X_train.dispose();
        if(this.y_train) this.y_train.dispose();
        if(this.X_test) this.X_test.dispose();
        if(this.y_test) this.y_test.dispose();
    }
}

export default DataLoader;
