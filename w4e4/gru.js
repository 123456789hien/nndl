// gru.js
class GRUModel {
    constructor(inputShape, outputSize){
        this.inputShape=inputShape;
        this.outputSize=outputSize;
        this.model=null;
        this.history=null;
    }

    buildModel(){
        this.model=tf.sequential({
            layers:[
                tf.layers.conv1d({filters:32,kernelSize:3,activation:'relu',inputShape:this.inputShape}),
                tf.layers.dropout({rate:0.2}),
                tf.layers.bidirectional({
                    layer: tf.layers.gru({units:128,returnSequences:true}),
                    mergeMode:'concat'
                }),
                tf.layers.dropout({rate:0.3}),
                tf.layers.gru({units:64,returnSequences:false}),
                tf.layers.dropout({rate:0.3}),
                tf.layers.dense({units:this.outputSize,activation:'sigmoid'})
            ]
        });
        this.model.compile({
            optimizer: tf.train.adam(0.0005),
            loss:'binaryCrossentropy',
            metrics:['binaryAccuracy']
        });
        return this.model;
    }

    async train(X_train,y_train,X_test,y_test,epochs=80,batchSize=16){
        if(!this.model) this.buildModel();
        this.history=await this.model.fit(X_train,y_train,{
            epochs, batchSize, validationData:[X_test,y_test],
            callbacks:{
                onEpochEnd:(epoch,logs)=>{
                    const progress=((epoch+1)/epochs)*100;
                    const status=`Epoch ${epoch+1}/${epochs} - loss:${logs.loss.toFixed(4)}, acc:${logs.binaryAccuracy.toFixed(4)}, val_loss:${logs.val_loss.toFixed(4)}, val_acc:${logs.val_binaryAccuracy.toFixed(4)}`;
                    const progressElement=document.getElementById('trainingProgress');
                    const statusElement=document.getElementById('status');
                    if(progressElement) progressElement.value=progress;
                    if(statusElement) statusElement.textContent=status;
                    console.log(status);
                    tf.nextFrame();
                }
            }
        });
        return this.history;
    }

    async predict(X){
        if(!this.model) throw new Error('Model not trained');
        return this.model.predict(X);
    }

    evaluatePerStock(yTrue,yPred,symbols,horizon=3){
        const yTrueArr=yTrue.arraySync();
        const yPredArr=yPred.arraySync();
        const stockAccuracies={};
        const stockPredictions={};

        symbols.forEach((sym,idx)=>{
            let correct=0,total=0;
            const preds=[];
            for(let i=0;i<yTrueArr.length;i++){
                for(let off=0;off<horizon;off++){
                    const tIdx=idx*horizon+off;
                    const tVal=yTrueArr[i][tIdx];
                    const pVal=yPredArr[i][tIdx]>0.5?1:0;
                    if(tVal===pVal) correct++;
                    total++;
                    preds.push({true:tVal,pred:pVal,correct:tVal===pVal});
                }
            }
            stockAccuracies[sym]=correct/total;
            stockPredictions[sym]=preds;
        });
        return {stockAccuracies,stockPredictions};
    }

    dispose(){
        if(this.model) this.model.dispose();
    }
}

export default GRUModel;
