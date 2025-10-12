class MNISTData {
    constructor() {
        this.trainImages = null;
        this.trainLabels = null;
        this.testImages = null;
        this.testLabels = null;
    }

    async load() {
        const mnist = await fetch('https://storage.googleapis.com/tfjs-examples/mnist/data/mnist.json')
            .then(res => res.json());

        const processImages = (images) => {
            return tf.tensor4d(images, [images.length, 28, 28, 1]).div(255.0);
        }

        const processLabels = (labels) => {
            return tf.tensor2d(labels, [labels.length, 10]);
        }

        this.trainImages = processImages(mnist.train_images);
        this.trainLabels = processLabels(mnist.train_labels);
        this.testImages = processImages(mnist.test_images);
        this.testLabels = processLabels(mnist.test_labels);
    }

    getTrainData() {
        return { xs: this.trainImages, ys: this.trainLabels };
    }

    getTestData() {
        return { xs: this.testImages, ys: this.testLabels };
    }
}

const data = new MNISTData();
