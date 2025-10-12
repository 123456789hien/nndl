let trainData, testData;

async function loadMNIST() {
  const mnist = new MnistData();
  await mnist.load();
  trainData = { xs: mnist.getTrainData().images, labels: mnist.getTrainData().labels };
  testData = { xs: mnist.getTestData().images, labels: mnist.getTestData().labels };
  console.log("Data loaded:", trainData.xs.shape, testData.xs.shape);
}

class MnistData {
  constructor() {
    this.TRAIN_IMAGES = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
    this.TRAIN_LABELS = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
    this.IMAGE_SIZE = 28 * 28;
    this.NUM_CLASSES = 10;
    this.NUM_TRAIN_ELEMENTS = 55000;
    this.NUM_TEST_ELEMENTS = 10000;
  }

  async load() {
    const imgRequest = fetch(this.TRAIN_IMAGES);
    const labelsRequest = fetch(this.TRAIN_LABELS);

    const [imgResponse, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);
    const imgBuffer = await imgResponse.arrayBuffer();
    const labelsBuffer = await labelsResponse.arrayBuffer();

    const datasetBytes = new Uint8Array(imgBuffer);
    const labels = new Uint8Array(labelsBuffer);

    this.trainImages = tf.tensor4d(datasetBytes.slice(0, this.NUM_TRAIN_ELEMENTS * this.IMAGE_SIZE).map(v => v / 255), [this.NUM_TRAIN_ELEMENTS, 28, 28, 1]);
    this.trainLabels = tf.oneHot(tf.tensor1d(labels.slice(0, this.NUM_TRAIN_ELEMENTS), 'int32'), this.NUM_CLASSES);

    this.testImages = tf.tensor4d(datasetBytes.slice(this.NUM_TRAIN_ELEMENTS * this.IMAGE_SIZE).map(v => v / 255), [this.NUM_TEST_ELEMENTS, 28, 28, 1]);
    this.testLabels = tf.oneHot(tf.tensor1d(labels.slice(this.NUM_TRAIN_ELEMENTS), 'int32'), this.NUM_CLASSES);
  }

  getTrainData() { return { images: this.trainImages, labels: this.trainLabels }; }
  getTestData() { return { images: this.testImages, labels: this.testLabels }; }
}
