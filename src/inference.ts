import * as tf from '@tensorflow/tfjs';

export class SuperResolutionModel {
    private model: tf.GraphModel;

    private constructor(model: tf.GraphModel) {
        this.model = model;
    }

    private predict(batches: tf.Tensor4D): tf.Tensor4D {
        return tf.tidy(() => ((this.model.predict(batches.toFloat()) as tf.Tensor4D).clipByValue(0, 255) as tf.Tensor4D).round().toInt());
    }

    public async resolveBatch(inputs: ImageData[]): Promise<ImageData[]> {
        if (inputs.length === 0) {
            return [];
        }
        const outputImageWidth = inputs[0].width * 4;
        const outputImageHeight = inputs[0].height * 4;
        const lowResolutionImages = tf.tidy(() => tf.stack(inputs.map(tf.browser.fromPixels)) as tf.Tensor4D);
        const highResolutionImages = tf.tidy(() => this.predict(lowResolutionImages).unstack() as tf.Tensor3D[]);
        lowResolutionImages.dispose();
        return Promise.all(highResolutionImages.map(highResolutionImage => tf.browser.toPixels(highResolutionImage).then(pixelData => {
            highResolutionImage.dispose();
            return new ImageData(pixelData, outputImageWidth, outputImageHeight);
        })));
    }

    public async resolve(input: ImageData): Promise<ImageData> {
        const outputImageWidth = input.width * 4;
        const outputImageHeight = input.height * 4;
        const lowResolutionImage = tf.tidy(() => tf.expandDims(tf.browser.fromPixels(input)) as tf.Tensor4D);
        const highResolutionImage = tf.tidy(() => this.predict(lowResolutionImage).squeeze() as tf.Tensor3D);
        lowResolutionImage.dispose();
        return tf.browser.toPixels(highResolutionImage).then(pixelData => {
            highResolutionImage.dispose();
            return new ImageData(pixelData, outputImageWidth, outputImageHeight);
        });
    }

    public static async open(path: string): Promise<SuperResolutionModel> {
        const graphModel = await tf.loadGraphModel(path);
        return new SuperResolutionModel(graphModel);
    }

    public close(): void {
        this.model.dispose();
    }
}