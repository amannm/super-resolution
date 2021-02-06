import * as tf from "@tensorflow/tfjs";
import { Tensor3D } from "@tensorflow/tfjs";

export class SuperResolutionModel {

    public static readonly SCALING_FACTOR = 4;

    private model: tf.GraphModel;
    private fromPixels2DContext: CanvasRenderingContext2D;

    private constructor(model: tf.GraphModel) {
        this.model = model;
        this.fromPixels2DContext = document.createElement('canvas').getContext('2d');
    }

    public static async open(path: string): Promise<SuperResolutionModel> {
        const graphModel = await tf.loadGraphModel(path);
        return new SuperResolutionModel(graphModel);
    }

    public close(): void {
        this.model.dispose();
    }

    public async resolve(inputBitmap: ImageBitmap): Promise<ImageBitmap> {
        const t0 = performance.now();
        const outputImageWidth = inputBitmap.width * SuperResolutionModel.SCALING_FACTOR;
        const outputImageHeight = inputBitmap.height * SuperResolutionModel.SCALING_FACTOR;
        const highResolutionImage = tf.tidy(() => {
            const input = this.copyToImageTensor(inputBitmap);
            const floatInput = tf.cast(input, "float32");
            const batchedInput = tf.expandDims(floatInput) as tf.Tensor4D;
            const prediction = this.model.predict(batchedInput) as tf.Tensor4D;
            const unbatchedOutput = tf.squeeze(prediction) as tf.Tensor3D;
            const clippedOutput = tf.clipByValue(unbatchedOutput, 0, 255);
            return tf.cast(clippedOutput, "int32");
        });
        const outputBitmap = await this.copyToImageBitmap(highResolutionImage, outputImageWidth, outputImageHeight);
        highResolutionImage.dispose();
        const t1 = performance.now();
        console.log(`resolution took ${t1 - t0} milliseconds.`);
        console.log(tf.memory());
        return outputBitmap;
    }

    private copyToImageTensor(imageBitmap: ImageBitmap): Tensor3D {
        this.fromPixels2DContext.canvas.width = imageBitmap.width;
        this.fromPixels2DContext.canvas.height = imageBitmap.height;
        this.fromPixels2DContext.drawImage(imageBitmap, 0, 0);
        const imageData = this.fromPixels2DContext.getImageData(0, 0, imageBitmap.width, imageBitmap.height);
        return tf.browser.fromPixels(imageData);
    }

    private async copyToImageBitmap(imageTensor: Tensor3D, width: number, height: number): Promise<ImageBitmap> {
        const imageDataArray = await tf.browser.toPixels(imageTensor);
        const imageData = new ImageData(imageDataArray, width, height);
        return await createImageBitmap(imageData);
    }
}