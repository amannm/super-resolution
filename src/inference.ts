import * as tf from "@tensorflow/tfjs";
import { Tensor3D } from "@tensorflow/tfjs";

export abstract class SuperResolutionModel {

    public static readonly SCALING_FACTOR = 4;

    private fromPixels2DContext: CanvasRenderingContext2D;

    protected constructor() {
        this.fromPixels2DContext = document.createElement('canvas').getContext('2d');
    }

    protected renderBitmap(imageBitmap: ImageBitmap): ImageData {
        this.fromPixels2DContext.canvas.width = imageBitmap.width;
        this.fromPixels2DContext.canvas.height = imageBitmap.height;
        this.fromPixels2DContext.drawImage(imageBitmap, 0, 0);
        return this.fromPixels2DContext.getImageData(0, 0, imageBitmap.width, imageBitmap.height);
    }

    public static async openLocal(modelPath: string): Promise<SuperResolutionModel> {
        const graphModel = await tf.loadGraphModel(modelPath);
        return new LocalSuperResolutionModel(graphModel);
    }

    public static async openRemote(serverRoot: string): Promise<SuperResolutionModel> {
        return new RemoteSuperResolutionModel(serverRoot);
    }

    public abstract resolve(inputBitmap: ImageBitmap): Promise<ImageBitmap>

    public abstract close(): void
}

class RemoteSuperResolutionModel extends SuperResolutionModel {

    private baseUrl: string;

    public constructor(baseUrl: string) {
        super()
        this.baseUrl = baseUrl;
    }
    
    public close(): void {
    }

    public async resolve(imageBitmap: ImageBitmap): Promise<ImageBitmap> {
        const t0 = performance.now();
        const imageData = this.renderBitmap(imageBitmap);
        const upscaledImageData = await this.upscale(imageData);
        const outputBitmap = await createImageBitmap(upscaledImageData);
        const t1 = performance.now();
        console.log(`resolution took ${t1 - t0} milliseconds.`);
        return outputBitmap;
    }

    private toRgb(rgbaArray: Uint8ClampedArray): Uint8ClampedArray {
        const destinationArray = new Uint8ClampedArray(rgbaArray.length * 3 / 4);
        const bandLength = destinationArray.length / 3;
        for (let i = 0, x = 0; i < rgbaArray.length; i += 4) {
            destinationArray[x] = rgbaArray[i];
            destinationArray[x+bandLength] = rgbaArray[i+1];
            destinationArray[x+bandLength*2] = rgbaArray[i+2];
            x++;
        } 
        return destinationArray;
    }

    private toRgba(rgbArray: Uint8ClampedArray): Uint8ClampedArray {
        const destinationArray = new Uint8ClampedArray(rgbArray.length * 4 / 3);
        const bandLength = rgbArray.length / 3;
        for (let i = 0, x = 0; i < bandLength; i++) {
            destinationArray[x++] = rgbArray[i];
            destinationArray[x++] = rgbArray[i+bandLength];
            destinationArray[x++] = rgbArray[i+bandLength*2];
            destinationArray[x++] = 255;
        } 
        return destinationArray;
    }

    private async upscale(imageData: ImageData): Promise<ImageData> {
        return await new Promise((resolve, reject) => {
            const request = new XMLHttpRequest();
            request.open("POST", this.baseUrl + "/api/v1/upscale?width=" + imageData.width + "&height=" + imageData.height, true);
            request.setRequestHeader("Content-Type", "application/octet-stream");
            request.responseType = "arraybuffer";
            request.withCredentials = true;
            request.onload = () => {
                const status = request.status;
                if (status == 200) {
                    const upscaledRgbData = new Uint8ClampedArray(request.response);
                    const upscaledRgbaData = this.toRgba(upscaledRgbData);
                    const upscaledImageData = new ImageData(upscaledRgbaData, imageData.width * SuperResolutionModel.SCALING_FACTOR, imageData.height * SuperResolutionModel.SCALING_FACTOR)
                    resolve(upscaledImageData);
                } else {
                    reject(status);
                }
            };
            const rgbData = this.toRgb(imageData.data);
            request.send(rgbData);
        });
    }
}

class LocalSuperResolutionModel extends SuperResolutionModel {

    private model: tf.GraphModel;

    public constructor(model: tf.GraphModel) {
        super()
        this.model = model;
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
        const imageData = this.renderBitmap(imageBitmap);
        return tf.browser.fromPixels(imageData);
    }

    private async copyToImageBitmap(imageTensor: Tensor3D, width: number, height: number): Promise<ImageBitmap> {
        const imageDataArray = await tf.browser.toPixels(imageTensor);
        const imageData = new ImageData(imageDataArray, width, height);
        return await createImageBitmap(imageData);
    }
}