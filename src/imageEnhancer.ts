import { SuperResolutionModel } from "./inference"

interface ImagePatch {
    topLeftX: number,
    topLeftY: number,
    originalImage: ImageData,
    enhancedImage: ImageData
}

export class ImageEnhancer {

    public static readonly MAX_PATCH_SIZE = 128;

    private canvas: HTMLCanvasElement;
    private patches: ImagePatch[];
    private enhancedLayerOpacity: number;

    public constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.patches = [];
        this.enhancedLayerOpacity = 1.0;
    }

    public async load(image: ImageBitmap): Promise<void> {
        // render image on pixel buffer
        const widthParams = ImageEnhancer.splitLength(image.width);
        const heightParams = ImageEnhancer.splitLength(image.height);
        this.canvas.width = image.width - widthParams.remainder;
        this.canvas.height = image.height - heightParams.remainder;
        const ctx = this.canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, this.canvas.width, this.canvas.height, 0, 0, this.canvas.width, this.canvas.height);

        // extract image patches from rendered image
        this.patches = [];
        for (let gridY = 0; gridY < heightParams.stepCount; gridY++) {
            for (let gridX = 0; gridX < widthParams.stepCount; gridX++) {
                const cellTopLeftX = gridX * widthParams.stepSize;
                const cellTopLeftY = gridY * heightParams.stepSize
                const imageData = ctx.getImageData(cellTopLeftX, cellTopLeftY, widthParams.stepSize, heightParams.stepSize);
                this.patches.push({
                    topLeftX: cellTopLeftX,
                    topLeftY: cellTopLeftY,
                    originalImage: imageData,
                    enhancedImage: null
                });
            }
        }

        // prepare for enhancement
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.canvas.width *= SuperResolutionModel.SCALING_FACTOR;
        this.canvas.height *= SuperResolutionModel.SCALING_FACTOR;
        await Promise.all(this.patches.map(patch => this.drawOriginalPatch(patch)));
        this.canvas.style.width = (this.canvas.width / SuperResolutionModel.SCALING_FACTOR) + "px";
        this.canvas.style.height = (this.canvas.height / SuperResolutionModel.SCALING_FACTOR) + "px";
    }

    public async toggleEnhancedVisibility(): Promise<void> {
        if (this.enhancedLayerOpacity !== 0) {
            this.enhancedLayerOpacity = 0;
        } else {
            this.enhancedLayerOpacity = 1.0;
        }
        await this.draw();
        console.log("enhanced layer visibility: " + this.enhancedLayerOpacity);
    }

    public async enhance(model: SuperResolutionModel): Promise<void> {
        // prepare pixel buffer
        this.canvas.removeAttribute("style");

        // incremental patch enhancement to avoid blowing up the GPU
        for (let i = 0; i < this.patches.length; i++) {
            const patch = this.patches[i];
            await model.resolve(patch.originalImage).then(enhancedImage => {
                patch.enhancedImage = enhancedImage;
                return this.drawEnhancedPatch(patch);
            });
        }

        // present results
        this.canvas.style.width = (this.canvas.width / SuperResolutionModel.SCALING_FACTOR) + "px";
        this.canvas.style.height = (this.canvas.height / SuperResolutionModel.SCALING_FACTOR) + "px";
        console.log("enhanced layer complete");
    }

    private async drawOriginalPatch(patch: ImagePatch): Promise<void> {
        const originalImage = await createImageBitmap(patch.originalImage, {
            resizeWidth: patch.originalImage.width * SuperResolutionModel.SCALING_FACTOR,
            resizeHeight: patch.originalImage.height * SuperResolutionModel.SCALING_FACTOR,
            resizeQuality: "high"
        });
        const ctx = this.canvas.getContext("2d");
        ctx.drawImage(originalImage, patch.topLeftX * SuperResolutionModel.SCALING_FACTOR, patch.topLeftY * SuperResolutionModel.SCALING_FACTOR);
        originalImage.close();
    }

    private async drawEnhancedPatch(patch: ImagePatch): Promise<void> {
        const enhancedImage = await createImageBitmap(patch.enhancedImage);
        const ctx = this.canvas.getContext("2d");
        ctx.save();
        ctx.globalAlpha = this.enhancedLayerOpacity;
        ctx.drawImage(enhancedImage, patch.topLeftX * SuperResolutionModel.SCALING_FACTOR, patch.topLeftY * SuperResolutionModel.SCALING_FACTOR);
        ctx.restore();
        enhancedImage.close();
    }

    private async drawPatch(patch: ImagePatch): Promise<void> {
        await this.drawOriginalPatch(patch);
        if (patch.enhancedImage !== null) {
            await this.drawEnhancedPatch(patch);
        }
    }

    private async draw(): Promise<void> {
        this.canvas.removeAttribute("style");
        const ctx = this.canvas.getContext("2d");
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        await Promise.all(this.patches.map(patch => this.drawPatch(patch)));
        this.canvas.style.width = (this.canvas.width / SuperResolutionModel.SCALING_FACTOR) + "px";
        this.canvas.style.height = (this.canvas.height / SuperResolutionModel.SCALING_FACTOR) + "px";
    }

    private static splitLength(length: number): { stepSize: number, stepCount: number, remainder: number } {
        let stepCount = Math.floor(length / ImageEnhancer.MAX_PATCH_SIZE);
        if (stepCount === 0) {
            return {
                stepSize: length,
                stepCount: 1,
                remainder: 0
            }
        } else {
            if (length % ImageEnhancer.MAX_PATCH_SIZE === 0) {
                return {
                    stepSize: ImageEnhancer.MAX_PATCH_SIZE,
                    stepCount: stepCount,
                    remainder: 0
                };
            } else {
                stepCount++;
                return {
                    stepSize: Math.floor(length / stepCount),
                    stepCount: stepCount,
                    remainder: length % stepCount
                };
            }
        }
    }
}