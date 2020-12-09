import { SuperResolutionModel } from "./inference"

interface ImagePatch {
    topLeftX: number,
    topLeftY: number,
    originalImage: ImageBitmap,
    enhancedImage?: ImageBitmap
}

export class ImageEnhancer {

    public static readonly PATCH_SIZE = 128;

    private canvas: HTMLCanvasElement;
    private patches: ImagePatch[];
    private enhancedLayerOpacity: number;

    public constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.patches = [];
        this.enhancedLayerOpacity = 1.0;
    }

    public async load(image: ImageBitmap): Promise<void> {

        // clear existing state
        while (this.patches.length > 0) {
            const patch = this.patches.pop();
            patch.originalImage.close();
            if (patch.enhancedImage) {
                patch.enhancedImage.close();
            }
        }

        // calculate new patch grid
        const widthParams = ImageEnhancer.splitLength(image.width);
        const heightParams = ImageEnhancer.splitLength(image.height);

        // resize canvas
        this.canvas.width = (widthParams.stepCount * ImageEnhancer.PATCH_SIZE - widthParams.remainder) * SuperResolutionModel.SCALING_FACTOR;
        this.canvas.height = (heightParams.stepCount * ImageEnhancer.PATCH_SIZE - heightParams.remainder) * SuperResolutionModel.SCALING_FACTOR;
        this.canvas.style.width = (this.canvas.width / SuperResolutionModel.SCALING_FACTOR) + "px";
        this.canvas.style.height = (this.canvas.height / SuperResolutionModel.SCALING_FACTOR) + "px";

        // extract and draw image patches
        const newPatchTasks = [];
        for (let gridY = 0; gridY < heightParams.stepCount; gridY++) {
            for (let gridX = 0; gridX < widthParams.stepCount; gridX++) {
                const widthOverhang = gridX === widthParams.stepCount - 1 ? widthParams.remainder : 0;
                const heightOverhang = gridY === heightParams.stepCount - 1 ? heightParams.remainder : 0;
                const cellTopLeftX = gridX * ImageEnhancer.PATCH_SIZE - widthOverhang;
                const cellTopLeftY = gridY * ImageEnhancer.PATCH_SIZE - heightOverhang;
                const newPatchTask = createImageBitmap(image, cellTopLeftX, cellTopLeftY, ImageEnhancer.PATCH_SIZE, ImageEnhancer.PATCH_SIZE).then(imageData => {
                    const patch = {
                        topLeftX: cellTopLeftX,
                        topLeftY: cellTopLeftY,
                        originalImage: imageData
                    };
                    this.drawOriginalPatch(patch);
                    return patch;
                });
                newPatchTasks.push(newPatchTask);
            }
        }
        this.patches = await Promise.all(newPatchTasks);

        console.log("new image loaded");
        
    }

    public async enhance(model: SuperResolutionModel): Promise<void> {
        const t0 = performance.now();

        for (let i = 0; i < this.patches.length; i++) {
            const patch = this.patches[i];
            patch.enhancedImage = await model.resolve(patch.originalImage);
            this.drawEnhancedPatch(patch);
        }

        const t1 = performance.now();
        console.log(`enhance completed in ${t1 - t0} milliseconds.`);
    }

    private drawOriginalPatch(patch: ImagePatch): void {
        const ctx = this.canvas.getContext("2d");
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(
            patch.originalImage,
            patch.topLeftX * SuperResolutionModel.SCALING_FACTOR,
            patch.topLeftY * SuperResolutionModel.SCALING_FACTOR,
            ImageEnhancer.PATCH_SIZE * SuperResolutionModel.SCALING_FACTOR, 
            ImageEnhancer.PATCH_SIZE * SuperResolutionModel.SCALING_FACTOR
        );
        ctx.restore();
    }

    private drawEnhancedPatch(patch: ImagePatch): void {
        const ctx = this.canvas.getContext("2d");
        ctx.save();
        ctx.globalAlpha = this.enhancedLayerOpacity;
        ctx.drawImage(
            patch.enhancedImage,
            patch.topLeftX * SuperResolutionModel.SCALING_FACTOR,
            patch.topLeftY * SuperResolutionModel.SCALING_FACTOR
        );
        ctx.restore();
    }

    private drawPatch(patch: ImagePatch): void {
        this.drawOriginalPatch(patch);
        if (patch.enhancedImage) {
            this.drawEnhancedPatch(patch);
        }
    }

    private draw(): void {
        const ctx = this.canvas.getContext("2d");
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.patches.forEach(patch => this.drawPatch(patch));
    }

    public toggleEnhancedVisibility(): void {
        if (this.enhancedLayerOpacity !== 0) {
            this.enhancedLayerOpacity = 0;
            this.draw();
            console.log("enhanced layer off");
        } else {
            this.enhancedLayerOpacity = 1.0;
            this.draw();
            console.log("enhanced layer on");
        }
    }

    private static splitLength(length: number): { stepCount: number, remainder: number } {
        let stepCount = Math.floor(length / ImageEnhancer.PATCH_SIZE);
        if (stepCount === 0) {
            return {
                stepCount: 1,
                remainder: ImageEnhancer.PATCH_SIZE - length
            };
        } else {
            if (length % ImageEnhancer.PATCH_SIZE === 0) {
                return {
                    stepCount: stepCount,
                    remainder: 0
                };
            } else {
                return {
                    stepCount: stepCount + 1,
                    remainder: ImageEnhancer.PATCH_SIZE - length % ImageEnhancer.PATCH_SIZE
                };
            }
        }
    }
}