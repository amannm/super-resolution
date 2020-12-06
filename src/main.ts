import {SuperResolutionModel} from "./inference";

async function load() {
    const img = document.getElementById("image") as HTMLImageElement;

    const inputCanvas = document.getElementById("input") as HTMLCanvasElement;
    inputCanvas.width = img.clientWidth;
    inputCanvas.height = img.clientHeight;
    const inputCtx = inputCanvas.getContext('2d');
    inputCtx.drawImage(img, 0, 0, inputCanvas.width, inputCanvas.height);
    inputCtx.save();

    const outputCanvas = document.getElementById("output") as HTMLCanvasElement;
    outputCanvas.width = img.clientWidth * 4;
    outputCanvas.height = img.clientHeight * 4;

    const imageGrid = subdivideImage(inputCtx, inputCanvas.width, inputCanvas.height);

    const outputCtx = outputCanvas.getContext("2d");
    const model = await SuperResolutionModel.open("../model/esrgan/model.json");
    imageGrid.forEach(cell => model.resolve(cell.image).then(resolvedImage => outputCtx.putImageData(resolvedImage, cell.gridX * resolvedImage.width, cell.gridY * resolvedImage.height)));
    model.close();
    outputCtx.save();
    
    //tileImage(outputCtx, upscaledImages, outputCanvas.width);
}

function subdivideImage(ctx: CanvasRenderingContext2D, width: number, height: number): {
    gridX: number,
    gridY: number,
    image: ImageData
}[] {
    const inputs = [];
    const widthParams = subdivideLength(width);
    const heightParams = subdivideLength(height);
    for (let gridY = 0; gridY < heightParams.numSteps; gridY++) {
        for (let gridX = 0; gridX < widthParams.numSteps; gridX++) {
            const imageData: ImageData = ctx.getImageData(gridX * widthParams.stepSize, gridY * heightParams.stepSize, widthParams.stepSize, heightParams.stepSize);
            inputs.push({
                gridX: gridX,
                gridY: gridY,
                image: imageData
            });
        }
    }
    return inputs;
}

function subdivideLength(length: number, maxStepSize: number = 128): { stepSize: number, numSteps: number, extraPixels: number} {
    const steps = Math.floor(length / maxStepSize);
    if (steps === 0) {
        return {
        stepSize: length,
        numSteps: 1,
        extraPixels: 0
        }
    } else {
        if (length % maxStepSize === 0) {
            return {
                stepSize: maxStepSize,
                numSteps: steps,
                extraPixels: 0
            };
        } else {
            const numSteps = steps + 1;
            return {
                stepSize: Math.floor(length / numSteps),
                numSteps: numSteps,
                extraPixels: length % numSteps
            };
        }
    }
}

load().then(() => {
    console.log("done");
});
