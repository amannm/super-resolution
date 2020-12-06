import * as tf from '@tensorflow/tfjs';

async function main() {
    const modelLoadTask = tf.loadGraphModel("../model/srgan/model.json");
    const img = document.getElementById("image") as HTMLImageElement;
    const canvas = document.getElementById("canvas") as HTMLCanvasElement;
    canvas.width = img.clientWidth * 4;
    canvas.height = img.clientHeight * 4;
    const normalized = tf.tidy(() => tf.browser.fromPixels(img).expandDims().toFloat());
    const model = await modelLoadTask;
    const finalImage: tf.Tensor3D = tf.tidy(()=> {
        const result = model.predict(normalized) as tf.Tensor4D;
        model.dispose();
        return result.clipByValue(0, 255).round().toInt().squeeze();
    });
    tf.browser.toPixels(finalImage, canvas);
    finalImage.dispose();
}

main().then(() => {
    console.log("done");
});
