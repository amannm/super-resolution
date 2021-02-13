import { SuperResolutionModel } from "./inference";
import { ImageEnhancer } from "./imageEnhancer"

const inputCanvas = document.getElementById("canvas") as HTMLCanvasElement;
const enhancer = new ImageEnhancer(inputCanvas);

async function load() {

    // key actions
    window.addEventListener("keydown", event => {
        switch (event.key) {
            case "e":
                enhancer.toggleEnhancedVisibility();
                break;
            default:
                break
        }
    });

    // file drag and drop trigger
    const root = document.getElementById("root");
    root.addEventListener("drop", (e) => {
        console.log('File(s) dropped');
        e.preventDefault();
        if (e.dataTransfer.items) {
            for (let i = 0; i < e.dataTransfer.items.length; i++) {
                if (e.dataTransfer.items[i].kind === 'file') {
                    const file = e.dataTransfer.items[i].getAsFile();
                    processImageFile(file);
                }
            }
        } else {
            for (let i = 0; i < e.dataTransfer.files.length; i++) {
                const file = e.dataTransfer.files[i];
                processImageFile(file);
            }
        }
    });
    root.addEventListener("dragover", (e) => {
        console.log('File(s) in drop zone');
        e.preventDefault();
    });

    // hacky convenience things
    (window as any).lol = {
        enhancer: enhancer
    }
}

load();

const processImageFile = (file: File) => {
    const loadModelTask = SuperResolutionModel.openLocal("../model/esrgan/model.json");
    //const loadModelTask = SuperResolutionModel.openRemote("http://amann-linux:8080");
    const fileReader = new FileReader();
    fileReader.onload = async () => {
        const image = await createImageBitmap(file);
        await enhancer.load(image);
        image.close();
        const model = await loadModelTask;
        await enhancer.enhance(model);
        model.close();
        console.log("image processing complete");
    };
    fileReader.readAsDataURL(file);
}