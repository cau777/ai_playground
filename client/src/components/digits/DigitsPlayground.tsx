import {BtnPrimary} from "../BtnPrimary";
import {Canvas} from "./Canvas";
import * as server from "../../utils/server-interface";
import {DigitsResultsGraph} from "./DigitsResultsGraph";
import {NavControls} from "../NavControls";
import {Component, createSignal} from "solid-js";
import {useDigitsT} from "~/components/LanguagesContext";
import {PlaygroundContainer} from "~/components/PlaygroundContainer";

const SIZE = 200;
const BORDER = 20;

function getPixel(data: ImageData, row: number, col: number) {
    return data.data[row * data.width * 4 + col * 4 + 3]; // Read the alpha component
}

function checkRow(data: ImageData, row: number) {
    for (let x = 0; x < data.width; x++)
        if (getPixel(data, row, x) > 2) return true;
    return false;
}

function checkCol(data: ImageData, col: number) {
    for (let x = 0; x < data.height; x++)
        if (getPixel(data, x, col) > 2) return true;
    return false;
}

function getRealBounds(data: ImageData): { top: number, right: number, left: number, bottom: number } {
    let top, left, right, bottom;
    for (top = 0; top < data.height; top++)
        if (checkRow(data, top)) break;
    for (bottom = data.height - 1; bottom >= 0; bottom--)
        if (checkRow(data, bottom)) break;
    
    for (left = 0; left < data.width; left++)
        if (checkCol(data, left)) break;
    for (right = data.width - 1; right >= 0; right--)
        if (checkCol(data, right)) break;
    
    return {top, right, bottom, left}
}

export const DigitsPlayground: Component = () => {
    let canvasRef: HTMLCanvasElement | undefined = undefined;
    let resizeCanvasRef: HTMLCanvasElement | undefined = undefined;
    let framedCanvasRef: HTMLCanvasElement | undefined = undefined;
    
    let [result, setResult] = createSignal<number[]>();
    let [busy, setBusy] = createSignal(false);
    let t = useDigitsT();
    
    /**
     * @summary Create a white border around the digit, scaling it down if necessary.
     * This makes the images more similar to the MNIST dataset
     */
    function frame() {
        let centerCanvas = framedCanvasRef;
        let canvas = canvasRef;
        if (centerCanvas === undefined || canvas == undefined) throw new TypeError();
        
        let centerContext = centerCanvas.getContext("2d")!;
        let canvasContext = canvas.getContext("2d")!;
        
        centerContext.clearRect(0, 0, SIZE, SIZE);
        let data = canvasContext.getImageData(0, 0, SIZE, SIZE, {colorSpace: "srgb"});
        let realBounds = getRealBounds(data);
        let width = realBounds.right - realBounds.left;
        let height = realBounds.bottom - realBounds.top;
        
        let usableSize = SIZE - BORDER * 2;
        let scale = usableSize / Math.max(width, height);
        let nWidth = width * scale;
        let nHeight = height * scale;
        
        centerContext.drawImage(canvas, realBounds.left, realBounds.top,
            width, height,
            BORDER + (usableSize - nWidth) / 2, BORDER + (usableSize - nHeight) / 2,
            nWidth, nHeight);
    }
    
    /**
     * @summary Resize the image to MNIST size (28x28)
     */
    function resize() {
        let resizeCanvas = resizeCanvasRef;
        let canvas = framedCanvasRef;
        if (resizeCanvas === undefined || canvas == undefined) throw new TypeError();
        let resizeContext = resizeCanvas.getContext("2d")!;
        
        resizeContext.clearRect(0, 0, 28, 28);
        resizeContext.drawImage(canvas, 0, 0, 28, 28);
    }
    
    function preparePixels() {
        let resizeContext = resizeCanvasRef?.getContext("2d")!;
        let img = resizeContext.getImageData(0, 0, 28, 28, {colorSpace: "srgb"});
        // Get only the transparency channel
        let alpha = img.data.filter((value, index) => index % 4 === 3);
        return Array.from(alpha);
    }
    
    async function evaluate() {
        setBusy(true);
        setResult(undefined);
        
        try {
            frame();
            resize();
            let pixels = preparePixels();
            let result = await server.digits_eval(pixels);
            setResult(result);
        } finally {
            setBusy(false);
        }
    }
    
    return (
        <NavControls>
            <PlaygroundContainer>
                <h1 class={"text-3xl font-black text-primary-100 mb-3"}>{t.title}</h1>
                <p class={""}>{t.instructions}</p>
                <p class={"text-font-2"}>{t.limitations}</p>
                <div>
                    <div class={"mb-2"}>
                        <Canvas registerCanvas={c => canvasRef = c} size={SIZE}></Canvas>
                    </div>
                    <canvas ref={framedCanvasRef} class={"bg-white hidden"} width={SIZE} height={SIZE}></canvas>
                    <canvas ref={resizeCanvasRef} class={"bg-white hidden"} width={28} height={28}></canvas>
                    
                    <BtnPrimary disabled={busy()} label={t.evaluateBtn} onClick={evaluate}></BtnPrimary>
                </div>
                
                <DigitsResultsGraph probabilities={result()}></DigitsResultsGraph>
                
                <h2 class={"text-xl font-semibold mt-4"}>{t.examples}</h2>
                <img alt={"MNIST examples"} class={"mt-2 max-h-64"}
                     src={"https://www.researchgate.net/profile/Steven-Young-5/publication/306056875/figure/fig1/AS:393921575309346@1470929630835/Example-images-from-the-MNIST-dataset.png"}/>
            </PlaygroundContainer>
        </NavControls>
    )
}