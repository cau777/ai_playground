import {FC, useRef, useState} from "react";
import {BtnPrimary} from "./BtnPrimary";
import {Canvas} from "./Canvas";
import * as server from "../utils/server-interface";

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

export const DigitsPlayground: FC = () => {
    let canvasRef = useRef<HTMLCanvasElement>();
    let resizeCanvasRef = useRef<HTMLCanvasElement>(null);
    let framedCanvasRef = useRef<HTMLCanvasElement>(null);
    let [result, setResult] = useState("");
    let [busy, setBusy] = useState(false);
    
    /**
     * @summary Create a white border around the digit, scaling it down if necessary.
     * This makes the images more similar to the MNIST dataset
     */
    function frame() {
        let centerCanvas = framedCanvasRef.current;
        let canvas = canvasRef.current;
        if (centerCanvas === null || canvas == null) throw new TypeError();
        
        let centerContext = centerCanvas.getContext("2d")!;
        let canvasContext = canvas.getContext("2d")!;
        
        centerContext.clearRect(0, 0, SIZE, SIZE);
        let data = canvasContext.getImageData(0, 0, SIZE, SIZE, {colorSpace: "srgb"});
        let realBounds = getRealBounds(data);
        let clampedBounds = {
            left: Math.min(realBounds.left, BORDER),
            top: Math.min(realBounds.top, BORDER),
            right: Math.max(realBounds.right, SIZE - BORDER),
            bottom: Math.max(realBounds.bottom, SIZE - BORDER)
        };
        
        centerContext.drawImage(canvas, clampedBounds.left, clampedBounds.top,
            clampedBounds.right - clampedBounds.left, clampedBounds.bottom - clampedBounds.top,
            BORDER, BORDER, SIZE - BORDER * 2, SIZE - BORDER * 2);
    }
    
    /**
     * @summary Resize the image to MNIST size (28x28)
     */
    function resize() {
        let resizeCanvas = resizeCanvasRef.current;
        let canvas = framedCanvasRef.current;
        if (resizeCanvas === null || canvas == null) throw new TypeError();
        let resizeContext = resizeCanvas.getContext("2d")!;
        
        resizeContext.clearRect(0, 0, 28, 28);
        resizeContext.drawImage(canvas, 0, 0, 28, 28);
    }
    
    function preparePixels() {
        let resizeContext = resizeCanvasRef.current?.getContext("2d")!;
        let img = resizeContext.getImageData(0, 0, 28, 28, {colorSpace: "srgb"});
        // Get only the transparency channel
        let alpha = img.data.filter((value, index) => index % 4 === 3);
        return Array.from(alpha);
    }
    
    async function evaluate() {
        setBusy(true);
        frame();
        resize();
        let pixels = preparePixels();
        console.log(await server.evaluate(pixels));
        setBusy(false);
    }
    
    return (
        <div className={"grid-center"}>
            <div>
                <div className={"mb-2"}>
                    <Canvas registerCanvas={c => canvasRef.current = c} size={SIZE}></Canvas>
                </div>
                <canvas ref={framedCanvasRef} className={"bg-white hidden"} width={SIZE} height={SIZE}></canvas>
                <canvas ref={resizeCanvasRef} className={"bg-white hidden"} width={28} height={28}></canvas>
                
                <BtnPrimary disabled={busy} label={"Evaluate"} onClick={evaluate}></BtnPrimary>
            </div>
        </div>
    )
}