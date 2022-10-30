import {FC, useRef, useState} from "react";
import {BtnPrimary} from "./BtnPrimary";
import {Canvas} from "./Canvas";
import {WorkersCoordinator} from "../utils/workers/WorkersCoordinator";
import {WasmMain} from "../utils/WasmMainInterface";

export const DigitsPlayground: FC = () => {
    let canvasRef = useRef<HTMLCanvasElement>();
    let resizeCanvasRef = useRef<HTMLCanvasElement>(null);
    let [result, setResult] = useState("");
    let [busy, setBusy] = useState(false);
    
    function preparePixels() {
        // Use another canvas to resize image to MNIST size (28x28)
        let resizeCanvas = resizeCanvasRef.current;
        let canvas = canvasRef.current;
        if (resizeCanvas === null || canvas == null) throw new TypeError();
        let resizeContext = resizeCanvas.getContext("2d")!;
        
        resizeContext.clearRect(0, 0, 28, 28);
        resizeContext.drawImage(canvas, 0, 0, 28, 28);
        
        let img = resizeContext.getImageData(0, 0, 28, 28, {colorSpace: "srgb"});
        // Get only the transparency channel
        let pixels = img.data.filter((value, index) => index % 4 === 3);
        return Float32Array.from(pixels);
    }
    
    async function evaluate() {
        setBusy(true);
        let pixels = preparePixels();
        let main = await WasmMain;
        let arg = await main.prepareEval(pixels);
        
        let coord = new WorkersCoordinator(1);
        coord.enqueueEval(arg, value => {
            setResult(JSON.stringify(value))
            setBusy(false);
        });
    }
    
    return (
        <div>
            Digits!!!
            <div className={"mb-2"}>
                <Canvas registerCanvas={c => canvasRef.current = c}></Canvas>
            </div>
            <canvas ref={resizeCanvasRef} className={"bg-white"} width={28} height={28}></canvas>
            
            <p>
                Result = {result}
            </p>
            <BtnPrimary disabled={busy} label={"Evaluate"} onClick={evaluate}></BtnPrimary>
        </div>
    )
}