import {FC, useRef, useState} from "react";
import {BtnPrimary} from "./BtnPrimary";
import {Canvas} from "./Canvas";
import {prepare_digits} from "wasm";
import {WasmInterface} from "../utils/WasmInterface";

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
        let main = await WasmInterface.create(1);
        
        let inputs = prepare_digits(pixels); // TODO: encapsulate prepare_digits
        let value = await main.processEval(inputs);
        setResult(JSON.stringify(value))
        setBusy(false);
        main.close();
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