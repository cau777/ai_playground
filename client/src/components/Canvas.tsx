import {FC, useEffect, useRef, useState} from "react";
import {BtnPrimary} from "./BtnPrimary";
import {ClearIcon} from "./icons/ClearIcon";

/**
 * @link https://stackoverflow.com/questions/50393418/what-happens-if-i-dont-test-passive-event-listeners-support
 */
function isPassiveEnabled() {
    let result = false;
    try {
        // Test via a getter in the options object to see if the passive property is accessed
        let opts = Object.defineProperty({}, 'passive', {
            get: function () {
                result = true;
            }
        });
        
        let name: any = "testPassive";
        let listener: any = null;
        
        window.addEventListener(name, listener, opts);
        window.removeEventListener(name, listener, opts);
    } catch (e) {}
    
    return result;
}

const passiveOptions: any = isPassiveEnabled() ? {passive: false} : false;

type Props = {
    registerCanvas: (element: HTMLCanvasElement) => void;
    size: number;
}

export const Canvas: FC<Props> = (props) => {
    let ref = useRef<HTMLCanvasElement>(null);
    let prevPos = useRef<[number, number]>();
    let [drawing, setDrawing] = useState(false);
    
    useEffect(() => {
        const mouseDraw = (e: MouseEvent) => drawMove(e.clientX, e.clientY);
        const touchDraw = (e: TouchEvent) => drawMove(e.touches[0].clientX, e.touches[0].clientY);
        const preventScroll = (e: TouchEvent) => e.preventDefault();
        const cancel = () => setDrawing(false);
        
        if (drawing) {
            window.addEventListener("mousemove", mouseDraw);
            window.addEventListener("touchmove", touchDraw);
            window.addEventListener("touchmove", preventScroll, passiveOptions);
            window.addEventListener("mouseup", cancel);
            window.addEventListener("touchend", cancel);
            return () => {
                window.removeEventListener("mousemove", mouseDraw);
                window.removeEventListener("touchmove", touchDraw);
                window.removeEventListener("touchmove", preventScroll, passiveOptions);
                window.removeEventListener("mouseup", cancel);
                window.removeEventListener("touchend", cancel);
                prevPos.current = undefined;
            }
        }
    }, [drawing]);
    
    useEffect(() => {
        props.registerCanvas(ref.current!);
    }, []);
    
    function clear() {
        ref.current?.getContext("2d")!.clearRect(0, 0, props.size, props.size);
    }
    
    function drawMove(screenX: number, screenY: number) {
        let canvas = ref.current;
        if (canvas === null) return;
        
        let ctx = canvas.getContext("2d")!;
        let rect = canvas.getBoundingClientRect();
        let currX = screenX - rect.left;
        let currY = screenY - rect.top;
        
        if (prevPos.current !== undefined) {
            ctx.beginPath();
            ctx.moveTo(prevPos.current[0], prevPos.current[1]);
            ctx.lineTo(currX, currY);
            ctx.lineWidth = 6;
            ctx.strokeStyle = "black";
            ctx.shadowColor = "black";
            ctx.shadowBlur = 4;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.stroke();
            ctx.closePath();
        }
        
        prevPos.current = [currX, currY];
    }
    
    return (
        <div className={"flex mt-9"}>
            <canvas className={"bg-white border-2 border-back-1"} width={props.size} height={props.size} ref={ref}
                    onMouseDown={() => setDrawing(true)}
                    onTouchStart={() => setDrawing(true)}></canvas>
            <div>
                <BtnPrimary label={"Clear"} onClick={clear}>
                    <ClearIcon className={"w-6"}></ClearIcon>
                </BtnPrimary>
            </div>
        </div>
    )
}