import {FC, useRef, useState} from "react";
import {Slider} from "./Slider";
import {BtnPrimary} from "./BtnPrimary";
import {assign} from "../utils/server_interface";
import {WasmInterface} from "../utils/WasmInterface";

let wasmInterface = await WasmInterface.create();

export const TrainMenu: FC = () => {
    let cores = Math.max(1, navigator.hardwareConcurrency-1);
    let workers = useRef(1);
    let canRepeatWork = useRef(false);
    let promises = useRef<Promise<void>[]>([]);
    let [busy, setBusy] = useState(false);
    
    async function worker() {
        while (canRepeatWork.current) {
            let taskResponse = await assign();
            let task = taskResponse.data;
    
            switch (task.type) {
                case "Test":
                    await wasmInterface.processTest(task.url, task.model_url, task.version, task.batch)
                    break
                case "Train":
                    await wasmInterface.processTrain(task.url);
                    break
            }
        }
    }
    
    function run() {
        console.log("run");
        setBusy(true);
        canRepeatWork.current = true;
        for (let i = 0; i < workers.current; i++) {
            promises.current.push(worker());
        }
    }
    
    async function stop() {
        console.log("stop");
        canRepeatWork.current = false;
        await Promise.all(promises.current);
        setBusy(false);
    }
    
    return (
        <div className={"max-w-3xl p-4"}>
            Training!!!
            <Slider label={"Workers"} disabled={busy} min={1} max={cores} onChange={o => workers.current = o}></Slider>
            <div>
                <BtnPrimary label={"Start"} disabled={busy} onClick={run}></BtnPrimary>
                <BtnPrimary label={"Stop"} disabled={!busy} onClick={stop}></BtnPrimary>
            </div>
        </div>
    )
}