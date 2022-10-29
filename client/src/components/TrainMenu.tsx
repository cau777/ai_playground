import {FC, useRef, useState} from "react";
import {Slider} from "./Slider";
import {BtnPrimary} from "./BtnPrimary";
import {assign, submitTest} from "../utils/server_interface";
import {LogsView} from "./LogsView";
import {TrainSocket} from "../utils/TrainSocket";
import {WorkersCoordinator} from "../utils/workers/WorkersCoordinator";
import {WasmMainInterface} from "../utils/WasmMainInterface";
import {sleep} from "../utils/promises";

export const TrainMenu: FC = () => {
    let cores = Math.max(1, navigator.hardwareConcurrency - 1);
    let workers = useRef(1);
    let canRepeatWork = useRef(false);
    let [busy, setBusy] = useState(false);
    let running = useRef(0);
    let socket = useRef<TrainSocket>();
    
    async function createTasks(coord: WorkersCoordinator, main: WasmMainInterface) {
        while (canRepeatWork.current && coord.queueSize < 4) {
            running.current++;
            
            let response = await assign();
            switch (response.type) {
                case "Train": {
                    let arg = await main.prepareTrain(response.url);
                    coord.enqueueTrain(arg, value => {
                        main.loadDeltas(value)
                            .then(() => createTasks(coord, main))
                            .then(() => running.current--);
                    });
                    break
                }
                case "Validate": {
                    let {version, batch, url, model_url} = response;
                    let arg = await main.prepareValidate(url, model_url);
                    coord.enqueueValidate(arg, value => {
                        submitTest(version, batch, value)
                            .then(() => createTasks(coord, main))
                            .then(() => running.current--);
                    })
                    break
                }
            }
            
            await sleep(80);
        }
    }
    
    async function run() {
        console.log("run");
        setBusy(true);
        
        canRepeatWork.current = true;
        socket.current = new TrainSocket();
        let coord = new WorkersCoordinator(workers.current);
        let main = await WasmMainInterface.create(socket.current);
        await createTasks(coord, main);
    }
    
    async function stop() {
        console.log("stop");
        canRepeatWork.current = false;
        await new Promise<void>(res => {
            let interval  = setInterval(() => {
                if (running.current == 0) {
                    clearInterval(interval);
                    res();
                }
            }, 50);
        });
        socket.current?.close();
        setBusy(false);
    }
    
    return (
        <div className={"max-w-3xl p-4"}>
            Training!!!
            <Slider label={"Workers"} disabled={busy} min={1} max={cores} onChange={o => workers.current = o}></Slider>
            <LogsView></LogsView>
            <div>
                <BtnPrimary label={"Start"} disabled={busy} onClick={run}></BtnPrimary>
                <BtnPrimary label={"Stop"} disabled={!busy} onClick={stop}></BtnPrimary>
            </div>
        </div>
    )
}