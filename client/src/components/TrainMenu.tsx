import {FC, useRef, useState} from "react";
import {Slider} from "./Slider";
import {BtnPrimary} from "./BtnPrimary";
import {assign, submitTest} from "../utils/server_interface";
import {LogsView} from "./LogsView";
import {WasmInterface} from "../utils/WasmInterface";
import {sleep} from "../utils/promises";

export const TrainMenu: FC = () => {
    let cores = Math.max(1, navigator.hardwareConcurrency - 1);
    let workers = useRef(1);
    let canRepeatWork = useRef(false);
    let [busy, setBusy] = useState(false);
    let wasmRef = useRef<WasmInterface>();
    
    async function createTasks(main: WasmInterface) {
        while (canRepeatWork.current) {
            if (main.coord.queueSize >= 4) {
                await sleep(80);
                continue;
            }

            let [_, response] = await Promise.all([sleep(80), assign()]);
            if (!canRepeatWork.current) return;
            switch (response.type) {
                case "Train": {
                    main.processTrain(response.url, workers.current).then();
                    break;
                }
                case "Validate": {
                    let {version, batch, url, model_url} = response;
                    main.processValidate(url, model_url)
                        .then(o => submitTest(version, batch, o))
                    break;
                }
            }
        }
    }

    async function run() {
        console.log("run");
        setBusy(true);
        canRepeatWork.current = true;
        let wasm = await WasmInterface.create(workers.current);
        wasmRef.current = wasm;
        await createTasks(wasm);
    }

    async function stop() {
        console.log("stop");
        canRepeatWork.current = false;
        if (wasmRef.current === undefined) return;
        let main = wasmRef.current;

        await new Promise<void>(res => {
            let interval = setInterval(() => {
                if (main.coord.activeJobs === 0) {
                    clearInterval(interval);
                    res();
                }
            }, 20);
        });

        main.close();
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
                <br/>
                <BtnPrimary label={"Manual open"} disabled={!busy} onClick={async () => {
                    wasmRef.current = await WasmInterface.create(workers.current);
                }}></BtnPrimary>

                {/*
                <BtnPrimary label={"Manual assign"} disabled={!busy} onClick={async () => {
                    let response = await assign();
                    switch (response.type) {
                        case "Train": {
                            main.processTrain(response.url).then();
                            break;
                        }
                        case "Validate": {
                            let {version, batch, url, model_url} = response;
                            main.processValidate(url, model_url)
                                .then(o => submitTest(version, batch, o))
                            break;
                        }
                    }
                }}></BtnPrimary>

                <BtnPrimary label={"Manual sync"} disabled={!busy} onClick={() => {
                    let deltas = mainThreadWasm.export_current();
                    serverMethods.submitTrain(deltas)
                        .then(() => updating = false);
                }}></BtnPrimary>
            */}
            </div>
        </div>
    )
}