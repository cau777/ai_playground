import {WorkerRequest, WorkerResponse} from "./messages";
import init, {eval_job, train_job, validate_job} from "wasm";
// importScripts("/pkg/wasm");

// console.log(wasm);
// const init, {train_job, validate_job} = wasm;

// @ts-ignore
self.bindings = {
    insertLog(message: string, level: number) {
        console.log(level, message);
        respond({type: "log", level, message});
    }
}


function respond(response: WorkerResponse) {
    postMessage(response)
}

onmessage = async function (message) {
    let request = message.data as WorkerRequest;
    switch (request.type) {
        case "init":
            await init();
            respond({type: "response", data: null});
            break
        case "process":
            let result;
            
            switch (request.task) {
                case "Validate":
                    result = validate_job(request.arg);
                    break
                case "Train":
                    result = train_job(request.arg);
                    break
                case "Eval":
                    result = eval_job(request.arg);
                    break
            }
            
            respond({type: "response", data: result});
            break
    }
}