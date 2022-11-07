import {WorkerRequest, WorkerResponse, WorkerTasks} from "./messages";
import init, {eval_job, train_job, validate_job} from "wasm";

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

function findWasmFunc(task: WorkerTasks): (...args: any) => any {
    switch (task) {
        case "Eval":
            return eval_job;
        case "Train":
            return train_job;
        case "Validate":
            return validate_job;
    }
}

onmessage = async function (message) {
    let request = message.data as WorkerRequest;
    switch (request.type) {
        case "init":
            await init();
            respond({type: "response", data: null});
            break
        case "process":
            let result = findWasmFunc(request.task)(...request.args);
            respond({type: "response", data: result});
            break
    }
}