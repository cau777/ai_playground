import {Task} from "../server_interface";

export type WorkerRequest = {
    type: "process",
    task: Task["type"],
    arg: Uint8Array,
} | {
    type: "init"
}

export type WorkerResponse = {
    type: "response",
    data: unknown,
} | {
    type: "log",
    message: string,
    level: number
}
