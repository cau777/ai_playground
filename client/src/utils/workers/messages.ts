export type WorkerTasks = "Train"|"Validate"|"Eval";
export type WorkerRequest = {
    type: "process",
    task: WorkerTasks,
    args: unknown[],
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
