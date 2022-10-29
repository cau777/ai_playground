import {WorkerRequest, WorkerResponse} from "./messages";
import {Task} from "../server_interface";
import {insertLog} from "../logging";

type Job = { task: Task["type"], arg: Uint8Array, callback: (data: unknown) => void };

export class WorkersCoordinator {
    private queue: Job[] = [];
    private readonly workers: Worker[];
    private readonly availableWorkers: Set<Worker>;
    
    public constructor(workerCount: number) {
        this.workers = [];
        this.availableWorkers = new Set<Worker>();
        
        for (let i = 0; i < workerCount; i++) {
            let worker = new Worker(new URL("./worker.ts", import.meta.url), {
                type: "module"
            });
            this.workers.push(worker);
            this.availableWorkers.add(worker);
        }
        
        for (let worker of this.workers) {
            this.assign({type: "init"}, worker, () => {
            });
        }
    }
    
    public get queueSize() {
        return this.queue.length;
    }
    
    public get activeJobs() {
        return this.workers.length - this.availableWorkers.size;
    }
    
    public enqueueTrain(arg: Uint8Array, callback: (data: Uint8Array) => void) {
        this.enqueue({
            task: "Train", arg, callback: (value) => {
                if (value instanceof Uint8Array) return callback(value);
                throw new TypeError("Train function didn't return Uint8Array");
            }
        });
    }
    
    public enqueueValidate(arg: Uint8Array, callback: (data: number) => void) {
        this.enqueue({
            task: "Validate", arg, callback: (value) => {
                if (typeof value === "number") return callback(value);
                throw new TypeError("Validate function didn't return number " + JSON.stringify(value));
            }
        });
    }
    
    private enqueue(job: Job) {
        // noinspection LoopStatementThatDoesntLoopJS
        for (const worker of this.availableWorkers) {
            this.assignJob(job, worker);
            return;
        }
        
        this.queue.push(job);
    }
    
    private assign(request: WorkerRequest, worker: Worker, callback: (value: unknown) => void) {
        this.availableWorkers.delete(worker);
        let start = performance.now();
        let promise = new Promise<unknown>((res, rej) => {
            let messageListener = (value: MessageEvent<unknown>) => {
                let response = value.data as WorkerResponse;
                if (response.type === "log") {
                    insertLog(response.message, response.level);
                } else {
                    worker.removeEventListener("message", messageListener);
                    res(response.data);
                }
            };
            worker.addEventListener("message", messageListener);
            
            let errorListener = (e: any) => {
                worker.removeEventListener("error", errorListener);
                console.error(e);
                insertLog("Error: " + e, 2);
                rej(e);
            }
            worker.addEventListener("error", rej);
        });
        promise.then(o => {
            let duration = performance.now() - start;
            insertLog(`Finished ${request.type} task in ${duration}ms`, 0);
            
            this.finishJob(worker);
            callback(o);
        });
        
        worker.postMessage(request);
    }
    
    private finishJob(worker: Worker) {
        this.availableWorkers.add(worker);
        if (this.queue.length !== 0) {
            this.assignJob(this.queue[0], worker);
            this.queue.splice(0, 1);
        }
    }
    
    private assignJob(job: Job, worker: Worker) {
        this.assign({type: "process", task: job.task, arg: job.arg}, worker, job.callback);
    }
}