import * as methods from "wasm";
import init from "wasm";
import {provideData, submitTest} from "./server_interface";
import {TrainSocket} from "./TrainSocket";
import * as serverMethods from "./server_interface";
import {insertLog} from "./logging";

// Workaround to access js functions from rust
// @ts-ignore
window.bindings = {
    insertLog
}

enum Mode {
    None, Train, Test
}

export class WasmInterface {
    public static async create() {
        await init();
        let config = await provideData("local|digits/model.xml"); // TODO: release
        return new WasmInterface(config);
    }

    private trainSocket: TrainSocket;
    private mode = Mode.None;
    private count = 0;
    private activeJobs = 0;

    private constructor(private config: Uint8Array) {
        this.trainSocket = new TrainSocket();
    }

    public async processTest(url: string, modelUrl: string, version: number, batch: number) {
        await this.switchTestMode(modelUrl);
        this.printInfo();
        this.activeJobs++;
        let data = await provideData(url);
        let result: number = methods.test(data);
        await submitTest(version, batch, result);
        this.activeJobs--;
    }

    public async processTrain(url: string) {
        await this.switchTrainMode();
        this.printInfo();
        this.activeJobs++;
        await this.trainSocket.assertConnected();
        let data = await provideData(url);
        methods.train(data);
        await this.trainSocket.pushIfNecessary();
        this.activeJobs--;
    }

    private async clearLastMode() {
        // Wait for any active job to finish
        await new Promise<void>(res => {
            let interval = setInterval(() => {
                if (this.activeJobs === 0) {
                    clearInterval(interval);
                    res();
                }
            }, 50);
        });

        // Clear unused resources
        switch (this.mode) {
            case Mode.Train: {
                await this.trainSocket.close();
                break
            }
            default:
                break
        }
    }

    private async switchTestMode(modelUrl: string) {
        if (this.mode === Mode.Test) return;
        await this.clearLastMode();
        this.mode = Mode.Test;

        let initialData = await provideData(modelUrl);
        methods.load_initial(initialData, this.config);
    }

    private async switchTrainMode() {
        if (this.mode === Mode.Train) return;
        await this.clearLastMode();
        this.mode = Mode.Train;
        let initialData = await serverMethods.getMostRecent();
        methods.load_initial(initialData, this.config);
    }

    private printInfo() {
        console.log("Started task", this.count++, "of type", Mode[this.mode]);
    }
}