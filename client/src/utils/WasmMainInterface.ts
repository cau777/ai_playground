import * as methods from "wasm";
import init from "wasm";
import {provideData} from "./server_interface";
import {TrainSocket} from "./TrainSocket";
import * as serverMethods from "./server_interface";
import {insertLog} from "./logging";

// Workaround to access js functions from rust
// @ts-ignore
window.bindings = {
    insertLog
}

export class WasmMainInterface {
    public static async create(trainSocket: TrainSocket) {
        let config = await provideData("local|digits/model.xml"); // TODO: release
        await init();
        methods.load_config(config);
        return new WasmMainInterface(trainSocket);
    }
    
    private loadingTrain?: Promise<void>;
    
    public constructor(private trainSocket: TrainSocket) {
    }
    
    public async prepareTrain(url: string) {
        await (this.loadingTrain ??= this.loadTrain());
        await this.trainSocket.assertConnected();
        let pairs = await provideData(url);
        
        return methods.prepare_train_job(pairs);
    }
    
    public async prepareValidate(url: string, model_url: string) {
        let [pairs, storage] = await Promise.all([provideData(url), provideData(model_url)]);
        return methods.prepare_validate_job(pairs, storage);
    }
    
    public async prepareEval(data: Float32Array) {
        let bestUrl = await serverMethods.getBest();
        let storage = await provideData(bestUrl);
        return methods.prepare_eval_job(data, storage);
    }
    
    public async loadDeltas(deltas: Uint8Array) {
        methods.load_train_deltas(deltas);
        await this.trainSocket.pushIfNecessary();
        console.log("loaded deltas")
    }
    
    private async loadTrain() {
        let storage = await serverMethods.getMostRecent();
        methods.load_initial(storage);
    }
}

export const WasmMain = WasmMainInterface.create(new TrainSocket());

// export class WasmInterface {
//     public static async create() {
//         await init();
//         let config = await provideData("local|digits/model.xml"); // TODO: release
//         return new WasmInterface(config);
//     }
//
//     private trainSocket: TrainSocket;
//     private mode = Mode.None;
//     private count = 0;
//     private activeJobs = 0;
//     private switchingMode = false;
//
//     public constructor(private config: Uint8Array) {
//         this.trainSocket = new TrainSocket();
//     }
//
//     public async processTest(url: string, modelUrl: string, version: number, batch: number) {
//         await this.switchTestMode(modelUrl);
//         this.printInfo();
//         this.activeJobs++;
//         let data = await provideData(url);
//         let result: number = methods.test(data);
//         await submitTest(version, batch, result);
//         this.activeJobs--;
//     }
//
//     public async processTrain(url: string) {
//         await this.switchTrainMode();
//         this.printInfo();
//         this.activeJobs++;
//         await this.trainSocket.assertConnected();
//         let data = await provideData(url);
//         methods.train(data);
//         await this.trainSocket.pushIfNecessary();
//         this.activeJobs--;
//     }
//
//     private async clearLastMode() {
//         // Wait for any active job to finish
//         await new Promise<void>(res => {
//             let interval = setInterval(() => {
//                 if (this.activeJobs === 0 || !this.switchingMode) {
//                     clearInterval(interval);
//                     res();
//                 }
//             }, 50);
//         });
//         this.switchingMode = true;
//
//         // Clear unused resources
//         switch (this.mode) {
//             case Mode.Train: {
//                 await this.trainSocket.close();
//                 break
//             }
//             default:
//                 break
//         }
//     }
//
//     private async switchTestMode(modelUrl: string) {
//         if (this.mode === Mode.Test) return;
//         await this.clearLastMode();
//         this.mode = Mode.Test;
//
//         let initialData = await provideData(modelUrl);
//         methods.load_initial(initialData, this.config);
//         this.switchingMode = false;
//     }
//
//     private async switchTrainMode() {
//         if (this.mode === Mode.Train) return;
//         await this.clearLastMode();
//         this.mode = Mode.Train;
//
//         let initialData = await serverMethods.getMostRecent();
//         methods.load_initial(initialData, this.config);
//         this.switchingMode = false;
//     }
//
//     private printInfo() {
//         console.log("Started task", this.count++, "of type", Mode[this.mode]);
//     }
// }