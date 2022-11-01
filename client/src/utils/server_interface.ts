import {ShareFileClient} from "@azure/storage-file-share";
import axios from "axios";
import {TempCache} from "./TempCache";

const TEST_MODE = true;
const ServerUrl = TEST_MODE ? "http://localhost:8000" : "https://ai-playground-server.livelybay-b5b6ca38.brazilsouth.azurecontainerapps.io";
const Cache = new TempCache<Promise<Uint8Array>>();

export function provideData(url: string) {
    return Cache.get_or(url, async () => {
        let [prefix, path] = url.split("|", 2);
        path = path.startsWith("/") ? path.substring(1) : path;

        switch (prefix) {
            case "azure-fs": {
                return new ShareFileClient(path).download()
                    .then(o => o.blobBody)
                    .then(o => o!.arrayBuffer())
                    .then(o => new Uint8Array(o));
            }
            case "local": {
                let response = await axios.get<ArrayBuffer>("http://localhost:9900/" + path, {responseType: "arraybuffer"});
                return new Uint8Array(response.data);
            }
            default:
                throw new RangeError(`Prefix ${prefix} is not supported`);
        }
    });
}

export type Task = {
    type: "Train",
    url: string,
} | {
    type: "Validate",
    version: number,
    batch: number,
    model_url: string,
    url: string
};

export function assign() {
    return axios.get<Task>(ServerUrl + "/assign").then(o => o.data);
}

export function submitTest(version: number, batch: number, accuracy: number) {
    return axios.post(ServerUrl + "/submit_test", {
        version, batch, accuracy
    });
}

export function registerTrainWorker() {
    return axios.get<string>(ServerUrl + "/register").then(o => o.data);
}

export async function getMostRecent() {
    let response = await axios.get<ArrayBuffer>(ServerUrl + "/recent", {responseType: "arraybuffer"});
    return new Uint8Array(response.data);
}

export async function getBest() {
    return axios.get<string>(ServerUrl + "/best").then(o => o.data)
}

export function submitTrain(bytes: Uint8Array) {
    return axios.post(ServerUrl + "/submit_train", bytes);
}