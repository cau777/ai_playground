import {ShareFileClient} from "@azure/storage-file-share";
import axios from "axios";
import {TempCache} from "./TempCache";

const Cache = new TempCache<Promise<Uint8Array>>();

export function provide_data(url: string) {
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
    model_config: string,
    model_data: string,
    data_path: string,
} & ({
    task_name: "train",
} | {
    task_name: "test",
    version: number,
    batch: number,
});

let TEST_MODE = true;
let ServerUrl = TEST_MODE ? "http://localhost:8000" : "https://ai-playground-server.livelybay-b5b6ca38.brazilsouth.azurecontainerapps.io";

export function assign() {
    return axios.post<Task>(ServerUrl + "/assign", {});
}

export function submit(body: Uint8Array) {
    return axios.request({
        method: "post",
        data: body,
        url: ServerUrl + "/submit",
        headers: {
            'Content-Type': 'text/plain'
        },
    })
}