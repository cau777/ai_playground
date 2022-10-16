import init, {process_task} from "wasm";
import axios from "axios";
import {logger, ShareDirectoryClient, ShareFileClient} from "@azure/storage-file-share";
import {TempCache} from "./utils/TempCache";

type Task = {
    task_name: string,
    model_config: string,
    model_data: string,
    data_path: string,
}

//const ServerUrl = "https://ai-playground-server.livelybay-b5b6ca38.brazilsouth.azurecontainerapps.io";
//let url = "https://aiplaygroundmodels.file.core.windows.net/testy?sv=2021-06-08&ss=f&srt=o&sp=rl&se=2030-10-11T07:34:07Z&st=2022-10-10T23:34:07Z&sip=0.0.0.0-255.255.255.255&spr=https&sig=WqIPvmNfe52nD3KomqyRh9c40ftJHdCSIEMLCtTRxIM%3D";

let test_mode = true;
let ServerUrl = test_mode ? "http://localhost:8000" : "https://ai-playground-server.livelybay-b5b6ca38.brazilsouth.azurecontainerapps.io";

async function download(path: string) {
    path = path.startsWith("/") ? path.substring(1) : path;
    const TEST_MODE = false;
    if (TEST_MODE && path.startsWith("models")) {
        let response = await axios.get<ArrayBuffer>(path.replace("models", "http://localhost:9900"), {responseType: "arraybuffer"});
        return new Uint8Array(response.data);
    } else {
        let client = mount_client(path);
        return client.download().then(o => o.blobBody).then(o => o!.arrayBuffer()).then(o => new Uint8Array(o));
    }
}

function mount_client(path: string) {
    path = path.startsWith("/") ? path.substring(1) : path;

    let url = "https://aiplaygroundmodels.file.core.windows.net/" + path +
        "?sv=2021-06-08&ss=f&srt=o&sp=rl&se=2030-10-11T07:34:07Z&st=2022-10-10T23:34:07Z&sip=0.0.0.0-255.255.255.255&spr=https&sig=WqIPvmNfe52nD3KomqyRh9c40ftJHdCSIEMLCtTRxIM%3D";

    console.log(url);
    return new ShareFileClient(url);
}

let cache = new TempCache<Promise<Uint8Array>>();

function App() {
    async function btn() {
        await init();
    }

    async function process1() {
        await init();

        let taskResponse = await axios.post<Task>(ServerUrl + "/assign", {});
        let task = taskResponse.data;
        let name = task.task_name;

        let data = await cache.get_or(task.data_path, () => download(task.data_path));
        let model_config = await cache.get_or(task.model_config, () => download(task.model_config));
        let model_data = await cache.get_or(task.model_data, () => download(task.model_data));

        let result = process_task(name, model_data, model_config, data) as Uint8Array;
        await axios.request({
            method: "post",
            data: result,
            url: ServerUrl + "/submit",
            headers: {
                'Content-Type': 'text/plain'
            },
        })
    }

    return (
        <div>
            Vite
            <button onClick={btn}>Test</button>
            <button onClick={process1}>Process1</button>
        </div>
    )
}

export default App
