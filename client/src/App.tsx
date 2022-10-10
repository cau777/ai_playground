import init, {process_task} from "wasm";
import axios from "axios";
import {ShareDirectoryClient} from "@azure/storage-file-share";

function App() {
    async function btn() {
        await init();
        
        // let result = process_task("", d);
        // console.log(result);
    }
    
    async function process1() {
        await init();
        let localUrl = "http://localhost:8000";
        
        let url = "https://aiplaygroundmodels.file.core.windows.net/testy?sv=2021-06-08&ss=f&srt=o&sp=rl&se=2030-10-11T07:34:07Z&st=2022-10-10T23:34:07Z&sip=0.0.0.0-255.255.255.255&spr=https&sig=WqIPvmNfe52nD3KomqyRh9c40ftJHdCSIEMLCtTRxIM%3D";
        let access = new ShareDirectoryClient(url);
        let file = access.getFileClient("train.0.dat");
        let trainData = await file.download().then(o => o.blobBody).then(o => o!.arrayBuffer());
        let model = await axios.get<ArrayBuffer>(localUrl + "/current", {responseType: "arraybuffer"});
        
        let result = process_task("", new Uint8Array(model.data), new Uint8Array(trainData));
        console.log(result);
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
