import init, {process_train, process_test, e_load, e_train} from "wasm";
import {assign, provide_data, submit} from "./utils/server_interface";

const mod = await init();
const mem = mod.memory;
if (!(mem instanceof WebAssembly.Memory)) {
    console.log("mod", mod);
    throw new Error("Maybe exports have changed? Try renaming __wbindgen_export_0 to memory");
}

function App() {
    async function btn() {
        let model_data = await provide_data("local|digits/0.model");
        let model_config = await provide_data("local|digits/model.xml");
        
        e_load(model_data, model_config);
        for (let i = 0; i < 100; i++) {
            let data = await provide_data(`azure-fs|https://aiplaygroundmodels.file.core.windows.net/static/digits/train/train.${i}.dat?sv=2021-06-08&ss=f&srt=o&sp=rl&se=2030-10-11T07:34:07Z&st=2022-10-10T23:34:07Z&sip=0.0.0.0-255.255.255.255&spr=https&sig=WqIPvmNfe52nD3KomqyRh9c40ftJHdCSIEMLCtTRxIM%3D`);
            e_train(data);
        }
    }

    async function process(count: number) {
        await init();
        let list = [];

        for (let i = 0; i < count; i++) {
            let taskResponse = await assign();
            let task = taskResponse.data;
            list.push(task);

            let data = await provide_data(task.data_path);
            let model_config = await provide_data(task.model_config);
            let model_data = await provide_data(task.model_data);

            let result;
            switch (task.task_name) {
                case "train": {
                    result = process_train(model_data, model_config, data) as Uint8Array;
                    break;
                }
                case "test": {
                    result = process_test(model_data, model_config, data, task.version, task.batch);
                    break;
                }
            }
            
            await submit(result);
        }

        console.table(list);
    }

    return (
        <div>
            Vite
            <button onClick={btn}>Test</button>
            <br/>
            <button onClick={() => process(1)}>Process 1</button>
            <br/>
            <button onClick={() => process(100)}>Process 100</button>
            <br/>
            <button onClick={() => process(500)}>Process 500</button>
            <br/>
        </div>
    )
}

export default App
