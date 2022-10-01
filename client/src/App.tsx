import init, {test_bytes, test_proto} from "codebase";

function App() {
    async function btn() {
        await init();
        // let arr = new Uint8Array([1, 2, 3, 4, 56, 255]);
        // let result = test_bytes(arr);
        let result = test_proto();
        console.log(result);
    }
    
    return (
        <div>
            Vite
            <button onClick={btn}>Test</button>
        </div>
    )
}

export default App
