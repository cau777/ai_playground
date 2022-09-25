import init, {ones} from "codebase";

function App() {
    async function btn() {
        await init();
        let result = ones();
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
