import init, {test} from "../../codebase/pkg";

function App() {
    async function btn() {
        await init();
        let result = test();
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
