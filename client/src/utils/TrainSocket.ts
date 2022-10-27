import * as methods from "wasm";
import * as serverMethods from "./server_interface";

// TODO: close timer
export class TrainSocket {
    private socket?: WebSocket;
    private closing = false;
    
    public async assertConnected() {
        if (this.socket !== undefined) return;
        
        let url = await serverMethods.registerTrainWorker();
        this.socket = new WebSocket(url);
        this.socket.addEventListener("message", e => {
            if (!(e.data instanceof ArrayBuffer))
                throw new TypeError("Invalid type for socket message");
            let buffer = e.data as ArrayBuffer;
            methods.load_deltas(new Uint8Array(buffer));
        });
    }
    
    public async pushIfNecessary() {
        if (!methods.should_push() || this.socket === undefined) return;
        
        let result = methods.export_bytes();
        if (result.length > 500_000) {
            console.log("Mode than 500KB", result.length);
        }
        this.socket.send(result);
    }
    
    public async close() {
        if (this.closing || this.socket === undefined) return;
        this.closing = true;
        await this.pushIfNecessary();
        this.socket.close();
        this.closing = false;
    }
}
