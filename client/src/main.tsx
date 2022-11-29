import React from "react"
import ReactDOM from "react-dom/client"
import App from "./App"
import "./index.css"
import * as server from "./utils/server-interface";

import "./i18n";
server.wakeUp().then();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
        <App/>
    </React.StrictMode>
)
