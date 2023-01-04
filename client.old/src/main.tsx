import React from "react"
import ReactDOM from "react-dom/client"
import App from "./App"
import "./index.css"
import * as server from "./utils/server-interface";

import "./i18n";
import {createBrowserRouter, RouterProvider} from "react-router-dom";
import {NotFound} from "./components/NotFound";
import {DigitsPlayground} from "./components/digits/DigitsPlayground";
import {ChessPlayground} from "./components/chess/ChessPlayground";

server.wakeUp().then();

const router = createBrowserRouter([
    {
        path: "/digits",
        element: <DigitsPlayground/>,
    },
    {
        path: "/chess",
        element: <ChessPlayground/>,
    },
    {
        path: "/",
        element: <App/>,
        errorElement: <NotFound/>,
    },
], {basename: import.meta.env.BASE_URL})

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
        <RouterProvider router={router}></RouterProvider>
    </React.StrictMode>
)
