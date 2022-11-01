import {FC, useEffect, useRef, useState} from "react";
import {addLogListener, Log, removeLogListener} from "../utils/logging";
import {LogItem} from "./LogItem";
import {keepAtMost} from "../utils/arrays";

export const LogsView: FC = () => {
    // let [state, setState] = useState<Log[]>([
    //     {id: 0, time: new Date(), message: "info", level: 0},
    //     {id: 1, time: new Date(), message: "warning", level: 1},
    //     {id: 2, time: new Date(), message: "error", level: 2}]);
    
    let [state, setState] = useState<Log[]>([]);
    let ref = useRef<HTMLDivElement>(null);
    
    useEffect(() => {
        function callback(log: Log) {
            setState(s => [...keepAtMost(s, 500), log]);
        }
        
        addLogListener(callback);
        return () => removeLogListener(callback);
    }, []);
    
    useEffect(() => {
        let element = ref.current;
        if (element !== null)
            element.scrollTop = element.scrollHeight;
    });
    
    return (
        <div ref={ref}
             className={"max-h-[50vh] m-2 px-2 py-1 rounded border-2 border-back-2 bg-back-1 overflow-auto empty:hidden"}>
            {state.map(o => (
                <LogItem key={o.id} {...o}></LogItem>
            ))}
        </div>
    )
}