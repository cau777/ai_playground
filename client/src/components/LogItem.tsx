import {FC, memo} from "react";
import {Log} from "../utils/logging";
import dateFormat from "dateformat";

type Props = Log;

function style(level: number) {
    switch (level) {
        case 0: // Info
            return "";
        case 1: // Warning
            return "text-orange-400";
        case 2:
            return "text-red-400"
    }
}

function formatTime(time: Date) {
    return dateFormat(time, "hh:MM:ss");
}

const LogItemBase: FC<Props> = (props) => {
    return (
        <div className={"flex border-b-[1px] border-back-3 last:border-b-0 rounded py-0.5"}>
            <div className={"text-sm text-font-2 font-mono "+style(props.level)}>{props.message}</div>
            <div className={"ml-auto text-xs text-font-3 my-auto"}>{formatTime(props.time)}</div>
        </div>
    )
}

export const LogItem = memo(LogItemBase);