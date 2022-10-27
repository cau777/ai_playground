export type Log = {message: string, level: number, time: Date, id: number};
type LogCallback = (log: Log) => void;

let callbacks = new Set<LogCallback>();
let currentId= 0;

export function addLogListener(func: LogCallback) {
    callbacks.add(func);
}

export function removeLogListener(func: LogCallback) {
    callbacks.delete(func);
}

export function insertLog(message: string, level: number) {
    callbacks.forEach(o => o({message, level, time: new Date(), id: currentId++}));
}