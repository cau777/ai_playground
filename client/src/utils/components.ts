export function combine(...classes: (string | { [key: string]: boolean|undefined|null })[]) {
    let result = "";
    for (const cls of classes) {
        if (typeof cls === "string") {
            result += cls + " ";
        } else {
            for (const key in cls) {
                if (cls[key]) result += key + " ";
            }
        }
    }
    return result;
}