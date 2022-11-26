import axios from "axios";

const ServerUrl = "https://ai-playground-server.livelybay-b5b6ca38.brazilsouth.azurecontainerapps.io";

export async function evaluate(array: number[]) {
    let response = await axios.post<number[]>(ServerUrl + "/eval", array, {
        responseType: "json",
    });
    return response.data;
}

export async function wakeUp() {
    // Useless endpoint that just wakes up the Azure instance for the next requests
    return axios.get(ServerUrl + "/wakeup");
}