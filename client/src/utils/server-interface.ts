import axios from "axios";

// const ServerUrl = "http://localhost:8000";
const ServerUrl = "https://ai-playground-server.livelybay-b5b6ca38.brazilsouth.azurecontainerapps.io";

export async function digits_eval(array: number[]) {
    let response = await axios.post<number[]>(ServerUrl + "/digits/eval", array, {
        responseType: "json",
    });
    return response.data;
}

export async function wakeUp() {
    // Useless endpoint that just wakes up the Azure instance for the next requests
    return axios.get(ServerUrl + "/wakeup");
}