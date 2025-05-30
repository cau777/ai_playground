import axios from "axios";

const ServerUrl = import.meta.env.DEV
    ? "http://localhost:8000"
    : "https://aipg-versions-server2-dg3w6.ondigitalocean.app";

export async function digits_eval(array: number[]) {
    let response = await axios.post<number[]>(ServerUrl + "/digits/eval", array, {
        responseType: "json",
    });
    return response.data;
}

type StartGameResponse = {
    board: string;
    player_side: boolean;
    game_id: string;
    possible: [string, string][];
    game_state: string;
    opening: string;
};

export type StartGameOptions = {
    side?: boolean;
    openings_book: string;
}

export async function chess_start_game(options: StartGameOptions) {
    let response = await axios.post<StartGameResponse>(ServerUrl + "/chess/start", options, {
        responseType: "json",
    });
    return response.data;
}

type MoveResponse = {
    board: string;
    possible: [string, string][];
    game_state: string;
    opening: string;
}

export async function chess_move(gameId: string, from: string, to: string) {
    let response = await axios.post<MoveResponse>(ServerUrl + "/chess/move", {
        game_id: gameId, from, to,
    }, {
        responseType: "json",
    });
    return response.data;
}

export async function wakeUp() {
    // Useless endpoint that just wakes up the Azure instance for the next requests
    return axios.get(ServerUrl + "/wakeup");
}

// GameResult::Undefined => "gameResultUndefined",
//     GameResult::Draw(reason) => match reason {
//     DrawReason::Aborted => "gameResultAborted",
//         DrawReason::FiftyMoveRule => "gameResultFiftyMoveRule",
//         DrawReason::Repetition => "gameResultRepetition",
//         DrawReason::Stalemate => "gameResultStalemate",
//         DrawReason::InsufficientMaterial => "gameResultInsufficientMaterial",
// },
// GameResult::Win(side, reason) => match reason {
//     WinReason::Checkmate => match side {
//         true => "gameResultCheckmateWhite",
//             false => "gameResultCheckmateBlack",
//     },
//     WinReason::Timeout => match side {
//         true => "gameResultTimeoutWhite",
//             false => "gameResultTimeoutBlack",
//     },
// }