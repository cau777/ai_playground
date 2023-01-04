import {FC, useEffect, useState} from "react";
import {NavControls} from "../NavControls";
import {ChessBoard} from "./ChessBoard";
import * as server from "../../utils/server-interface";

const INITIAL_BOARD = `
r n b q k b n r
p p p p p p p p
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
P P P P P P P P
R N B Q K B N R`;

type BoardState = {
    board: string;
    possible: Map<string, Set<string>>;
    gameId: string;
    state: string;
    opening: string;
}

function createMapSet(possible: [string, string][]) {
    let result = new Map<string, Set<string>>();
    
    for (let [from, to] of possible) {
        if (!result.has(from))
            result.set(from, new Set<string>());
        
        let set = result.get(from)!;
        set.add(to);
    }
    
    return result;
}

export const ChessPlayground: FC = () => {
    let [game, setGame] = useState<BoardState>();
    
    useEffect(() => {
        server.chess_start_game(true)
            .then(o => {
                setGame({
                    board: o.board,
                    gameId: o.game_id,
                    possible: createMapSet(o.possible),
                    state: o.game_state,
                    opening: o.opening,
                });
            });
    }, []);
    
    async function moved(from: string, to: string) {
        if (game === undefined) return;
        if (!game.possible.get(from)?.has(to))
            return;
        console.log(from, to);
        
        let response = await server.chess_move(game.gameId, from, to);
        setGame({
            gameId: game.gameId,
            possible: createMapSet(response.possible),
            board: response.board,
            state: response.game_state,
            opening: response.opening,
        });
    }
    
    
    return (
        <NavControls>
            <div className={"mx-12"}>
                <h6>{game?.state}</h6>
                <h6>{game?.opening}</h6>
                <ChessBoard interactive={game?.state === "gameResultUndefined"} board={game?.board ?? INITIAL_BOARD}
                            possible={game?.possible ?? new Map()} onMove={moved}></ChessBoard>
            </div>
        </NavControls>
    )
}