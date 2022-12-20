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
    // TODO: game state
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
    let [board, setBoard] = useState<BoardState>();
    
    useEffect(() => {
        server.chess_start_game(true)
            .then(o => {
                setBoard({
                    board: o.board,
                    gameId: o.game_id,
                    possible : createMapSet(o.possible),
                });
            });
    }, []);
    
    async function moved(from: string, to: string) {
        if (board === undefined) return;
        if (!board.possible.get(from)?.has(to))
            return;
        console.log(from, to);
        
        let response = await server.chess_move(board.gameId, from, to);
        setBoard({
            gameId: board.gameId,
            possible: createMapSet(response.possible),
            board: response.board,
        });
    }
    
    
    return (
        <NavControls>
            Chess
            <ChessBoard board={board?.board ?? INITIAL_BOARD} possible={board?.possible ?? new Map()} onMove={moved}></ChessBoard>
        </NavControls>
    )
}