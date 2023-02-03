import {NavControls} from "../NavControls";
import {ChessBoard} from "./ChessBoard";
import * as server from "../../utils/server-interface";
import {Component, createSignal, onMount} from "solid-js";
import {useChessT} from "~/components/LanguagesContext";
import {BoardState} from "~/components/chess/board_state";
import {GameInfo} from "~/components/chess/GameInfo";
import {PlaygroundContainer} from "~/components/PlaygroundContainer";

const INITIAL_BOARD = `
r n b q k b n r
p p p p p p p p
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
P P P P P P P P
R N B Q K B N R`;

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

export const ChessPlayground: Component = () => {
    let [game, setGame] = createSignal<BoardState>();
    let [busy, setBusy] = createSignal(false);
    
    let board = () => game()?.board ?? INITIAL_BOARD;
    let t = useChessT();
    
    onMount(() => {
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
    });
    
    async function moved(from: string, to: string, g?: BoardState) {
        if (busy()) return;
        if (g === undefined) return;
        if (!g.possible.get(from)?.has(to))
            return;
        
        setBusy(true);
        let response = await server.chess_move(g.gameId, from, to);
        setGame({
            gameId: g.gameId,
            possible: createMapSet(response.possible),
            board: response.board,
            state: response.game_state,
            opening: response.opening,
        });
        setBusy(false);
    }
    
    let interactive = () => game()?.state === "gameResultUndefined" && !busy();
    return (
        <NavControls>
            <PlaygroundContainer>
                <h1 class={"text-3xl font-black text-primary-100 mb-3"}>{t.title}</h1>
                {/*TODO: description*/}
                <GameInfo opening={game()?.opening} result={game()?.state} playerTurn={!busy()}></GameInfo>
                <ChessBoard interactive={interactive()} board={board()}
                            possible={game()?.possible ?? new Map()}
                            onMove={(from, to) => moved(from, to, game())}></ChessBoard>
            </PlaygroundContainer>
        </NavControls>
    )
}