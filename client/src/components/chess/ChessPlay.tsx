import {Component, createEffect, createSignal, onMount} from "solid-js";
import {BoardState, createMapSet} from "~/components/chess/board_state";
import * as server from "~/utils/server-interface";
import {GameInfo} from "~/components/chess/GameInfo";
import {ChessBoard} from "~/components/chess/ChessBoard";

type Props = BoardState & {
    setState: (val: BoardState) => void;
};

export const ChessPlay: Component<Props> = (props) => {
    let [busy, setBusy] = createSignal(false);
    
    async function moved(from: string, to: string, g?: BoardState) {
        if (busy()) return;
        if (g === undefined) return;
        if (!g.possible.get(from)?.has(to))
            return;
        
        setBusy(true);
        let response = await server.chess_move(g.gameId, from, to);
        props.setState({
            gameId: g.gameId,
            possible: createMapSet(response.possible),
            board: response.board,
            state: response.game_state,
            opening: response.opening,
            initialSide: props.initialSide,
        });
        setBusy(false);
    }
    
    let interactive = () => props.state === "gameResultUndefined" && !busy();
    let divRef: HTMLDivElement|undefined=undefined;
    
    onMount(() => {
        divRef?.scrollIntoView();
    });
    
    return (
        <div ref={divRef}>
            <ChessBoard interactive={interactive()} board={props.board} possible={props.possible}
                        onMove={(from, to) => moved(from, to, props)} reversed={!props.initialSide}></ChessBoard>
            <GameInfo opening={props.opening} result={props.state} playerTurn={!busy()}></GameInfo>
        </div>
    )
}