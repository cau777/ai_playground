import {NavControls} from "../NavControls";
import {ChessBoard} from "./ChessBoard";
import * as server from "../../utils/server-interface";
import {Component, createSignal, Show} from "solid-js";
import {useChessT} from "~/components/LanguagesContext";
import {BoardState, createMapSet} from "~/components/chess/board_state";
import {PlaygroundContainer} from "~/components/PlaygroundContainer";
import {NewGameMenu} from "~/components/chess/NewGameMenu";
import {ChessPlay} from "~/components/chess/ChessPlay";

const INITIAL_BOARD = `
r n b q k b n r
p p p p p p p p
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
P P P P P P P P
R N B Q K B N R`;

const EMPTY_BOARD = "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ " +
    "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _";

export const ChessPlayground: Component = () => {
    let [game, setGame] = createSignal<BoardState>();
    let [loading, setLoading] = createSignal(false);
    let t = useChessT();
    
    return (
        <NavControls>
            <PlaygroundContainer>
                <h1 class={"text-3xl font-black text-primary-100 mb-3"}>{t.title}</h1>
                {/*TODO: description*/}
                
                <div class={"flex flex-col"}>
                    <NewGameMenu onClickPlay={async (options) => {
                        setLoading(true);
                        
                        try {
                            let res = await server.chess_start_game(options);
                            setGame({
                                board: res.board,
                                gameId: res.game_id,
                                possible: createMapSet(res.possible),
                                state: res.game_state,
                                opening: res.opening,
                                initialSide: res.player_side,
                            });
                        } finally {
                            setLoading(false);
                        }
                    }} loading={loading()}></NewGameMenu>
                    
                    <Show when={game() === undefined} keyed={true}>
                    <ChessBoard board={EMPTY_BOARD} reversed={false} onMove={() => {
                    }} possible={createMapSet([])} interactive={false}></ChessBoard>
                    </Show>
                    <Show when={game() !== undefined} keyed={true}>
                        <ChessPlay {...game()!} setState={setGame}></ChessPlay>
                    </Show>
                </div>
            
            </PlaygroundContainer>
        </NavControls>
    )
}