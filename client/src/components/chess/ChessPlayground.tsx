import {FC} from "react";
import {NavControls} from "../NavControls";
import {ChessBoard} from "./ChessBoard";

const INITIAL_BOARD = `
r n b q k b n r
p p p p p p p p
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
P P P P P P P P
R N B Q K B N R`;

export const ChessPlayground: FC = () => {
    return (
        <NavControls>
            Chess
            <ChessBoard board={INITIAL_BOARD} onMove={console.log}></ChessBoard>
        </NavControls>
    )
}