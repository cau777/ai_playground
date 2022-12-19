import {FC} from "react";
import {NavControls} from "../NavControls";
import {ChessBoard} from "./ChessBoard";

export const ChessPlayground: FC = () => {
    return (
        <NavControls>
            Chess
            <ChessBoard></ChessBoard>
        </NavControls>
    )
}