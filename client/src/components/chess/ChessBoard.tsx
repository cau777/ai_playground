import {FC} from "react";
import {ChessPiece} from "./ChessPiece";

export const ChessBoard: FC = () => {
    return (
        <div className={"grid grid-cols-8 grid-rows-8 h-[30rem] w-[30rem] border-t-2 border-r-2"}>
            {new Array(64).fill(null).map((_, i) => (
                <div key={i}
                     className={"border-l-2 border-b-2 " + ((Math.floor(i/8)+ i % 8) % 2 == 0 ? "bg-primary-50" : "bg-primary-400")}>
                    <ChessPiece notation={"P"}></ChessPiece>
                </div>
            ))}
        </div>
    )
}