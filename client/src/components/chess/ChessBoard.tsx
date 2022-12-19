import {FC, useState} from "react";
import {ChessPiece} from "./ChessPiece";

type Props = {
    board: string;
    onMove: (from: string, to: string) => void;
}

type SelectedInfo = {
    notation: string;
}

// TODO: highlight selected
// TODO: show valid moves

export const ChessBoard: FC<Props> = (props) => {
    let pieces = props.board.replaceAll("\n", "").replaceAll(" ", "").split("");
    if (pieces.length != 64) throw new RangeError("Malformed literal: invalid size");
    let [selected, setSelected] = useState<SelectedInfo>();
    
    function pieceClick(index: number) {
        let row = 7- Math.floor(index / 8);
        let col = index % 8;
        // 65 is the code point for letter A
        let notation = String.fromCodePoint(65 + col) + (row + 1);
        if (selected === undefined) {
            setSelected({notation});
        } else {
            if (selected.notation === notation) {
                setSelected(undefined);
            } else {
                // TODO: check if move is valid
                props.onMove(selected.notation, notation);
                setSelected(undefined);
            }
        }
    }
    
    return (
        <div className={"grid grid-cols-8 grid-rows-8 h-[30rem] w-[30rem] border-t-2 border-r-2"}>
            {pieces.map((piece, i) => (
                <div key={i} onClick={() => pieceClick(i)}
                     className={"p-1 border-l-2 border-b-2 " + ((Math.floor(i / 8) + i % 8) % 2 == 0 ? "bg-primary-50" : "bg-primary-400")}>
                    <ChessPiece notation={piece}></ChessPiece>
                </div>
            ))}
        </div>
    )
}