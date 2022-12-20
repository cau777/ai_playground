import {FC, useState} from "react";
import {ChessPiece} from "./ChessPiece";
import {PossibleMoveIndicator} from "./PossibleMoveIndicator";

type Props = {
    board: string;
    onMove: (from: string, to: string) => void;
    possible: Map<string, Set<string>>;
    interactive: boolean;
}

type SelectedInfo = {
    notation: string;
}

function indexToNotation(index: number) {
    let row = 7 - Math.floor(index / 8);
    let col = index % 8;
    // 65 is the code point for letter A
    return String.fromCodePoint(65 + col) + (row + 1);
}

export const ChessBoard: FC<Props> = (props) => {
    let pieces = props.board.replaceAll("\n", "").replaceAll(" ", "").split("");
    if (pieces.length != 64) throw new RangeError("Malformed literal: invalid size");
    let [selected, setSelected] = useState<SelectedInfo>();
    
    function pieceClick(notation: string) {
        if(!props.interactive) return;
        if (selected === undefined) {
            setSelected({notation});
        } else {
            if (selected.notation === notation) {
                setSelected(undefined);
            } else {
                props.onMove(selected.notation, notation);
                setSelected(undefined);
            }
        }
    }
    
    return (
        <div className={"grid grid-cols-8 grid-rows-8 h-[30rem] w-[30rem] border-t-2 border-r-2"}>
            {pieces.map((piece, i) => {
                let notation = indexToNotation(i);
                let lightSquare = (Math.floor(i / 8) + i % 8) % 2 == 0;
                let canMoveTo = selected !== undefined && props.possible.get(selected.notation)?.has(notation) === true;
                
                return (
                    <div key={i} onClick={() => pieceClick(notation)}
                         className={"relative border-l-2 border-b-2 " +
                             (lightSquare ? "bg-primary-50 " : "bg-primary-400 ") +
                             (selected?.notation === notation ? "bg-primary-200 " : "")}>
                        {canMoveTo && (
                            <div className={"absolute t-0 l-0 w-full h-full z-10 opacity-20"}>
                                <PossibleMoveIndicator></PossibleMoveIndicator>
                            </div>
                        )}
                        <div className={"p-1"}>
                            <ChessPiece notation={piece} canMoveTo={canMoveTo}></ChessPiece>
                        </div>
                    </div>
                );
            })}
        </div>
    )
}