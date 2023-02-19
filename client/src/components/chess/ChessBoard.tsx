import {ChessPiece} from "./ChessPiece";
import {PossibleMoveIndicator} from "./PossibleMoveIndicator";
import {Component, createSignal, Index, Show} from "solid-js";

type Props = {
    board: string;
    onMove: (from: string, to: string) => void;
    possible: Map<string, Set<string>>;
    interactive: boolean;
    reversed: boolean;
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

export const ChessBoard: Component<Props> = (props) => {
    let pieces = () => {
        let value = props.board.replaceAll("\n", "").replaceAll(" ", "").split("");
        if (value.length != 64) throw new RangeError("Malformed literal: invalid size");
        return value;
    };
    let [selected, setSelected] = createSignal<SelectedInfo>();
    
    function pieceClick(piece: string, notation: string, selected?: SelectedInfo) {
        if (!props.interactive) return;
        
        if (selected === undefined) {
            if (piece === "_") return;
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
    
    let ordered = () => props.reversed ? pieces().reverse() : pieces();
    
    return (
        <div
            class={"select-none grid grid-cols-8 grid-rows-8 max-h-[30rem] max-w-[30rem] border-t-2 border-r-2 bg-primary-400"}>
            <Index each={ordered()}>{(piece, i) => {
                let ir = () => props.reversed ? 63 - i : i;
                let notation = () => indexToNotation(ir());
                
                let lightSquare = ()=>(Math.floor(ir() / 8) + ir() % 8) % 2 == 0;
                
                let canMoveTo = () => {
                    let s = selected();
                    return s !== undefined && props.possible.get(s.notation)?.has(notation()) === true;
                };
                let isSelected = () => selected()?.notation === notation();
                
                return (
                    <div onClick={() => pieceClick(piece(), notation(), selected())}
                         class={"relative border-l-2 border-b-2"}
                         classList={{"bg-primary-50": lightSquare(), "bg-primary-200": isSelected()}}>
                        <Show when={canMoveTo()} keyed={false}>
                            <div class={"absolute t-0 l-0 w-full h-full z-10 opacity-20"}>
                                <PossibleMoveIndicator></PossibleMoveIndicator>
                            </div>
                        </Show>
                        <div class={"p-1"}>
                            <ChessPiece notation={piece()}></ChessPiece>
                        </div>
                    </div>
                );
            }}</Index>
        </div>
    )
}