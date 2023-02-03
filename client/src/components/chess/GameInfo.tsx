import {Component} from "solid-js";
import {useChessT} from "~/components/LanguagesContext";

type Props = {
    opening?: string;
    result?: string;
    
    playerTurn: boolean;
}

export const GameInfo: Component<Props> = (props) => {
    let t = useChessT();
    // @ts-ignore
    let result = () => t[props.result ?? "gameResultUndefined"];
    
    return (
        <div class={"leading-tight min-h-[5em]"}>
            {props.opening && <p>{t.opening}: {props.opening} </p>}
            <p>{t.gameResult}: {result()}</p>
            <p>{props.playerTurn ? t.playerTurn : t.aiTurn}</p>
        </div>
    )
}