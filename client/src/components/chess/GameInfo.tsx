import {Component} from "solid-js";
import {useChessT} from "~/components/LanguagesContext";

type Props = {
    opening?: string;
    result?: string;
    isLoading: boolean;
    playerTurn: boolean;
}

export const GameInfo: Component<Props> = (props) => {
    let t = useChessT();
    // @ts-ignore
    let result = () => t[props.result ?? "gameResultUndefined"];
    let turn = () => {
        if (props.isLoading)
            return t.loading;
        else if (props.playerTurn)
            return t.playerTurn;
        else
            return t.aiTurn;
    };
    
    return (
        <div class={"leading-tight min-h-[5em]"}>
            {props.opening && <p>{t.opening}: {props.opening} </p>}
            <p>{t.gameResult}: {result()}</p>
            <p>{turn()}</p>
        </div>
    )
}