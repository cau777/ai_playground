import {Title} from "solid-start";
import {useChessT} from "~/components/LanguagesContext";
import {ChessPlayground} from "~/components/chess/ChessPlayground";

export default function Chess() {
    let t = useChessT();
    return (
        <main>
            <Title>{t.title}</Title>
            <ChessPlayground></ChessPlayground>
        </main>
    )
}