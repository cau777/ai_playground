import {DigitsPlayground} from "~/components/digits/DigitsPlayground";
import {Title} from "solid-start";
import {useDigitsT} from "~/components/LanguagesContext";

export default function Home() {
    let t = useDigitsT();
    return (
        <main>
            <Title>{t.title}</Title>
            <DigitsPlayground></DigitsPlayground>
        </main>
    );
}
