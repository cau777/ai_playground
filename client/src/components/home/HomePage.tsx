import {Component, For} from "solid-js";
import {NavControls} from "~/components/NavControls";
import {A} from "solid-start";
import {useChessT, useDigitsT, useHomeT, useTranslation} from "~/components/LanguagesContext";
import {DigitsPageIcon} from "~/components/icons/DigitsPageIcon";
import {ChessPageIcon} from "~/components/icons/ChessPageIcon";

export const HomePage: Component = () => {
    let trans = useTranslation();
    let digitsT = useDigitsT();
    let chessT = useChessT();
    let homeT = useHomeT();
    let cards = () => [
        {
            href: "/digits",
            title: digitsT.title,
            icon: <DigitsPageIcon class={"w-full"}></DigitsPageIcon>,
        },
        {
            href: "/chess",
            title: `${chessT.title} (${homeT.statusEarlyDev})`,
            icon: <ChessPageIcon></ChessPageIcon>,
        }
    ];
    
    return (
        <NavControls>
            <div class={"container my-8 mx-auto"}>
                <header class={"mb-8"}>
                    <h1 class={"text-center text-3xl font-bold mb-3 text-primary-100"}>{homeT.welcome}</h1>
                    <p class={"px-3 xl:px-5"}>
                        {homeT.description}
                        <A target={"_blank"} class={"simple-link"}
                           href={"https://github.com/cau777/ai_playground"}>GitHub</A>.
                    </p>
                </header>
                
                <main class={"grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-8 gap-4 mx-2"}>
                    <For each={cards()}>{(c, _) => (
                        <A href={import.meta.env.BASE_URL + trans?.lang + c.href}>
                            <div class={"bg-back-1 rounded-xl px-2 py-3 h-full flex flex-col"}>
                                <h4 class={"text-center font-semibold mb-3 text-lg px-2"}>{c.title}</h4>
                                <div class={"rounded-xl border-2 mt-auto mx-3 mb-2 bg-back-2"}>
                                    {c.icon}
                                </div>
                            </div>
                        </A>
                    )}</For>
                </main>
            </div>
        </NavControls>
    )
}
