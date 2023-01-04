import {Component} from "solid-js";
import {NavControls} from "~/components/NavControls";
import {A} from "solid-start";
import {useTranslation} from "~/components/LanguagesContext";

export const HomePage: Component = () => {
    let trans = useTranslation();
    // TODO: translate
    // TODO: improve
    
    return (
        <NavControls>
            <div class={"container my-2 mx-auto"}>
                <header>
                    <h1 class={"text-center text-3xl font-semibold"}>Welcome</h1>
                    <p>TODO: project description</p>
                </header>
                <main class={"flex flex-wrap gap-4"}>
                    <A href={import.meta.env.BASE_URL + trans?.lang + "/digits"}>
                        <div class={"bg-back-1 rounded-xl px-2 py-3"}>
                            Digits
                        </div>
                    </A>
                    <A href={import.meta.env.BASE_URL + trans?.lang + "/chess"}>
                        <div class={"bg-back-1 rounded-xl px-2 py-3"}>
                            Chess
                        </div>
                    </A>
                </main>
            </div>
        </NavControls>
    )
}