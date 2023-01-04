import {ParentComponent} from "solid-js";
import {A} from "solid-start";
import {useChessT, useDigitsT, useTranslation} from "~/components/LanguagesContext";


export const NavControls: ParentComponent = (props) => {
    let trans = useTranslation();
    let digitsT = useDigitsT();
    let chessT = useChessT();
    
    return (
        <>
            <div class={"py-2 flex gap-4 bg-back-1 px-3 lg:px-12"}>
                {/*TODO: translate*/}
                <A class={"hover:text-font-0 text-font-1"} activeClass={"text-font-0"}
                   href={import.meta.env.BASE_URL + trans?.lang + "/"}>Home</A>
                <A class={"hover:text-font-0 text-font-1"} activeClass={"text-font-0"}
                       href={import.meta.env.BASE_URL + trans?.lang + "/digits"}>{digitsT.title}</A>
                <A class={"hover:text-font-0 text-font-1"} activeClass={"text-font-0"}
                       href={import.meta.env.BASE_URL + trans?.lang + "/chess"}>{chessT.title} (WIP)</A>
            </div>
            {props.children}
        </>
    );
}