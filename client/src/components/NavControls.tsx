import {ParentComponent} from "solid-js";
import {A} from "solid-start";
import {useChessT, useDigitsT, useHomeT, useTranslation} from "~/components/LanguagesContext";


export const NavControls: ParentComponent = (props) => {
    let trans = useTranslation();
    let digitsT = useDigitsT();
    let chessT = useChessT();
    let homeT = useHomeT();
    
    return (
        <>
            <div class={"py-2 flex gap-4 bg-back-1 px-3 lg:px-12"}>
                <A class={"hover:text-font-0 text-font-1"} activeClass={"text-font-0"}
                   href={import.meta.env.BASE_URL + trans?.lang + "/"}>{homeT.title}</A>
                <A class={"hover:text-font-0 text-font-1"} activeClass={"text-font-0"}
                       href={import.meta.env.BASE_URL + trans?.lang + "/digits"}>{digitsT.title}</A>
                <A class={"hover:text-font-0 text-font-1"} activeClass={"text-font-0"}
                       href={import.meta.env.BASE_URL + trans?.lang + "/chess"}>{chessT.title} ({homeT.statusEarlyDev})</A>
            </div>
            {props.children}
        </>
    );
}