import {LanguagesProvider} from "~/components/LanguagesContext";
import {Outlet} from "@solidjs/router";

export default function PtPage() {
    return (
        <LanguagesProvider lang={"pt"}>
            <Outlet></Outlet>
        </LanguagesProvider>
    )
}