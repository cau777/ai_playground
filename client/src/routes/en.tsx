import {LanguagesProvider} from "~/components/LanguagesContext";
import {Outlet} from "@solidjs/router";

export default function EnPage() {
    return (
        <LanguagesProvider lang={"en"}>
            <Outlet></Outlet>
        </LanguagesProvider>
    )
}