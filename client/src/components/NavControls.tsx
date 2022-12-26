import {FC, PropsWithChildren} from "react";
import {Link, useLocation} from "react-router-dom";

export const NavControls: FC<PropsWithChildren> = (props) => {
    let {pathname} = useLocation();
    console.log(pathname)
    return (
        <>
            <div className={"py-2 flex gap-4 bg-back-1 px-3 lg:px-12"}>
                <Link className={"hover:text-font-0 " + (pathname == "/digits" ? "text-font-0" : "text-font-1")} to={"/digits"}>Digits</Link>
                <Link className={"hover:text-font-0 " + (pathname == "/chess" ? "text-font-0" : "text-font-1")} to={"/chess"}>Chess (WIP)</Link>
            </div>
            {props.children}
        </>
    );
}