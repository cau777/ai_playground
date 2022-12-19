import {FC, PropsWithChildren} from "react";
import {Link} from "react-router-dom";

export const NavControls: FC<PropsWithChildren> = (props) => {
    return (
        <div>
            <Link to={"/digits"}>Digits</Link>
            <Link to={"/chess"}>Chess</Link>
            {props.children}
        </div>
    );
}