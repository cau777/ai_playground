import {FC, PropsWithChildren} from "react";
import {combine} from "../utils/components";

type Props = {
    label: string;
    onClick: () => void;
    disabled?: boolean;
}

export const BtnPrimary: FC<PropsWithChildren<Props>> = (props) => {
    
    return (
        <button onClick={props.onClick} title={props.label}
                className={combine("bg-primary-700 rounded px-2 py-1 border-2 border-primary-800", {"bg-primary-800 border-primary-900": props.disabled})}
                disabled={props.disabled}>
            {props.children ?? props.label}
        </button>
    )
}