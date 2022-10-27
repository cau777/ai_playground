import {FC} from "react";

type Props = {
    label: string;
    onClick: () => void;
    disabled?: boolean;
}

export const BtnPrimary: FC<Props> = (props) => {
    return (
        <button onClick={props.onClick} className={"bg-primary-700 rounded px-2 py-1 border-2 border-primary-800"} disabled={props.disabled}>
            {props.label}
        </button>
    )
}