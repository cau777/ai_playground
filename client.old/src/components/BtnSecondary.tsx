import {FC} from "react";

type Props = {
    label: string;
    onClick: () => void;
}

export const BtnSecondary: FC<Props> = (props) => {
    return (
        <button onClick={props.onClick} className={"bg-primary-800 rounded px-2 py-1 border-2 border-primary-900"}>
            {props.label}
        </button>
    )
}