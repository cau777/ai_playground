import {Component} from "solid-js";

type Props = {
    label: string;
    onClick: () => void;
}

export const BtnSecondary: Component<Props> = (props) => {
    return (
        <button onClick={props.onClick} class={"bg-primary-800 rounded px-2 py-1 border-2 border-primary-900"}>
            {props.label}
        </button>
    )
}