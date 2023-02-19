import {ParentComponent} from "solid-js";
import {ClassList} from "~/utils/types";

type Props = {
    label: string;
    onClick: () => void;
    disabled?: boolean;
    classList?: ClassList;
}

export const BtnPrimary: ParentComponent<Props> = (props) => {
    
    return (
        <button onClick={props.onClick} title={props.label}
                classList={{"bg-primary-800 border-primary-900": props.disabled, ...props.classList}}
                class={"bg-primary-700 rounded px-2 py-1 border-2 border-primary-800"}
                disabled={props.disabled}>
            {props.children ?? props.label}
        </button>
    )
}