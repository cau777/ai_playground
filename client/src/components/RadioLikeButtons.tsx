import {Component, createSignal, For} from "solid-js";
import {BtnPrimary} from "~/components/BtnPrimary";
import {ClassList} from "~/utils/types";

type Props = {
    options: {label: string, value: any}[];
    
    selected: any;
    
    setSelected: (val: any) => void;
    
    classList?: ClassList;
}

export const RadioLikeButtons: Component<Props> = (props) => {
    return (
        <div classList={props.classList}>
            <For each={props.options}>{o => {
                let isSelected = () => props.selected == o.value;
                return (
                    <button class={"border-2 rounded px-2 py-1 transition-all duration-100"} classList={{
                        "bg-primary-700 border-primary-800":isSelected(),
                        "border-primary-700":!isSelected(),
                    }} onClick={() => props.setSelected(o.value)}>{o.label}</button>
                );
            }}</For>
        </div>
    )
}