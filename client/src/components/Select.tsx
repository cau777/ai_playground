import {Component, For} from "solid-js";

type Props = {
    options: {value: string, label: string}[];
    onSelect: (val: string) => void;
}

export const Select: Component<Props> = (props) => {
    return (
        <select class={"rounded bg-back-1 px-2 py-1 border-2 border-black outline-0"} onChange={e => props.onSelect(e.currentTarget.value)}>
            <For each={props.options}>{(o) => (
                <option value={o.value}>{o.label}</option>
            )}</For>
        </select>
    )
}