import {FC, useState} from "react";

type Props = {
    disabled: boolean;
    min: number;
    max: number;
    onChange: (value: number) => void;
    label: string;
}

// TODO: selected num feedback
export const Slider: FC<Props> = (props) => {
    let [state, setState] = useState(props.min);
    
    function update(value: string) {
        let num = Number(value);
        setState(num);
        props.onChange(num);
    }
    
    return (
        <label className={"flex flex-col mx-2 my-1"}>
            <span className={"text-font-2"}>{props.label}</span>
            
            <input type="range" min={props.min} max={props.max} value={state} disabled={props.disabled}
                   onChange={o => update(o.currentTarget.value)} className={""}/>
            <div className={"flex justify-between mx-1 font-semibold"}>
                <span>{props.min}</span>
                <span>{props.max}</span>
            </div>
        </label>
        
    )
}