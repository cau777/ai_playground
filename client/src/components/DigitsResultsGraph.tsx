import {FC} from "react";
import BarChart from "./BarChart";

type Props = {
    probabilities?: number[];
}

export const DigitsResultsGraph: FC<Props> = (props) => {
    let weighted;
    if (props.probabilities !== undefined) {
        let logs = props.probabilities.map(Math.exp);
        let logSum = logs.reduce((acc, val) => acc + val);
        weighted = logs.map(o => o / logSum);
    } else {
        weighted = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    }
    weighted = weighted.map(o => o *100);
    
    let combined: [string, number][] = [];
    for (let i = 0; i < 10; i++) {
        combined.push([i + "", weighted[i]]);
    }
    
    let labels = combined.map(o => o[0]);
    let data = combined.map(o => o[1]);
    
    console.log(labels);
    console.log(data)
    
    return (
        <div className={"max-w-xl"}>
            
            <BarChart data={{
                labels, datasets: [{
                    data,
                    label: "Probability"
                }]
            }} options={{
                scales: {y: {beginAtZero: true}}
            }}></BarChart>
        </div>
    )
}