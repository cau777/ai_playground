import {FC} from "react";
import BarChart from "./BarChart";

type Props = {
    probabilities: number[];
}

export const DigitsResultsGraph: FC<Props> = (props) => {
    let logs = props.probabilities.map(Math.exp);
    let logSum = logs.reduce((acc, val) => acc + val);
    let weighted = logs.map(o => o / logSum);
    
    let combined: [string, number][] = [];
    for (let i = 0; i < 10; i++) {
        combined.push([i + "", weighted[i]]);
    }
    combined = combined.sort((a, b) => a[1] - b[1]);
    
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