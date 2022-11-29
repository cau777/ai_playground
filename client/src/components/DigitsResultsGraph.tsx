import {FC} from "react";
import BarChart from "./BarChart";
import {useTranslation} from "react-i18next";

type Props = {
    probabilities?: number[];
}

function prepareValues(probabilities?: number[]) {
    let weighted;
    if (probabilities !== undefined) {
        let logs = probabilities.map(Math.exp);
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
    return combined
}

export const DigitsResultsGraph: FC<Props> = (props) => {
    let {t} = useTranslation(["digits"]);
    let combined = prepareValues(props.probabilities);
    let labels = combined.map(o => o[0]);
    let data = combined.map(o => o[1]);
    
    return (
        <div className={"max-w-xl"}>
            <BarChart data={{
                labels, datasets: [{
                    data,
                    label: t("probabilityLabel")!
                }]
            }} options={{
                scales: {y: {beginAtZero: true}}
            }}></BarChart>
        </div>
    )
}