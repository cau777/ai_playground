import {Component, createEffect} from "solid-js";
import {useDigitsT} from "~/components/LanguagesContext";
import "../../utils/chart-setup";

import {
    Chart,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    BarController
} from "chart.js";

Chart.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    BarController,
);

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
    weighted = weighted.map(o => o * 100);
    
    let combined: [string, number][] = [];
    for (let i = 0; i < 10; i++) {
        combined.push([i + "", weighted[i]]);
    }
    return combined
}

export const DigitsResultsGraph: Component<Props> = (props) => {
    let t = useDigitsT();
    let combined = () => prepareValues(props.probabilities);
    let canvas: HTMLCanvasElement | undefined = undefined;
    let chart: Chart | undefined = undefined;
    
    function prepare_data_obj(labels: string[], data: number[]) {
        return {
            labels,
            datasets: [{
                data,
                label: t.probabilityLabel,
            }]
        };
    }
    
    createEffect(() => {
        let c = combined();
        let labels = c.map(o => o[0]);
        let data = c.map(o => o[1]);
        console.log("change");
        
        if (chart === undefined) {
            chart = new Chart(canvas!, {
                type: "bar",
                data: prepare_data_obj(labels, data),
            });
        } else {
            chart.data = prepare_data_obj(labels, data);
            chart.update();
        }
    });
    
    return (
        <div class={"max-w-xl"}>
            <canvas ref={canvas}></canvas>
        </div>
    )
}