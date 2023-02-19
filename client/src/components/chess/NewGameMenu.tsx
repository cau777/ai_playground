import {Component, createSignal, Show} from "solid-js";
import {StartGameOptions} from "~/utils/server-interface";
import {BtnPrimary} from "~/components/BtnPrimary";
import {Select} from "~/components/Select";
import {RadioLikeButtons} from "~/components/RadioLikeButtons";
import {useChessT} from "~/components/LanguagesContext";


type Props = {
    onClickPlay: (val: StartGameOptions) => void;
    loading: boolean;
}

export const NewGameMenu: Component<Props> = (props) => {
    let [options, setOptions] = createSignal<StartGameOptions>({
        side: true,
        openings_book: "complete",
    });
    
    let t = useChessT();
    
    return (
        <section class={"flex"}>
            <div class={"p-4 rounded border-2 bg-back-1 mb-3 border-back-2"}>
                <div class={"grid grid-cols-2 gap-y-3"}>
                    <label>{t.playAs}</label>
                    <RadioLikeButtons options={[
                        {label: t.whiteSide, value: true}, {label: t.randomSide, value: undefined},
                        {label: t.blackSide, value: false}
                    ]} setSelected={val => setOptions(prev => ({...prev, side: val}))}
                                      classList={{"flex gap-3": true}}
                                      selected={options().side}></RadioLikeButtons>
                    
                    <label>{t.openingsBook}</label>
                    <Select options={[{label: t.complete, value: "complete"}, {label: t.none, value: "none"},
                        {label: t.gambits, value: "gambits"}, {label: t.mainlines, value: "mainlines"},
                        {label: "e4", value: "e4"}, {label: "d4", value: "d4"}]}
                            onSelect={val => setOptions(prev => ({...prev, openings_book: val}))}></Select>
                </div>
                
                <div class={"grid-center mt-5"}>
                    <Show when={props.loading} keyed={true}>
                        <BtnPrimary label={t.loading} onClick={() => {
                        }} disabled={true} classList={{"text-lg font-semibold": true}}></BtnPrimary>
                    </Show>
                    <Show when={!props.loading} keyed={true}>
                        <BtnPrimary label={t.start} onClick={() => {
                            props.onClickPlay(options());
                        }} classList={{"text-lg font-semibold": true}}></BtnPrimary>
                    </Show>
                </div>
            </div>
        </section>
    )
}