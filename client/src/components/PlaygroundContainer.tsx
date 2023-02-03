import {ParentComponent} from "solid-js";

export const PlaygroundContainer: ParentComponent = (props) => (
    <div class={"max-w-xl m-2 md:m-4 lg:m-8 xl:m-12"}>{props.children}</div>
)