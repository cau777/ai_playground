import {mount, StartClient} from "solid-start/entry-client";
import {wakeUp} from "~/utils/server-interface";

mount(() => {
    wakeUp().then();
    return <StartClient/>;
}, document);
