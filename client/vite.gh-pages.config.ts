import solid from "solid-start/vite";
import {defineConfig} from "vite";
// @ts-ignore
import ssg from "solid-start-static";

export default defineConfig({
    plugins: [solid({
        adapter: ssg({})
    })],
    base: "/ai_playground/",
    build: {
        minify: "esbuild",
    }
});