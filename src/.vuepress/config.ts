import {defineUserConfig} from "vuepress";

import theme from "./theme.js";
export default defineUserConfig({
    base: "/blog/",
    lang: "zh-CN",
    title: "SANIKKI",
    description: "vuepress-theme-hope 的博客演示",
    head: [

        ["link", {rel: "icon", href: "sanikki_circle.png"}],
        ["meta", { name: "referrer", content: "no-referrer"}],
        ["link", {rel: "preconnect", href: "https://fonts.googleapis.com"}],
        ["link", {rel: "preconnect", href: "https://fonts.gstatic.com", crossorigin: ""},],
        ["link", {
            href: "https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;700&display=swap",
            rel: "stylesheet",
        },
        ],
        // 字体
        ["link", {rel: "preconnect", href: "https://fonts.googleapis.com"}],
        ["link", {rel: "preconnect", href: "https://fonts.gstatic.com", crossorigin: ""}],
        [
            "link",
            {
                href: "https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&family=ZCOOL+KuaiLe&display=swap",
                rel: "stylesheet"
            }
        ]
    ],

    theme,

    // 和 PWA 一起启用
    // shouldPrefetch: false,
});
