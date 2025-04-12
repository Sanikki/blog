import {navbar} from "vuepress-theme-hope";

export default navbar([
    "/",
    "/demo/",
    {
        text: "博文",
        icon: "pen-to-square",
        prefix: "/posts/",
        children: [
            {
                text: "苹果",
                icon: "pen-to-square",
                prefix: "apple/",
                children: [
                    {text: "苹果1", icon: "pen-to-square", link: "1"},
                    {text: "苹果2", icon: "pen-to-square", link: "2"},
                    "3",
                    "4",
                ],
            },
            {
                text: "香蕉",
                icon: "pen-to-square",
                prefix: "banana/",
                children: [
                    {
                        text: "香蕉 1",
                        icon: "pen-to-square",
                        link: "1",
                    },
                    {
                        text: "香蕉 2",
                        icon: "pen-to-square",
                        link: "2",
                    },
                    "3",
                    "4",
                ],
            },
            {text: "樱桃", icon: "pen-to-square", link: "cherry"},
            {text: "火龙果", icon: "pen-to-square", link: "dragonfruit"},
            "tomato",
            "strawberry",
        ],
    },
    {
        text: "Java",
        prefix: "/java/",
        children: [
            {
                text: "Java全栈",
                link: "cherry",
                children: [
                    {
                        text: "Java并发",
                        link: "1",
                    },
                    {
                        text: "Java虚拟机",
                        link: "2",
                    },
                    {
                        text: "Java集合",
                        link: "3",
                    },
                    {
                        text: "Java算法",
                        link: "4",
                    },
                    {
                        text: "网络编程",
                        link: "5"
                    }
                ],
            },
        ],
    },
    {
        text: "Spring",
        prefix: "/spring/",
        children: [
            {
                text: "Spring框架",
                link: "cherry",
                children: [
                    {
                        text: "Spring",
                        link: "1",
                    },
                    {
                        text: "SpringBoot",
                        link: "2",
                    },
                    {
                        text: "SpringCloud",
                        link: "3",
                    }
                ],
            },
        ],
    },
    {
        text: "人工智能",
        prefix: "/ai/",
        children: [
            {
                text: "AI",
                children: [
                    {
                        text: "机器学习",
                        link: "1",
                    },
                    {
                        text: "深度学习",
                        link: "deepLearning",
                    }

                ],
            }
        ],
    },
    {
        text: "V2 文档",
        icon: "book",
        link: "https://theme-hope.vuejs.press/zh/",
    },
]);
