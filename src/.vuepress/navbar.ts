import { navbar } from "vuepress-theme-hope";

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
          { text: "苹果1", icon: "pen-to-square", link: "1" },
          { text: "苹果2", icon: "pen-to-square", link: "2" },
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
      { text: "樱桃", icon: "pen-to-square", link: "cherry" },
      { text: "火龙果", icon: "pen-to-square", link: "dragonfruit" },
      "tomato",
      "strawberry",
    ],
  },
  {
    text: "技术笔记",
    prefix: "/tech/",
    children: [
      {
        text: "Java",
        prefix: "banana/",
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
      { text: "框架",  icon:"simple-icons:spring", link: "cherry" },
      { text: "MySQL", link: "dragonfruit" },
    ],
  },
  {
    text: "V2 文档",
    icon: "book",
    link: "https://theme-hope.vuejs.press/zh/",
  },
]);
