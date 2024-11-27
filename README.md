# Code Study - 编程学习文档系统

一个基于 Vue 3 + Vite 构建的编程学习文档系统，支持 Markdown 渲染和数学公式显示。

## 技术栈

- Vue 3
- Vue Router
- Markdown-it
- KaTeX
- Vite

## 功能特性

- ✨ Markdown 文档渲染
- 📐 LaTeX 数学公式支持
- 🎨 代码高亮
- 📱 响应式布局
- 🌲 树形导航菜单

## 项目结构

```
code-study/
├── src/
│   ├── components/        # 组件目录
│   │   └── Sidebar.vue   # 侧边栏导航组件
│   ├── views/            # 页面组件
│   │   ├── Basic.vue     # 基础课程页面
│   │   └── array/        # 数组相关课程
│   │       └── BinarySearch.vue
│   ├── App.vue           # 根组件
│   ├── main.js           # 入口文件
│   └── style.css         # 全局样式
├── public/               # 静态资源
├── index.html            # HTML 模板
└── package.json          # 项目配置文件
```

## 安装和运行

1. 安装依赖：

```bash
npm install
```

2. 启动开发服务器：

```bash
npm run dev
```

3. 构建生产版本：

```bash
npm run build
```

## 主要依赖版本

```json
{
  "dependencies": {
    "vue": "^3.5.13",
    "vue-router": "^4.5.0",
    "markdown-it": "^14.1.0",
    "markdown-it-texmath": "^1.0.0",
    "katex": "^0.16.11"
  }
}
```

## Markdown 示例

### 基础语法

```markdown
# 标题1
## 标题2

- 列表项1
- 列表项2

> 引用文本
```

### 数学公式

```markdown
单行公式：$f(x)=x^2+2x+1$

多行公式：
$$
\begin{aligned}
y &= mx + b \\
E &= mc^2
\end{aligned}
$$
```

### 代码块

```markdown
```javascript
function example() {
    console.log("Hello World");
}
```
```

## 自定义配置

### 添加新路由

在 `main.js` 中添加新路由：

```javascript
const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/your-path',
      component: () => import('./views/YourComponent.vue')
    }
  ]
})
```

### 添加导航菜单

在 `Sidebar.vue` 中的 `menuData` 数组中添加新的菜单项：

```javascript
const menuData = [
  {
    title: '新章节',
    children: [
      { title: '新小节', path: '/your-path' }
    ]
  }
]
```

## 注意事项

1. 数学公式渲染需要使用正确的 LaTeX 语法
2. 代码块需要指定语言以启用语法高亮
3. 确保所有依赖版本兼容

## 贡献指南

1. Fork 本仓库
2. 创建新的功能分支
3. 提交你的更改
4. 发起 Pull Request

## 许可证

[MIT License](LICENSE)