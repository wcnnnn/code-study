# Code Study - 机器学习文档系统

一个基于 Vue 3 + Vite 构建的机器学习文档系统,支持 Markdown 渲染、数学公式显示和动态背景效果。

## 技术栈

- Vue 3 
- Vue Router
- Markdown-it
- KaTeX - 数学公式渲染
- Particles.js - 粒子动画背景
- AOS - 滚动动画
- Font Awesome - 图标库

## 功能特性

- ✨ Markdown 文档渲染
- 📐 LaTeX 数学公式支持 
- 🎨 代码高亮
- 📱 响应式布局
- 🌲 树形导航菜单
- 🌓 深色/浅色主题切换
- 🎯 粒子动画背景
- 📋 代码块一键复制
- 🔍 文档搜索功能
- 🎭 页面过渡动画

## 项目结构

```
code-study/
├── src/
│   ├── components/           # 组件目录
│   │   ├── Sidebar.vue      # 侧边栏导航组件
│   │   ├── ContentView.vue  # 内容展示组件
│   │   └── BackgroundAnimation.vue # 背景动画组件
│   ├── content/             # Markdown 文档目录
│   │   └── basic/          # 基础课程文档
│   ├── App.vue             # 根组件
│   ├── main.js            # 入口文件
│   └── style.css          # 全局样式
├── public/                # 静态资源
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
    "katex": "^0.16.11",
    "particles.js": "^2.0.0",
    "aos": "^2.3.4"
  }
}
```

## 文档编写指南

### Markdown 基础语法

```markdown
# 标题1
## 标题2

- 列表项1
- 列表项2

> 引用文本
```

### 数学公式

```markdown
行内公式：$f(x)=x^2+2x+1$

独立公式：
$$
\begin{aligned}
y &= mx + b \\
E &= mc^2
\end{aligned}
$$
```

### 代码块

```markdown
```python
def example():
    print("Hello World")
```
```

## 自定义配置

### 添加新文档

1. 在 `src/content` 目录下创建 Markdown 文件
2. 在 `Sidebar.vue` 中添加对应的导航项

### 修改导航菜单

在 `Sidebar.vue` 中的 `menuItems` 数组中添加或修改菜单项：

```javascript
const menuItems = reactive([
  {
    title: '章节名称',
    icon: 'fas fa-icon-name',
    children: [
      { title: '文档标题', path: '/category/article-name' }
    ]
  }
])
```

## 注意事项

1. 数学公式需要使用正确的 LaTeX 语法
2. 代码块需要指定语言以启用语法高亮
3. 图片资源建议放在 `public` 目录下引用

## 贡献指南

1. Fork 本仓库
2. 创建新的功能分支
3. 提交你的更改
4. 发起 Pull Request

## 许可证

[MIT License](LICENSE)