import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import MarkdownIt from 'markdown-it'
import texmath from 'markdown-it-texmath'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import './style.css'
import App from './App.vue'

// 配置 markdown-it
const md = new MarkdownIt({
  html: true,
  breaks: true,
  typographer: true,
  // 自定义图片渲染
  render: {
    image(tokens, idx) {
      const token = tokens[idx]
      const src = token.attrs[token.attrIndex('src')][1]
      const alt = token.content || ''
      return `<div class="image-container">
                <img src="${src}" alt="${alt}" loading="lazy">
              </div>`
    }
  }
})

// 使用 texmath 插件
md.use(texmath, {
  engine: katex,
  delimiters: 'dollars',
  katexOptions: {
    macros: {},
    throwOnError: false,
    strict: false
  }
})

// 将 markdown-it 实例添加到全局属性中
const app = createApp(App)
app.config.globalProperties.$md = md

// 配置路由
const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/basic/scienceplot'
    },
    {
      path: '/:category/:article',
      component: () => import('./components/ContentView.vue')
    }
  ]
})

app.use(router)
app.mount('#app')
