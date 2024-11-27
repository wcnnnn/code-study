<template>
  <div class="content">
    <div v-html="renderedContent"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, getCurrentInstance } from 'vue'

const { proxy } = getCurrentInstance()
const content = `
# 编程语言基础

## 什么是编程？

编程是编写程序的过程。程序是按事先设计的功能和性能要求编写的指令序列。

## 基本概念

1. 变量
2. 数据类型
3. 运算符
4. 控制流程
5. 函数
6. 对象

## 数学公式示例

单行公式：$f(x)=x^2+2x+1$

多行公式：

$$
\\begin{aligned}
y &= mx + b \\\\[2ex]
E &= mc^2
\\end{aligned}
$$

## 示例代码

\`\`\`javascript
// 变量声明
let message = "Hello World";

// 函数定义
function greet(name) {
    return \`Hello, \${name}!\`;
}

// 使用函数
console.log(greet("Programmer"));
\`\`\`
`

const renderedContent = ref('')

onMounted(() => {
  renderedContent.value = proxy.$md.render(content)
})
</script>

<style scoped>
.content {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

:deep(pre) {
  background: #f6f8fa;
  padding: 16px;
  border-radius: 6px;
  overflow: auto;
}

:deep(code) {
  font-family: monospace;
}

:deep(.katex) {
  font-size: 1.1em;
}

:deep(.katex-display) {
  margin: 1em 0;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 1em 0;
}
</style> 