<template>
  <div class="content">
    <div v-if="error" class="error-message">
      {{ error }}
    </div>
    <div v-else-if="loading" class="loading">
      加载中...
    </div>
    <div v-else v-html="renderedContent" ref="contentRef"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { getCurrentInstance } from 'vue'

const { proxy } = getCurrentInstance()
const route = useRoute()
const renderedContent = ref('')
const loading = ref(true)
const error = ref(null)
const contentRef = ref(null)

const addCopyButtons = () => {
  document.querySelectorAll('.copy-button').forEach(button => button.remove())
  
  const codeBlocks = document.querySelectorAll('pre code')
  codeBlocks.forEach((codeBlock) => {
    const container = codeBlock.parentElement
    
    const lang = codeBlock.className.replace('language-', '')
    if (lang) {
      container.setAttribute('data-lang', lang)
    }
    
    const button = document.createElement('button')
    button.className = 'copy-button'
    button.innerHTML = '<i class="fas fa-copy"></i>'
    button.style.cssText = `
      position: absolute;
      top: 0.5em;
      right: 0.5em;
      padding: 0.5em;
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 4px;
      color: var(--text-secondary);
      cursor: pointer;
      transition: all 0.3s;
      opacity: 0.6;
      z-index: 10;
    `
    
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(codeBlock.textContent)
        button.innerHTML = '<i class="fas fa-check"></i>'
        setTimeout(() => {
          button.innerHTML = '<i class="fas fa-copy"></i>'
        }, 2000)
      } catch (err) {
        console.error('Failed to copy:', err)
        button.innerHTML = '<i class="fas fa-times"></i>'
        setTimeout(() => {
          button.innerHTML = '<i class="fas fa-copy"></i>'
        }, 2000)
      }
    })

    button.addEventListener('mouseover', () => {
      button.style.opacity = '1'
      button.style.background = 'var(--primary-color)'
      button.style.color = 'white'
    })
    
    button.addEventListener('mouseout', () => {
      button.style.opacity = '0.6'
      button.style.background = 'var(--bg-secondary)'
      button.style.color = 'var(--text-secondary)'
    })

    container.appendChild(button)
  })
}

const loadContent = async () => {
  loading.value = true
  error.value = null
  
  try {
    const content = await import(`../content/${route.params.category}/${route.params.article}.md?raw`)
    renderedContent.value = proxy.$md.render(content.default)
    
    await nextTick()
    addCopyButtons()
  } catch (err) {
    console.error('加载失败:', err)
    error.value = '内容加载失败'
  } finally {
    loading.value = false
  }
}

watch(
  () => route.fullPath,
  async () => {
    await loadContent()
    nextTick(() => {
      addCopyButtons()
    })
  },
  { immediate: true }
)

watch(
  () => renderedContent.value,
  () => {
    nextTick(() => {
      addCopyButtons()
    })
  }
)
</script>

<style scoped>
.content {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.error-message {
  color: #ef4444;
  padding: 20px;
  text-align: center;
  background: #fef2f2;
  border: 1px solid #fee2e2;
  border-radius: 8px;
  margin: 20px 0;
}

.loading {
  text-align: center;
  padding: 40px;
  color: var(--text-secondary);
}

:deep(pre) {
  position: relative !important;
  padding: 2.5em 1.5em 1.5em !important;
  background: var(--code-block-bg) !important;
  border-radius: 8px !important;
  overflow: auto !important;
  margin: 1.5em 0 !important;
}

:deep(code) {
  font-family: 'Fira Code', monospace !important;
  font-size: 0.9em !important;
  line-height: 1.5 !important;
  padding: 0 !important;
}

:deep(pre::before) {
  content: attr(data-lang);
  position: absolute !important;
  top: 0.5em !important;
  left: 1em !important;
  font-size: 0.8em !important;
  color: var(--text-secondary) !important;
  opacity: 0.7 !important;
}
</style>