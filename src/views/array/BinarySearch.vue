<template>
  <div class="content">
    <div v-html="renderedContent"></div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { marked } from 'marked'

const content = `
# 二分查找

二分查找是一种在有序数组中查找特定元素的搜索算法。

## 算法步骤

1. 从数组的中间元素开始
2. 如果中间元素正好是目标值，则搜索结束
3. 如果目标值大于中间元素，则在大于中间元素的那一半数组中搜索
4. 如果目标值小于中间元素，则在小于中间元素的那一半数组中搜索
5. 重复步骤2-4，直到找到目标值或确定目标值不存在

\`\`\`javascript
function binarySearch(nums, target) {
    let left = 0;
    let right = nums.length - 1;
    
    while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        if (nums[mid] === target) return mid;
        if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    
    return -1;
}
\`\`\`
`

const renderedContent = ref('')

onMounted(() => {
  renderedContent.value = marked(content)
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
</style> 