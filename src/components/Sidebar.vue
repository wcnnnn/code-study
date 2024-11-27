<template>
  <div class="sidebar" :class="{ 'collapsed': collapsed }">
    <div class="sidebar-content">
      <div v-for="(item, index) in menuItems" :key="index" class="menu-group">
        <div class="menu-title" @click="toggleMenu(item)">
          <div class="menu-icon-title">
            <i :class="getMenuIcon(item)" class="menu-icon"></i>
            <span class="menu-text">{{ item.title }}</span>
          </div>
          <i class="fas fa-chevron-right arrow-icon" :class="{ 'rotated': item.isOpen }"></i>
        </div>
        <div class="sub-menu" :style="getSubMenuStyle(item)">
          <router-link
            v-for="(subItem, subIndex) in item.children"
            :key="subIndex"
            :to="subItem.path"
            class="menu-item"
            active-class="active"
          >
            <i class="fas fa-circle-dot sub-icon"></i>
            <span class="sub-text">{{ subItem.title }}</span>
          </router-link>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const collapsed = ref(false)

// 为每个菜单项添加图标
const menuItems = reactive([
  {
    title: '绘图基础',
    icon: 'fas fa-chart-line',
    isOpen: true,
    children: [
      { title: '绘图基础包', path: '/basic/scienceplot' }
    ]
  },
  {
    title: '回归',
    icon: 'fas fa-calculator',
    isOpen: false,
    children: [
      { title: '线性回归', path: '/basic/线性回归' },
      { title: '二分查找', path: '/array/线性回归' }
    ]
  },
  {
    title: '分类',
    icon: 'fas fa-tags',
    isOpen: false,
    children: [
      { title: '数组理论基础', path: '/array/basic' },
      { title: '二分查找', path: '/array/binary-search' }
    ]
  },
  {
    title: '优化',
    icon: 'fas fa-sliders',
    isOpen: false,
    children: [
      { title: '数组理论基础', path: '/array/basic' },
      { title: '二分查找', path: '/array/binary-search' }
    ]
  },
  {
    title: '评价',
    icon: 'fas fa-star',
    isOpen: false,
    children: [
      { title: '数组理论基础', path: '/array/basic' },
      { title: '二分查找', path: '/array/binary-search' }
    ]
  }
])

const toggleMenu = (item) => {
  if (!collapsed.value) {
    item.isOpen = !item.isOpen
  }
}

const getSubMenuStyle = (item) => {
  if (collapsed.value) return { display: 'none' }
  return {
    display: item.isOpen ? 'block' : 'none',
    overflow: 'hidden',
    transition: 'all 0.3s ease'
  }
}

const getMenuIcon = (item) => {
  return item.icon || 'fas fa-folder'
}

const currentPath = computed(() => route.path)
</script>

<style scoped>
.sidebar {
  width: var(--sidebar-width);
  background-color: var(--nav-bg);
  border-right: 1px solid var(--border-color);
  height: 100vh;
  transition: all 0.3s ease;
  overflow-x: hidden;
  overflow-y: auto;
  position: fixed;
  top: 60px;
  left: 0;
  bottom: 0;
  z-index: 100;
}

.sidebar.collapsed {
  width: 60px;
}

.sidebar-content {
  padding: 20px 0;
}

.menu-group {
  margin-bottom: 8px;
}

.menu-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 20px;
  cursor: pointer;
  color: var(--text-color);
  transition: all 0.3s;
  user-select: none;
}

.menu-icon-title {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.menu-icon {
  font-size: 1.1em;
  width: 20px;
  text-align: center;
  color: var(--primary-color);
}

.menu-text {
  transition: opacity 0.3s ease;
  white-space: nowrap;
}

.collapsed .menu-text,
.collapsed .arrow-icon {
  opacity: 0;
  width: 0;
}

.arrow-icon {
  transition: all 0.3s ease;
  font-size: 0.8em;
  opacity: 0.6;
}

.arrow-icon.rotated {
  transform: rotate(90deg);
}

.menu-title:hover {
  background-color: var(--hover-color);
}

.sub-menu {
  background-color: var(--bg-secondary);
  transition: all 0.3s ease;
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 20px 8px 52px;
  color: var(--text-secondary);
  text-decoration: none;
  transition: all 0.3s;
  font-size: 0.95em;
}

.sub-icon {
  font-size: 0.5em;
  color: var(--text-secondary);
}

.menu-item:hover {
  color: var(--primary-color);
  background-color: var(--hover-color);
}

.menu-item:hover .sub-icon {
  color: var(--primary-color);
}

.menu-item.active {
  color: var(--primary-color);
  background-color: var(--primary-shadow);
  font-weight: 500;
}

.menu-item.active .sub-icon {
  color: var(--primary-color);
}

/* 优化滚动条 */
.sidebar::-webkit-scrollbar {
  width: 4px;
}

.sidebar::-webkit-scrollbar-thumb {
  background: var(--scrollbar-thumb);
  border-radius: 2px;
}

.sidebar::-webkit-scrollbar-track {
  background: var(--scrollbar-track);
}

/* 悬浮提示 */
.collapsed .menu-title {
  position: relative;
}

.collapsed .menu-title:hover::after {
  content: attr(data-title);
  position: absolute;
  left: 100%;
  top: 50%;
  transform: translateY(-50%);
  background: var(--nav-bg);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.9em;
  white-space: nowrap;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  z-index: 1000;
  margin-left: 10px;
}
</style>