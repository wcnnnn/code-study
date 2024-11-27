<template>
  <div class="app-container" :class="{ 'dark-mode': isDarkMode }">
    <!-- 顶部导航栏 -->
    <nav class="navbar">
      <div class="nav-left">
        <button class="toggle-btn" @click="toggleSidebar">
          <i class="fas fa-bars"></i>
        </button>
        <h1 class="logo">Code Study</h1>
      </div>
      <div class="nav-right">
        <div class="search-box">
          <input type="text" placeholder="搜索知识点...">
          <i class="fas fa-search"></i>
        </div>
        <button class="theme-btn" @click="toggleTheme">
          <i :class="isDarkMode ? 'fas fa-sun' : 'fas fa-moon'"></i>
        </button>
      </div>
    </nav>

    <!-- 主要内容区 -->
    <div class="main-content">
      <Sidebar :class="{ 'collapsed': isSidebarCollapsed }" />
      <div class="content-container">
        <div class="breadcrumb">
          <router-link to="/">首页</router-link>
          <template v-for="(item, index) in breadcrumbs" :key="index">
            <span class="separator">/</span>
            <router-link :to="item.path">{{ item.name }}</router-link>
          </template>
        </div>
        <router-view v-slot="{ Component }">
          <transition name="page" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import Sidebar from './components/Sidebar.vue'

const isDarkMode = ref(false)
const isSidebarCollapsed = ref(false)
const route = useRoute()

const toggleTheme = () => {
  isDarkMode.value = !isDarkMode.value
  document.documentElement.classList.toggle('dark-theme')
}

const toggleSidebar = () => {
  isSidebarCollapsed.value = !isSidebarCollapsed.value
}

const breadcrumbs = computed(() => {
  const paths = route.path.split('/').filter(Boolean)
  return paths.map((path, index) => ({
    name: path.charAt(0).toUpperCase() + path.slice(1),
    path: '/' + paths.slice(0, index + 1).join('/')
  }))
})
</script>

<style>
.app-container {
  min-height: 100vh;
  background-color: var(--bg-color);
  color: var(--text-color);
}

.navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 60px;
  background-color: var(--nav-bg);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  z-index: 1000;
}

.nav-left, .nav-right {
  display: flex;
  align-items: center;
  gap: 20px;
}

.logo {
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--primary-color);
  margin: 0;
}

.toggle-btn, .theme-btn {
  background: none;
  border: none;
  color: var(--text-color);
  font-size: 1.2rem;
  cursor: pointer;
  padding: 8px;
  border-radius: 4px;
  transition: background-color 0.3s;
}

.toggle-btn:hover, .theme-btn:hover {
  background-color: var(--hover-color);
}

.search-box {
  position: relative;
}

.search-box input {
  padding: 8px 32px 8px 12px;
  border: 1px solid var(--border-color);
  border-radius: 20px;
  background-color: var(--input-bg);
  color: var(--text-color);
  outline: none;
  transition: all 0.3s;
  width: 200px;
}

.search-box input:focus {
  border-color: var(--primary-color);
  box-shadow: 0 0 0 2px var(--primary-shadow);
}

.search-box i {
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--icon-color);
}

.main-content {
  display: flex;
  padding-top: 60px;
  min-height: calc(100vh - 60px);
}

.content-container {
  flex: 1;
  padding: 20px;
  max-width: calc(100% - var(--sidebar-width));
  transition: max-width 0.3s;
}

.breadcrumb {
  margin-bottom: 20px;
  padding: 8px 16px;
  background-color: var(--breadcrumb-bg);
  border-radius: 4px;
  font-size: 0.9rem;
}

.breadcrumb a {
  color: var(--primary-color);
  text-decoration: none;
}

.breadcrumb .separator {
  margin: 0 8px;
  color: var(--text-secondary);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* 暗色主题变量 */
:root {
  --primary-color: #42b883;
  --primary-shadow: rgba(66, 184, 131, 0.2);
  --bg-color: #ffffff;
  --nav-bg: #ffffff;
  --text-color: #2c3e50;
  --text-secondary: #666666;
  --border-color: #dcdfe6;
  --hover-color: #f5f7fa;
  --input-bg: #ffffff;
  --icon-color: #909399;
  --breadcrumb-bg: #f8f9fa;
  --sidebar-width: 250px;
}

/* 暗色主题 */
.dark-theme {
  --primary-color: #42d392;
  --primary-shadow: rgba(66, 211, 146, 0.2);
  --bg-color: #1a1a1a;
  --nav-bg: #242424;
  --text-color: #e5e7eb;
  --text-secondary: #9ca3af;
  --border-color: #4b5563;
  --hover-color: #2d2d2d;
  --input-bg: #242424;
  --icon-color: #9ca3af;
  --breadcrumb-bg: #242424;
}

/* 页面过渡动画 */
.page-enter-active,
.page-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.page-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.page-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}
</style>
