<template>
    <div class="sidebar" :class="{ 'collapsed': collapsed }">
      <div class="sidebar-content">
        <div v-for="(item, index) in menuData" :key="index" class="menu-group">
          <div class="menu-title" @click="toggleMenu(index)">
            <span>{{ item.title }}</span>
            <i class="fas fa-chevron-right" :class="{ 'rotated': openMenus[index] }"></i>
          </div>
          <transition name="slide">
            <div class="sub-menu" v-show="openMenus[index]">
              <router-link
                v-for="(subItem, subIndex) in item.children"
                :key="subIndex"
                :to="subItem.path"
                class="menu-item"
                active-class="active"
              >
                {{ subItem.title }}
              </router-link>
            </div>
          </transition>
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue'
  
  const collapsed = ref(false)
  const openMenus = ref([])

const menuData = [
  {
    title: '绘图基础',
    children: [
      { title: '绘图基础包', path: '/basic/scienceplot' }
    ]
  },
  {
    title: '回归',
    children: [
      { title: '线性回归模型详解', path: '/basic/线性回归' },
      { title: '二分查找', path: '/array/线性回归' }
    ]
  },
  {
    title: '分类',
    children: [
      { title: '数组理论基础', path: '/array/basic' },
      { title: '二分查找', path: '/array/binary-search' }
    ]
  },
  {
    title: '优化',
    children: [
      { title: '数组理论基础', path: '/array/basic' },
      { title: '二分查找', path: '/array/binary-search' }
    ]
  },
  {
    title: '评价',
    children: [
      { title: '数组理论基础', path: '/array/basic' },
      { title: '二分查找', path: '/array/binary-search' }
    ]
  }
]

const toggleMenu = (index) => {
  openMenus.value[index] = !openMenus.value[index]
}
</script>

<style scoped>
.sidebar {
  width: var(--sidebar-width);
  background-color: var(--nav-bg);
  border-right: 1px solid var(--border-color);
  height: 100%;
  transition: width 0.3s;
  overflow-x: hidden;
  overflow-y: auto;
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
  transition: background-color 0.3s;
}

.menu-title:hover {
  background-color: var(--hover-color);
}

.menu-title i {
  transition: transform 0.3s;
}

.menu-title i.rotated {
  transform: rotate(90deg);
}

.sub-menu {
  padding: 5px 0;
}

.menu-item {
  display: block;
  padding: 8px 20px 8px 40px;
  color: var(--text-secondary);
  text-decoration: none;
  transition: all 0.3s;
}

.menu-item:hover {
  color: var(--primary-color);
  background-color: var(--hover-color);
}

.menu-item.active {
  color: var(--primary-color);
  background-color: var(--primary-shadow);
}

.slide-enter-active,
.slide-leave-active {
  transition: all 0.3s ease;
}

.slide-enter-from,
.slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>