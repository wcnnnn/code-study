<template>
  <div id="particles-js" class="particles-container"></div>
</template>

<script setup>
import { onMounted } from 'vue'
import 'particles.js'

onMounted(() => {
  window.particlesJS('particles-js', {
    particles: {
      number: { value: 80, density: { enable: true, value_area: 800 } },
      color: { value: '#42b883' },
      shape: { type: 'circle' },
      opacity: {
        value: 0.5,
        random: false,
        anim: { enable: false }
      },
      size: {
        value: 3,
        random: true,
        anim: { enable: false }
      },
      line_linked: {
        enable: true,
        distance: 150,
        color: '#42b883',
        opacity: 0.4,
        width: 1
      },
      move: {
        enable: true,
        speed: 2,
        direction: 'none',
        random: false,
        straight: false,
        out_mode: 'out',
        bounce: false
      }
    },
    interactivity: {
      detect_on: 'canvas',
      events: {
        onhover: { enable: true, mode: 'grab' },
        onclick: { enable: true, mode: 'push' },
        resize: true
      },
      modes: {
        grab: { distance: 140, line_linked: { opacity: 1 } },
        push: { particles_nb: 4 }
      }
    },
    retina_detect: true
  })
})
</script>

<style scoped>
.particles-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: -1;
  pointer-events: none;
}

.dark-theme .particles-container {
  filter: brightness(0.8);
}
</style> 