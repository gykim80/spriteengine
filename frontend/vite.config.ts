import {defineConfig} from 'vite';

export default defineConfig({
  build: {
    // Three.js 뷰포트는 LazyViewport에서 이미 지연 로드되는 별도 청크(≈640kB)라
    // 초기 번들과 무관하다. 의도된 크기이므로 경고 한도만 상향한다.
    chunkSizeWarningLimit: 700,
  },
});
