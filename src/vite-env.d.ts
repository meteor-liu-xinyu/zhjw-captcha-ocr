/**
 * 声明 `?url` 导入（vite 构建时把资源作为 URL 处理）。
 * 不依赖 vite/client，纯 tsc 也能编译。
 */
declare module '*?url' {
  const src: string
  export default src
}