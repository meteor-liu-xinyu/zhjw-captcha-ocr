/**
 * 构建后处理：把模型权重复制到 dist/assets/。
 * tsc 只编译 TS，不复制 .scuocr 二进制，这里手动复制。
 */
import { copyFileSync, mkdirSync, existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(fileURLToPath(import.meta.url))
const src = join(root, '..', 'src', 'assets', 'zhjw-model.scuocr')
const dstDir = join(root, '..', 'dist', 'assets')
const dst = join(dstDir, 'zhjw-model.scuocr')

if (!existsSync(src)) {
  console.error(`[fix-assets] 源权重不存在: ${src}`)
  process.exit(1)
}
mkdirSync(dstDir, { recursive: true })
copyFileSync(src, dst)
console.log(`[fix-assets] 已复制权重: ${dst}`)