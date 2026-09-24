/**
 * ZhjwCaptchaRecognizer 接口实现。
 *
 * 对接约定（见插件仓库 ocr-package-integration.md）：
 *   - warmup(): 预加载/预热模型，可多次调用，无副作用
 *   - recognize(image): 识别验证码图片，返回 4 位字符；
 *     置信度不足或图片不可用时返回空串 ""
 */

import { parseScuOcrAsync, infer, decode, type ScuOcrModel } from './model'
import { preprocess } from './preprocess'

// 模型权重：作为静态 asset 打包（SCUOCRZ1 无损压缩版，14.8KB / 整图 99.80%）。
// 用 new URL(..., import.meta.url) 兼容 Parcel 与 vite（二者均会将该文件复制为 asset 并返回可 fetch 的 URL）。
const modelUrl = new URL('./assets/zhjw-model.scuocr', import.meta.url).href

/**
 * 重试判定的 margin 阈值（2026-09-24 校准，out/confidence_calibration.json）：
 * margin = min over 4 slots of (p_top1 − p_top2)，低于此值时 recognize 返回空串
 * （由调用方触发验证码刷新）。阈值 0.30 跨模型版本稳定；更保守可用 0.40。
 */
export const MARGIN_THRESHOLD = 0.3

/**
 * 旧版置信度阈值（min-char 概率）—— **已废弃，仅为兼容保留导出**。
 * int4 量化后 min-prob 会系统性漂移（1% 分位 0.56 → 0.41），阈值不可跨版本复用，
 * 判定请用 MARGIN_THRESHOLD。
 */
export const CONFIDENCE_THRESHOLD = 0.3

export interface ZhjwCaptchaRecognizer {
  /** 预加载/预热模型（可选，无副作用） */
  warmup(): void
  /**
   * 识别验证码图片元素，返回 4 位字符。
   * 置信度不足或图片不可用时返回空串（由调用方触发验证码刷新）。
   */
  recognize(image: HTMLImageElement): Promise<string>
}

export interface ZhjwCaptchaOcrOptions {
  /** 自定义模型权重 URL（默认使用包内内置权重） */
  modelUrl?: string
}

/**
 * 工厂：创建一个识别器实例。
 */
export function createZhjwCaptchaOcr(options: ZhjwCaptchaOcrOptions = {}): ZhjwCaptchaRecognizer {
  const url = options.modelUrl ?? modelUrl
  let model: ScuOcrModel | null = null
  let loading: Promise<ScuOcrModel> | null = null

  function loadModel(): Promise<ScuOcrModel> {
    if (model) return Promise.resolve(model)
    if (!loading) {
      loading = fetch(url)
        .then((res) => {
          if (!res.ok) throw new Error(`Failed to fetch model: ${res.status} ${res.statusText}`)
          return res.arrayBuffer()
        })
        .then((buf) => parseScuOcrAsync(buf))
        .then((m) => {
          model = m
          return m
        })
        .catch((err) => {
          loading = null // 允许重试
          throw err
        })
    }
    return loading
  }

  return {
    warmup() {
      // 触发加载但不等待（无副作用，可多次调用）
      loadModel().catch(() => {
        /* 预热失败静默，recognize 时会再尝试 */
      })
    },

    async recognize(image: HTMLImageElement): Promise<string> {
      const m = await loadModel()

      // 从 HTMLImageElement 提取像素
      const width = image.naturalWidth || image.width
      const height = image.naturalHeight || image.height
      if (!width || !height) return ''

      const canvas = document.createElement('canvas')
      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext('2d')
      if (!ctx) return ''
      ctx.drawImage(image, 0, 0)
      let imageData: ImageData
      try {
        imageData = ctx.getImageData(0, 0, width, height)
      } catch {
        // 跨域图片导致 canvas 被污染
        return ''
      }

      // 预处理 → 推理 → 解码
      const input = preprocess(imageData)
      const logits = infer(m, input)
      const { text, margin } = decode(logits)

      // margin 不足（top1 与 top2 差距太小）返回空串，由调用方触发刷新重试
      if (margin < MARGIN_THRESHOLD) return ''
      return text
    },
  }
}