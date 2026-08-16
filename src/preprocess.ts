/**
 * 预处理流水线：验证码图片 → 模型输入 (1×32×64 CHW 展平)。
 *
 * 步骤（与训练/导出一致）：
 *   1. 原图统一缩放为 180×60（调用方保证，或在此缩放）
 *   2. 裁剪中间区域 (x=40, y=5, w=100, h=50)
 *   3. 去黑线：RGB 三通道均 < 130 → 替换为背景色 RGB(225,222,222)
 *   4. 灰度 + 反色：gray = 0.299R+0.587G+0.114B，result = 1 - gray/255
 *   5. 面积平均缩放（近似 INTER_AREA）到 64×32，CHW 展平 Float32Array
 */

export const SRC_W = 180
export const SRC_H = 60
export const CROP_X = 40
export const CROP_Y = 5
export const CROP_W = 100
export const CROP_H = 50
export const DST_W = 64
export const DST_H = 32

/**
 * 预处理一张验证码图片。
 *
 * @param imageData 原始图片像素（RGBA，尺寸应为 180×60；若不是会先缩放）
 * @returns Float32Array(2048)，CHW 展平（单通道 32×64），值域 [0,1]
 */
export function preprocess(imageData: ImageData): Float32Array {
  const { width: srcW, height: srcH, data } = imageData

  // 1. 若尺寸不是 180×60，先面积平均缩放到 180×60
  let rgba: Uint8ClampedArray
  if (srcW === SRC_W && srcH === SRC_H) {
    rgba = data
  } else {
    rgba = resizeArea(data, srcW, srcH, SRC_W, SRC_H)
  }

  // 2-4. 裁剪 + 去黑线 + 灰度反色 → cropH×cropW 灰度
  const gray = new Float32Array(CROP_H * CROP_W)
  for (let y = 0; y < CROP_H; y++) {
    for (let x = 0; x < CROP_W; x++) {
      const sx = CROP_X + x
      const sy = CROP_Y + y
      const idx = (sy * SRC_W + sx) * 4
      let r = rgba[idx]
      let g = rgba[idx + 1]
      let b = rgba[idx + 2]
      // 去黑线：三通道均 < 130 → 背景色
      if (r < 130 && g < 130 && b < 130) {
        r = 225
        g = 222
        b = 222
      }
      // 灰度 + 反色
      const grayVal = 0.299 * r + 0.587 * g + 0.114 * b
      gray[y * CROP_W + x] = 1 - grayVal / 255
    }
  }

  // 5. 面积平均缩放 → DST_H×DST_W
  const out = new Float32Array(DST_H * DST_W)
  const scaleX = CROP_W / DST_W
  const scaleY = CROP_H / DST_H
  for (let dy = 0; dy < DST_H; dy++) {
    const sy0 = Math.floor(dy * scaleY)
    const sy1 = Math.min(Math.ceil((dy + 1) * scaleY), CROP_H)
    for (let dx = 0; dx < DST_W; dx++) {
      const sx0 = Math.floor(dx * scaleX)
      const sx1 = Math.min(Math.ceil((dx + 1) * scaleX), CROP_W)
      let sum = 0
      let cnt = 0
      for (let sy = sy0; sy < sy1; sy++) {
        for (let sx = sx0; sx < sx1; sx++) {
          sum += gray[sy * CROP_W + sx]
          cnt++
        }
      }
      out[dy * DST_W + dx] = sum / cnt
    }
  }

  return out // 32×64 = 2048，单通道 CHW 展平
}

/** 面积平均缩放 RGBA 图片（近似 INTER_AREA）。 */
function resizeArea(
  src: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(dstW * dstH * 4)
  const scaleX = srcW / dstW
  const scaleY = srcH / dstH
  for (let dy = 0; dy < dstH; dy++) {
    const sy0 = Math.floor(dy * scaleY)
    const sy1 = Math.min(Math.ceil((dy + 1) * scaleY), srcH)
    for (let dx = 0; dx < dstW; dx++) {
      const sx0 = Math.floor(dx * scaleX)
      const sx1 = Math.min(Math.ceil((dx + 1) * scaleX), srcW)
      let r = 0, g = 0, b = 0, a = 0, cnt = 0
      for (let sy = sy0; sy < sy1; sy++) {
        for (let sx = sx0; sx < sx1; sx++) {
          const idx = (sy * srcW + sx) * 4
          r += src[idx]
          g += src[idx + 1]
          b += src[idx + 2]
          a += src[idx + 3]
          cnt++
        }
      }
      const oi = (dy * dstW + dx) * 4
      out[oi] = r / cnt
      out[oi + 1] = g / cnt
      out[oi + 2] = b / cnt
      out[oi + 3] = a / cnt
    }
  }
  return out
}