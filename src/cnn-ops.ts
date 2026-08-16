/**
 * 纯 TypeScript CNN 算子（无依赖，浏览器内运行）。
 *
 * 张量统一用 `Float32Array` 存储数据，形状用 `{ c, h, w }` 记录
 * （CHW 布局，与 PyTorch 一致）。
 */

export interface Tensor3D {
  data: Float32Array
  c: number
  h: number
  w: number
}

/**
 * 二维卷积（valid/same 由 padding 决定）。
 *
 * @param input  输入数据 (cIn × hIn × wIn)
 * @param cIn    输入通道数
 * @param hIn    输入高
 * @param wIn    输入宽
 * @param weight 卷积核 (cOut × cIn × kh × kw)
 * @param bias   偏置 (cOut)
 * @param cOut   输出通道数
 * @param kh     卷积核高
 * @param kw     卷积核宽
 * @param stride 步长
 * @param padding 填充
 */
export function conv2d(
  input: Float32Array,
  cIn: number,
  hIn: number,
  wIn: number,
  weight: Float32Array,
  bias: Float32Array,
  cOut: number,
  kh: number,
  kw: number,
  stride: number,
  padding: number,
): Tensor3D {
  const hOut = Math.floor((hIn + 2 * padding - kh) / stride + 1)
  const wOut = Math.floor((wIn + 2 * padding - kw) / stride + 1)
  const out = new Float32Array(cOut * hOut * wOut)

  for (let co = 0; co < cOut; co++) {
    for (let ho = 0; ho < hOut; ho++) {
      for (let wo = 0; wo < wOut; wo++) {
        let sum = bias[co]
        for (let ci = 0; ci < cIn; ci++) {
          for (let ki = 0; ki < kh; ki++) {
            const hi = ho * stride + ki - padding
            if (hi < 0 || hi >= hIn) continue
            for (let kj = 0; kj < kw; kj++) {
              const wj = wo * stride + kj - padding
              if (wj < 0 || wj >= wIn) continue
              const wIdx = ((co * cIn + ci) * kh + ki) * kw + kj
              const iIdx = (ci * hIn + hi) * wIn + wj
              sum += weight[wIdx] * input[iIdx]
            }
          }
        }
        out[(co * hOut + ho) * wOut + wo] = sum
      }
    }
  }
  return { data: out, c: cOut, h: hOut, w: wOut }
}

/** 2×2 最大池化（stride=2）。 */
export function maxpool2d(input: Float32Array, c: number, h: number, w: number, k: number, stride: number): Tensor3D {
  const hOut = Math.floor((h - k) / stride + 1)
  const wOut = Math.floor((w - k) / stride + 1)
  const out = new Float32Array(c * hOut * wOut)
  for (let ci = 0; ci < c; ci++) {
    for (let ho = 0; ho < hOut; ho++) {
      for (let wo = 0; wo < wOut; wo++) {
        let max = -Infinity
        for (let ki = 0; ki < k; ki++) {
          for (let kj = 0; kj < k; kj++) {
            const v = input[(ci * h + ho * stride + ki) * w + wo * stride + kj]
            if (v > max) max = v
          }
        }
        out[(ci * hOut + ho) * wOut + wo] = max
      }
    }
  }
  return { data: out, c, h: hOut, w: wOut }
}

/**
 * 自适应平均池化：把输入 (c × h × w) 池化到 (c × outH × outW)。
 * 每个输出位置对应输入的一个区域，区域内取平均（与 PyTorch 一致）。
 */
export function adaptiveAvgPool2d(
  input: Float32Array,
  c: number,
  h: number,
  w: number,
  outH: number,
  outW: number,
): Tensor3D {
  const out = new Float32Array(c * outH * outW)
  for (let ci = 0; ci < c; ci++) {
    for (let oi = 0; oi < outH; oi++) {
      const sh = Math.floor((oi * h) / outH)
      const eh = Math.ceil(((oi + 1) * h) / outH)
      for (let oj = 0; oj < outW; oj++) {
        const sw = Math.floor((oj * w) / outW)
        const ew = Math.ceil(((oj + 1) * w) / outW)
        let sum = 0
        let cnt = 0
        for (let hi = sh; hi < eh; hi++) {
          for (let wj = sw; wj < ew; wj++) {
            sum += input[(ci * h + hi) * w + wj]
            cnt++
          }
        }
        out[(ci * outH + oi) * outW + oj] = sum / cnt
      }
    }
  }
  return { data: out, c, h: outH, w: outW }
}

/** 全连接层：y = Wx + b。weight 布局 (outDim × inDim)。 */
export function linear(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
  inDim: number,
  outDim: number,
): Float32Array {
  const out = new Float32Array(outDim)
  for (let o = 0; o < outDim; o++) {
    let sum = bias[o]
    for (let i = 0; i < inDim; i++) {
      sum += weight[o * inDim + i] * input[i]
    }
    out[o] = sum
  }
  return out
}

/** ReLU 激活（原地）。 */
export function relu(x: Float32Array): Float32Array {
  for (let i = 0; i < x.length; i++) {
    if (x[i] < 0) x[i] = 0
  }
  return x
}

/** Sigmoid 激活（原地）。 */
export function sigmoid(x: Float32Array): Float32Array {
  for (let i = 0; i < x.length; i++) {
    x[i] = 1 / (1 + Math.exp(-x[i]))
  }
  return x
}