/**
 * 模型加载与推理：解析 .scuocr 权重 + CNN 前向传播。
 *
 * 支持 .scuocr 格式：
 *   version=1: fp32（每元素 4 字节）
 *   version=2: int8（每元素 1 字节，per-tensor 对称量化，zero_point 恒为 0）
 *   version=3: 混合精度（bits 低位=位宽，bit7=per-channel 标记；int4 打包存储）
 *
 * 定稿模型（v1.2.0）结构（含 SE 注意力 + 空间可分离 conv4 + slot 头）：
 *   Conv3×3(1→20)+BN+ReLU+MaxPool → 20×16×32
 *   Conv3×3(20→32)+BN+ReLU+MaxPool → 32×8×16
 *   Conv3×3(32→48)+BN+ReLU+MaxPool → 48×4×8
 *   空间可分离 conv4：横向(1×3) 48→48 + 纵向(3×1) 48→48，+ReLU+MaxPool → 48×2×4
 *   SE 注意力（48 通道，reduction=16）
 *   AdaptiveAvgPool((1,4)) → 48×1×4
 *   SlotHead：共享线性(48→20) + 每槽偏置(4×20) → 4×20
 *
 * 注意：.scuocr 权重已做 BN 折叠（foldBnIntoConv），推理时不再单独跑 BN。
 * 头类型按张量名判断：有 `head.fc.weight` 即 slot 头；否则 fc 头（兼容旧模型）。
 */

import { conv2d, maxpool2d, adaptiveAvgPool2d, linear, relu, sigmoid } from './cnn-ops'

/** 字符集（20 类，与训练一致） */
export const CHARSET = '2345678abcdefgmnpwxy'
export const CAPTCHA_LEN = 4
export const NUM_CLASSES = CHARSET.length // 20

/** 解析后的模型：张量名 → (数据, 形状) */
export interface ScuOcrModel {
  tensors: Map<string, { data: Float32Array; shape: number[] }>
  version: number
}

/**
 * 解析 .scuocr 二进制权重文件。
 * 支持 version=1 (fp32)、version=2 (int8 对称量化)、version=3 (混合精度 int8/int4)。
 */
export function parseScuOcr(buffer: ArrayBuffer): ScuOcrModel {
  const bytes = new Uint8Array(buffer)
  const dv = new DataView(buffer)
  let offset = 0

  // magic: 8 bytes "SCUOCRLT"
  const magic = new TextDecoder().decode(bytes.subarray(0, 8))
  if (magic !== 'SCUOCRLT') {
    throw new Error(`Invalid .scuocr magic: "${magic}"`)
  }
  offset += 8

  const version = dv.getUint32(offset, true)
  offset += 4
  const count = dv.getUint32(offset, true)
  offset += 4

  const tensors = new Map<string, { data: Float32Array; shape: number[] }>()

  for (let i = 0; i < count; i++) {
    // name
    const nameLen = dv.getUint32(offset, true)
    offset += 4
    const name = new TextDecoder().decode(bytes.subarray(offset, offset + nameLen))
    offset += nameLen

    // shape
    const ndim = dv.getUint32(offset, true)
    offset += 4
    const shape: number[] = []
    for (let d = 0; d < ndim; d++) {
      shape.push(dv.getUint32(offset, true))
      offset += 4
    }
    const n = shape.reduce((a, b) => a * b, 1)

    let data: Float32Array
    if (version === 1) {
      // fp32
      data = new Float32Array(buffer, offset, n)
      offset += n * 4
    } else if (version === 2) {
      // int8 对称量化：scale + zero_point + int8 data
      const scale = dv.getFloat32(offset, true)
      offset += 4
      const zeroPoint = dv.getInt32(offset, true)
      offset += 4
      const q = new Int8Array(buffer, offset, n)
      offset += n
      data = new Float32Array(n)
      for (let j = 0; j < n; j++) {
        data[j] = (q[j] - zeroPoint) * scale
      }
    } else if (version === 3) {
      // 混合精度 v3：
      //   bits 字段（bit7=per-channel 标记，低 7 位=位宽）
      //   per-tensor:   scale(f32) + zero_point(i32) + data
      //   per-channel:  占位 scale(f32) + zero_point(i32) + num_scales(u32) + scales(ns×f32) + data
      const bitsField = bytes[offset]
      offset += 1
      const bits = bitsField & 0x7f
      const perChannel = (bitsField & 0x80) !== 0

      if (perChannel) {
        offset += 4 + 4 // 占位 scale + zero_point
        const numScales = dv.getUint32(offset, true)
        offset += 4
        const scales = new Float32Array(numScales)
        for (let s = 0; s < numScales; s++) {
          scales[s] = dv.getFloat32(offset, true)
          offset += 4
        }
        data = readQuantized(bits, bytes, offset, n)
        offset += quantBytes(bits, n)
        // per-channel 反量化：每输出通道一个 scale，沿 dim0 广播
        const outDim = shape[0]
        if (numScales === outDim && outDim > 1) {
          const perCh = Math.floor(n / outDim)
          for (let co = 0; co < outDim; co++) {
            const s = scales[co]
            const base = co * perCh
            for (let j = 0; j < perCh; j++) data[base + j] *= s
          }
        } else {
          const s = numScales > 0 ? scales[0] : 1
          for (let j = 0; j < n; j++) data[j] *= s
        }
      } else {
        const scale = dv.getFloat32(offset, true)
        offset += 4
        offset += 4 // zero_point
        data = readQuantized(bits, bytes, offset, n)
        offset += quantBytes(bits, n)
        for (let j = 0; j < n; j++) data[j] *= scale
      }
    } else {
      throw new Error(`Unsupported .scuocr version: ${version}`)
    }

    tensors.set(name, { data, shape })
  }

  return { tensors, version }
}

/** int4 打包数据长度（每字节 2 值，向上取整）。 */
function quantBytes(bits: number, n: number): number {
  return bits === 8 ? n : Math.ceil(n / 2)
}

/**
 * 读取量化数据并反解为 float 数组（未乘 scale）。
 * int8: 直接读取；int4: 每字节 2 值，低 4 位在前，写入时做了 (q+8)&0x0F 偏置，解出后 -8。
 */
function readQuantized(bits: number, bytes: Uint8Array, offset: number, n: number): Float32Array {
  const data = new Float32Array(n)
  if (bits === 8) {
    for (let j = 0; j < n; j++) {
      data[j] = bytes[offset + j] << 24 >> 24 // int8 转 float
    }
    return data
  }
  // int4 打包
  const nbytes = Math.ceil(n / 2)
  for (let b = 0; b < nbytes; b++) {
    const packed = bytes[offset + b]
    const lo = (packed & 0x0f) - 8
    const hi = ((packed >> 4) & 0x0f) - 8
    data[b * 2] = lo
    if (b * 2 + 1 < n) data[b * 2 + 1] = hi
  }
  return data
}

/** 从模型取张量。 */
function t(model: ScuOcrModel, name: string): { data: Float32Array; shape: number[] } {
  const tensor = model.tensors.get(name)
  if (!tensor) {
    throw new Error(`Missing tensor: ${name}. Available: ${[...model.tensors.keys()].join(', ')}`)
  }
  return tensor
}

/**
 * 前向推理。
 *
 * @param model 解析后的模型
 * @param input 预处理后的输入，CHW 展平 (1 × 32 × 64) = 2048 个 float
 * @returns logits: Float32Array(80) = 4 位 × 20 类
 */
export function infer(model: ScuOcrModel, input: Float32Array): Float32Array {
  // ── Conv1: 1→20, 3×3, pad 1 → 20×32×64 → ReLU → MaxPool → 20×16×32
  let x = conv2d(input, 1, 32, 64, t(model, 'conv1.weight').data, t(model, 'conv1.bias').data, 20, 3, 3, 1, 1)
  relu(x.data)
  x = maxpool2d(x.data, x.c, x.h, x.w, 2, 2)

  // ── Conv2: 20→32 → 32×8×16
  x = conv2d(x.data, x.c, x.h, x.w, t(model, 'conv2.weight').data, t(model, 'conv2.bias').data, 32, 3, 3, 1, 1)
  relu(x.data)
  x = maxpool2d(x.data, x.c, x.h, x.w, 2, 2)

  // ── Conv3: 32→48 → 48×4×8
  x = conv2d(x.data, x.c, x.h, x.w, t(model, 'conv3.weight').data, t(model, 'conv3.bias').data, 48, 3, 3, 1, 1)
  relu(x.data)
  x = maxpool2d(x.data, x.c, x.h, x.w, 2, 2)

  // ── Conv4: 48→48。空间可分离（conv4.h(1×3) + conv4.v(3×1)）或标准 3×3
  if (model.tensors.has('conv4.h.weight')) {
    // 横向 (1,3) pad=(0,1) → 48×2×4；再纵向 (3,1) pad=(1,0) → 48×2×4
    const hW = t(model, 'conv4.h.weight').data
    x = conv2d(x.data, x.c, x.h, x.w, hW, new Float32Array(x.c), x.c, 1, 3, 1, 1, 0, 1)
    const vW = t(model, 'conv4.v.weight').data
    const vB = t(model, 'conv4.v.bias').data
    x = conv2d(x.data, x.c, x.h, x.w, vW, vB, x.c, 3, 1, 1, 1, 1, 0)
  } else {
    x = conv2d(x.data, x.c, x.h, x.w, t(model, 'conv4.weight').data, t(model, 'conv4.bias').data, 48, 3, 3, 1, 1)
  }
  relu(x.data)
  x = maxpool2d(x.data, x.c, x.h, x.w, 2, 2)

  // ── SE 注意力（48 通道，通道重标定）
  const c = x.c
  const h = x.h
  const w = x.w
  // Squeeze: 全局平均池化 → (48,)
  const squeezed = new Float32Array(c)
  for (let ci = 0; ci < c; ci++) {
    let sum = 0
    const base = ci * h * w
    for (let i = 0; i < h * w; i++) sum += x.data[base + i]
    squeezed[ci] = sum / (h * w)
  }
  // fc.0: c→4 + ReLU
  const se0W = t(model, 'se.fc.0.weight')
  const seIn = se0W.shape[1]
  let se = linear(squeezed, se0W.data, t(model, 'se.fc.0.bias').data, seIn, 4)
  relu(se)
  // fc.2: 4→c + Sigmoid
  const se2W = t(model, 'se.fc.2.weight')
  const seC = se2W.shape[0]
  se = linear(se, se2W.data, t(model, 'se.fc.2.bias').data, 4, seC)
  sigmoid(se)
  // Excitation: x * se.view(c,1,1)
  const excited = new Float32Array(x.data.length)
  for (let ci = 0; ci < c; ci++) {
    const base = ci * h * w
    const scale = se[ci]
    for (let i = 0; i < h * w; i++) {
      excited[base + i] = x.data[base + i] * scale
    }
  }

  // ── AdaptiveAvgPool((1,4)) → c×1×4
  x = adaptiveAvgPool2d(excited, c, h, w, 1, 4)

  if (model.tensors.has('head.fc.weight')) {
    // ── Slot 头：每槽共享线性(48→20) + 每槽独立偏置(4×20)
    // AdaptiveAvgPool 输出布局: data[ci * 4 + oj]（(c,1,4)）
    // 重排为每槽特征: slotFeat[oj * c + ci]
    const c0 = x.c // 48
    const nSlots = x.w // 4
    const slotFeat = new Float32Array(nSlots * c0)
    for (let ci = 0; ci < c0; ci++) {
      for (let oj = 0; oj < nSlots; oj++) {
        slotFeat[oj * c0 + ci] = x.data[ci * nSlots + oj]
      }
    }
    // head.fc: (20, 48)；head.slot_bias: (4, 20)
    const headW = t(model, 'head.fc.weight').data
    const headB = t(model, 'head.fc.bias').data
    const slotBias = t(model, 'head.slot_bias').data
    const logits = new Float32Array(nSlots * NUM_CLASSES)
    for (let s = 0; s < nSlots; s++) {
      const featBase = s * c0
      for (let k = 0; k < NUM_CLASSES; k++) {
        let sum = headB[k] + slotBias[s * NUM_CLASSES + k]
        const rowBase = k * c0
        for (let ci = 0; ci < c0; ci++) {
          sum += headW[rowBase + ci] * slotFeat[featBase + ci]
        }
        logits[s * NUM_CLASSES + k] = sum
      }
    }
    return logits
  }

  // ── 旧版 fc 头（兼容）
  const fc1W = t(model, 'fc1.weight')
  const fc1Out = fc1W.shape[0]
  const fc1In = fc1W.shape[1]
  let fc1 = linear(x.data, fc1W.data, t(model, 'fc1.bias').data, fc1In, fc1Out)
  relu(fc1)
  const outW = t(model, 'output_layer.weight')
  const outOut = outW.shape[0]
  const outIn = outW.shape[1]
  return linear(fc1, outW.data, t(model, 'output_layer.bias').data, outIn, outOut)
}

/**
 * 解码 logits (80 = 4×20) → 4 位字符 + 最低置信度。
 * 逐位 softmax + argmax。
 */
export function decode(logits: Float32Array): { text: string; confidence: number } {
  let text = ''
  let minConf = 1
  for (let pos = 0; pos < CAPTCHA_LEN; pos++) {
    const start = pos * NUM_CLASSES
    // softmax
    let maxExp = -Infinity
    for (let j = 0; j < NUM_CLASSES; j++) {
      if (logits[start + j] > maxExp) maxExp = logits[start + j]
    }
    let sum = 0
    const probs = new Float32Array(NUM_CLASSES)
    for (let j = 0; j < NUM_CLASSES; j++) {
      probs[j] = Math.exp(logits[start + j] - maxExp)
      sum += probs[j]
    }
    let bestIdx = 0
    let bestProb = -1
    for (let j = 0; j < NUM_CLASSES; j++) {
      probs[j] /= sum
      if (probs[j] > bestProb) {
        bestProb = probs[j]
        bestIdx = j
      }
    }
    text += CHARSET[bestIdx]
    if (bestProb < minConf) minConf = bestProb
  }
  return { text, confidence: minConf }
}