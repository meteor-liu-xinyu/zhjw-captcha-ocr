/**
 * 模型加载与推理：解析 .scuocr 权重 + CNN 前向传播。
 *
 * 支持 .scuocr 格式：
 *   version=1: fp32（每元素 4 字节）
 *   version=2: int8（每元素 1 字节，per-tensor 对称量化，zero_point 恒为 0）
 *
 * 网络结构（含 SE 注意力，窄版 v1.2.0）：
 *   Conv3×3(1→20)+BN+ReLU+MaxPool → 20×16×32
 *   Conv3×3(20→32)+BN+ReLU+MaxPool → 32×8×16
 *   Conv3×3(32→48)+BN+ReLU+MaxPool → 48×4×8
 *   Conv3×3(48→48)+BN+ReLU+MaxPool → 48×2×4
 *   SE 注意力（48 通道，reduction=16）
 *   AdaptiveAvgPool((1,4)) → 48×1×4 = 192
 *   FC1(192→96)+ReLU → Output(96→80)
 *
 * 注意：.scuocr 权重已做 BN 折叠（foldBnIntoConv），推理时不再单独跑 BN。
 */

import { conv2d, maxpool2d, adaptiveAvgPool2d, linear, relu, sigmoid } from './cnn-ops'

/** 字符集（20 类，与训练一致） */
export const CHARSET = '2345678abcdefgmnpwxy'
export const CAPTCHA_LEN = 4
export const NUM_CLASSES = CHARSET.length // 20

/** 解析后的模型：张量名 → (数据, 形状) */
export interface ScuOcrModel {
  tensors: Map<string, { data: Float32Array; shape: number[] }>
}

/**
 * 解析 .scuocr 二进制权重文件。
 * 支持 version=1 (fp32) 和 version=2 (int8 对称量化)。
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
    } else {
      throw new Error(`Unsupported .scuocr version: ${version}`)
    }

    tensors.set(name, { data, shape })
  }

  return { tensors }
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

  // ── Conv4: 48→48 → 48×2×4
  x = conv2d(x.data, x.c, x.h, x.w, t(model, 'conv4.weight').data, t(model, 'conv4.bias').data, 48, 3, 3, 1, 1)
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
  // fc.0: 48→4 + ReLU
  let se = linear(squeezed, t(model, 'se.fc.0.weight').data, t(model, 'se.fc.0.bias').data, 48, 4)
  relu(se)
  // fc.2: 4→48 + Sigmoid
  se = linear(se, t(model, 'se.fc.2.weight').data, t(model, 'se.fc.2.bias').data, 4, 48)
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

  // ── AdaptiveAvgPool((1,4)) → 48×1×4 = 192
  x = adaptiveAvgPool2d(excited, c, h, w, 1, 4)

  // ── FC1: 192→96 + ReLU
  let fc1 = linear(x.data, t(model, 'fc1.weight').data, t(model, 'fc1.bias').data, 192, 96)
  relu(fc1)

  // ── Output: 96→80
  const logits = linear(fc1, t(model, 'output_layer.weight').data, t(model, 'output_layer.bias').data, 96, 80)
  return logits
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