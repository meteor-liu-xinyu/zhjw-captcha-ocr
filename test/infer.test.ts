/**
 * 推理正确性测试：对比 Node 端 infer() 与 Python PyTorch 的 logits。
 * 运行：pnpm test
 *
 * 测试资产：
 *   src/assets/zhjw-model.scuocr —— 定稿模型（SCUOCRZ1 压缩容器，sep4 + slot 头，14.8KB/99.80%）
 *   test/zhjw-model-12k.scuocr   —— 体积优先变体（sep34，12.7KB/99.70%）
 * 参考值 test/test_data.json 由 Python 端（verify_scuocr 的解析 + FoldedCaptchaCNN 前向）生成。
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { parseScuOcrAsync, parseScuOcr, infer, decode } from '../src/model'

const root = dirname(fileURLToPath(import.meta.url))

function readBuf(p: string): ArrayBuffer {
  const buf = readFileSync(p)
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
}

async function loadModel(rel = '../src/assets/zhjw-model.scuocr') {
  return parseScuOcrAsync(readBuf(join(root, rel)))
}

const data = JSON.parse(readFileSync(join(root, 'test_data.json'), 'utf-8'))

describe('zhjw-captcha-ocr（v1.3.0 定稿模型）', () => {
  it('SCUOCRZ1 压缩容器解析（异步解压）', async () => {
    const model = await loadModel()
    expect(model.tensors.size).toBe(16)
    expect(model.tensors.has('conv1.weight')).toBe(true)
    expect(model.tensors.has('conv4.h.weight')).toBe(true)
    expect(model.tensors.has('se.fc.2.weight')).toBe(true)
    expect(model.tensors.has('head.fc.weight')).toBe(true)
  })

  it('sep34 变体（conv3+conv4 空间可分离）解析与前向', async () => {
    const model = await loadModel('../test/zhjw-model-12k.scuocr')
    expect(model.tensors.has('conv3.h.weight')).toBe(true)
    expect(model.tensors.has('conv3.v.weight')).toBe(true)
    const input = new Float32Array(data.input)
    const logits = infer(model, input)
    expect(logits.length).toBe(80)
    expect(Number.isFinite(logits[0])).toBe(true)
  })

  it('推理结果与 PyTorch 一致（误差 < 0.01）', async () => {
    const model = await loadModel()
    const input = new Float32Array(data.input)
    const logits = infer(model, input)
    let maxDiff = 0
    for (let i = 0; i < 80; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(logits[i] - data.logits[i]))
    }
    expect(maxDiff).toBeLessThan(0.01)
  })

  it('解码输出与 PyTorch 一致（text / confidence / margin）', async () => {
    const model = await loadModel()
    const input = new Float32Array(data.input)
    const logits = infer(model, input)
    const { text, confidence, margin } = decode(logits)
    expect(text).toBe(data.text)
    expect(confidence).toBeCloseTo(data.confidence, 3)
    expect(margin).toBeCloseTo(data.margin, 3)
  })

  it('同步 parseScuOcr 拒绝压缩容器（需要用异步版）', () => {
    const buf = readBuf(join(root, '../src/assets/zhjw-model.scuocr'))
    const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8))
    expect(magic).toBe('SCUOCRZ1')
    expect(() => parseScuOcr(buf)).toThrow(/Invalid .scuocr magic/)
  })
})
