/**
 * 推理正确性测试：对比 Node 端 infer() 与 Python PyTorch 的 logits。
 * 运行：pnpm test
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { parseScuOcr, infer, decode } from '../src/model'

const root = dirname(fileURLToPath(import.meta.url))

function loadModel() {
  const buf = readFileSync(join(root, '..', 'src', 'assets', 'zhjw-model.scuocr'))
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
  return parseScuOcr(ab)
}

describe('zhjw-captcha-ocr', () => {
  it('解析 .scuocr 权重', () => {
    const model = loadModel()
    expect(model.tensors.size).toBe(16)
    expect(model.tensors.has('conv1.weight')).toBe(true)
    expect(model.tensors.has('se.fc.2.weight')).toBe(true)
  })

  it('推理结果与 PyTorch 一致（误差 < 0.01）', () => {
    const model = loadModel()
    const data = JSON.parse(readFileSync(join(root, 'test_data.json'), 'utf-8'))
    const input = new Float32Array(data.input)
    const logits = infer(model, input)

    let maxDiff = 0
    for (let i = 0; i < 80; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(logits[i] - data.logits[i]))
    }
    expect(maxDiff).toBeLessThan(0.01)
  })

  it('解码输出与 PyTorch 一致', () => {
    const model = loadModel()
    const data = JSON.parse(readFileSync(join(root, 'test_data.json'), 'utf-8'))
    const input = new Float32Array(data.input)
    const logits = infer(model, input)
    const { text, confidence } = decode(logits)
    expect(text).toBe('p2xg')
    expect(confidence).toBeCloseTo(0.2605, 3)
  })
})