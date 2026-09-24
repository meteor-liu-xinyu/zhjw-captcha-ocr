/**
 * @scu-plus/zhjw-captcha-ocr
 *
 * SCU 教务处验证码 OCR 识别包。
 * CNN 推理引擎 + 模型权重，浏览器内本地推理，不依赖云端服务。
 *
 * 用法：
 *   import { createZhjwCaptchaOcr } from '@scu-plus/zhjw-captcha-ocr'
 *   const ocr = createZhjwCaptchaOcr()
 *   ocr.warmup()
 *   const text = await ocr.recognize(imgElement)
 */

export {
  createZhjwCaptchaOcr,
  CONFIDENCE_THRESHOLD,
  MARGIN_THRESHOLD,
  type ZhjwCaptchaRecognizer,
  type ZhjwCaptchaOcrOptions,
} from './recognizer'

export {
  CHARSET,
  CAPTCHA_LEN,
  NUM_CLASSES,
  parseScuOcr,
  parseScuOcrAsync,
  infer,
  decode,
  type ScuOcrModel,
} from './model'
export { preprocess } from './preprocess'