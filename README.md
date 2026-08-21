# @scu-plus/zhjw-captcha-ocr

SCU 教务处（zhjw）验证码 OCR 识别包。

CNN 推理引擎 + 模型权重，**浏览器内本地推理**，不依赖云端服务、不发送任何网络请求（除加载模型权重本身）。

## 相关仓库

- **训练仓库**（模型训练 / 量化 / .scuocr 导出）：[scu-zhjw-ocr](https://github.com/meteor-liu-xinyu/scu-zhjw-ocr)

## 特性

- 纯 TypeScript 实现 CNN 算子（conv2d / maxpool / adaptiveavgpool / linear / SE 注意力），零运行时依赖
- 模型权重内置（int8 量化，**66 KB**，测试集整图准确率 **99.0%**，单字符 99.75%），加载快
- 与插件仓库通过 `ZhwjCaptchaRecognizer` 接口对接（见 `ocr-package-integration.md`）

## 安装

```bash
# npm
npm install @scu-plus/zhjw-captcha-ocr

# pnpm（插件仓库用 pnpm 管理，同样支持）
pnpm add @scu-plus/zhjw-captcha-ocr
```

> 插件仓库若使用 pnpm workspace，可在 `package.json` 中通过 `workspace:*` 协议本地引入：
> ```json
> { "dependencies": { "@scu-plus/zhjw-captcha-ocr": "workspace:*" } }
> ```

## 使用

```ts
import { createZhjwCaptchaOcr } from '@scu-plus/zhjw-captcha-ocr'

const ocr = createZhjwCaptchaOcr()
ocr.warmup() // 可选：预加载模型

// 识别验证码图片
const text = await ocr.recognize(imgElement)
// text: 4 位字符；置信度不足时返回 ""
```

### 接口

```ts
export interface ZhjwCaptchaRecognizer {
  warmup(): void
  recognize(image: HTMLImageElement): Promise<string>
}
export function createZhjwCaptchaOcr(options?: {
  modelUrl?: string // 自定义模型权重 URL（默认使用内置权重）
}): ZhjwCaptchaRecognizer
```

### 行为约定

| 项 | 约定 |
|----|------|
| 返回 | `string`，固定 4 位；识别不可靠时返回空串 `""` |
| 置信度 | 单字符最低置信度 `0.3`，低于则整张返回空串 |
| 异常 | `recognize` 抛错由调用方捕获并降级 |
| 幂等 | `warmup()` 可多次调用，无副作用 |

## 模型规格

### 网络结构（含 SE 注意力，窄版 v1.1.0）

```
Conv3×3(1→20) + BN + ReLU + MaxPool2×2   → 20×16×32
Conv3×3(20→32) + BN + ReLU + MaxPool2×2  → 32×8×16
Conv3×3(32→48) + BN + ReLU + MaxPool2×2  → 48×4×8
Conv3×3(48→48) + BN + ReLU + MaxPool2×2  → 48×2×4
SE 注意力（48 通道，通道重标定）
AdaptiveAvgPool2d((1,4))                 → 48×1×4 = 192
FC1(192→96) + ReLU → Output(96→80)
```

- 输出 `80 = 4 位 × 20 类`，逐位 argmax 解码
- 字符集：`2345678abcdefgmnpwxy`（20 类）
- 输入尺寸：`64 × 32`（宽 × 高），单通道
- 参数量 67K，int8 权重 **66 KB**（原 107 KB，-38%），测试集整图准确率 **99.0%**（原 99.6%，基本无损）

### 权重格式

- 文件：`src/assets/zhjw-model.scuocr`（SCUOCRLT 二进制，version=2 int8）
- 权重已做 BN 折叠（`foldBnIntoConv`），推理时不再单独跑 BN
- int8 对称量化：`scale = max(|w|)/127`，`zero_point = 0`

### 预处理

| 步骤 | 参数 |
|------|------|
| 原图统一缩放 | 180 × 60 |
| 裁剪中间区域 | x=40, y=5, w=100, h=50 |
| 去黑线 | RGB 三通道均 < 130 → 替换为背景色 RGB(225,222,222) |
| 灰度 + 反色 | 灰度 `0.299R+0.587G+0.114B`，反色成黑底白字 |
| 缩放 | 面积平均缩放（近似 INTER_AREA）到 64 × 32，CHW 展平 |

## 开发

```bash
pnpm install
pnpm build   # tsc 编译 + 复制权重到 dist/
pnpm test    # vitest 单元测试
```

## 模型更新

模型权重由训练仓库（`zhjw-ocr`）导出。更新步骤：

1. 在训练仓库导出窄版 int8 权重（如改动通道数需同步更新 `src/model.ts` 的 `infer()`）：
   ```bash
   python export.py checkpoints/best.narrow.pt --int8 -o zhjw-model.scuocr
   ```
2. 复制到本仓库：`src/assets/zhjw-model.scuocr`
3. 用 PyTorch 重新生成 `test/test_data.json`（保持推理测试与权重一致）
4. 重新构建发布。

## 变更记录

- **v1.1.0**：换用窄版模型 `(20,32,48,48)/fc96`，权重 107KB → 66KB，准确率 99.6% → 99.0%
- **v1.0.0**：初始版本，原版模型 `(24,40,64,64)/fc120`，权重 107KB

## License

MIT