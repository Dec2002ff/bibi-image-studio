# Bibi Image Studio — Product & Engineering Plan

## 1. 原则

1. **不重复造轮子。** 继续复用上游已经成熟的 Qwen-Image-2.1 推理、GGUF/INT8、
   reference、mask、outpaint、gallery 和显存管理。
2. **产品层和推理层分开。** UI 不直接依赖某个具体后端实现。
3. **数据优先。** 风格、角色、生成任务都应能序列化，便于历史恢复、分享和以后迁移后端。
4. **本地优先。** 默认不上传图片；模型权重不进 Git。
5. **先稳定再花哨。** 任何 UI 大改必须在本地完成 smoke test 后合并。

## 2. 目标信息架构

主导航最终收敛为：

- Generate
- Edit
- Styles
- Characters
- Gallery
- Models
- Settings

现有 Generate / Edit / Gallery / Settings 继续作为第一阶段骨架。

## 3. Style Profile

Style Profile 不是单纯的一段 prompt，而是一份可复用创作配置。

建议字段：

```json
{
  "id": "zhaoye-comic",
  "name": "照夜肆漫画风",
  "description": "东方奇幻漫画风",
  "prompt_prefix": "",
  "prompt_suffix": "",
  "negative_prompt": "",
  "reference_images": [],
  "lora": [],
  "defaults": {
    "aspect_ratio": "16:9",
    "quality": "MiddleQuality",
    "true_cfg_scale": 1.0
  },
  "tags": ["comic", "oriental-fantasy"]
}
```

第一阶段不要训练专用模型；先验证“提示词模板 + 参考图 + LoRA + 参数”的组合是否足够稳定。

## 4. Character Profile

用于角色一致性。建议目录中的图片只保存路径，不塞进 JSON。

```json
{
  "id": "meng-siming",
  "name": "孟司嫇",
  "description": "角色固定描述",
  "reference_images": [
    "front.png",
    "three-quarter.png",
    "side.png",
    "full-body.png"
  ],
  "identity_strength": 0.85,
  "default_style_profile": "zhaoye-comic",
  "notes": ""
}
```

运行时将角色参考合并到 Qwen-Image-2.1 的多参考输入中。不要一开始额外引入换脸模型，
避免把“身份保持”和“换脸”混为一谈。

## 5. Variations / Remix

借鉴 Midjourney 的工作流，但不复制界面或商标：

- Variation：复用 seed / prompt / style / references，仅改变随机性或少量参数。
- Remix：选择上一张图作为参考，同时允许改 prompt、比例和 Style Profile。
- Upscale：第一阶段先保留原始输出；后续再评估独立超分模型。
- Reuse settings：直接复用现有 PNG metadata 能力。

## 6. Backend Adapter

定义统一内部请求：

```python
GenerationRequest(
    prompt,
    negative_prompt,
    references,
    mask,
    width,
    height,
    seed,
    steps,
    true_cfg_scale,
    loras,
    metadata,
)
```

后端至少规划：

- `DiffusersBackend`：默认，复用当前实现。
- `ComfyUIBackend`：未来可选，通过 HTTP/WebSocket API 调独立 ComfyUI 进程。

ComfyUI 代码不复制到本仓库。

## 7. 模型管理

程序仅保存 manifest，不保存模型权重。模型管理器负责：

- 检查本地是否存在。
- 显示大小 / 用途 / 许可证。
- 按硬件推荐精度。
- 下载到约定目录。
- 校验关键文件。
- 后续扫描 LoRA 目录。

Manifest 初稿位于 `config/model_manifest.json`。

## 8. 中文界面

现有 `fooocus_qwen/ui/i18n.py` 是双语言 tuple 结构，不适合无限扩语言。
不要直接把每个 tuple 生硬扩成三元组。

推荐本地实现时先重构为：

```python
T = {
    "app_title": {
        "en": "Bibi Image Studio",
        "ru": "...",
        "zh-CN": "Bibi 图像工作室"
    }
}
```

再迁移组件和运行时消息，保证已有测试不回归。

## 9. 分阶段实现

### Foundation
- 中文项目文档
- Manifest
- Style/Character schema
- 本地接管说明

### Local Bootstrap
- CUDA / GPU / RAM / disk 检测
- 安装环境
- 下载最合适的权重
- 跑现有 selftest
- 实际生成 1 张文生图 + 1 张编辑图

### Product Layer
- 中文 i18n
- 品牌与布局
- Style Profiles
- Character Profiles
- Variations / Remix
- 模型管理

### Extension
- LoRA 管理
- ComfyUI backend adapter
- 可选超分 / 背景去除 / 分割等辅助能力
