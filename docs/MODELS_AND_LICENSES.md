# Models, Dependencies and Licenses

本文件用于区分 **Bibi Image Studio 的程序代码**、**上游代码** 和 **模型权重**。
它不是法律意见；若要公开部署或商业使用，应重新核对各项目当时的最新许可。

| 组件 | 用途 | 来源 | 当前许可/处理原则 |
|---|---|---|---|
| Fooocus-Qwen-Image-2.1 | 当前应用底座 | ogoun/fooocus-qwen-image-2.1 | MIT；保留版权与许可文本 |
| Qwen-Image-2.1 | 主生成/编辑模型 | Qwen/Qwen-Image-2.1 | Qwen Research License；当前官方文本限制非商业用途，商业使用需另行取得许可 |
| Hugging Face Diffusers | 默认推理框架 | huggingface/diffusers | 作为依赖使用，遵循其许可证 |
| ComfyUI | 可选未来后端 | Comfy-Org/ComfyUI | GPLv3；仅作为独立程序通过 API 对接，不把其源码复制/嵌入本仓库 |
| DWPose weights | 姿势提取 | 上游安装流程 | 仅在本地安装阶段下载；需保留其各自许可信息 |
| 第三方 LoRA | 风格/角色辅助 | 用户自行添加 | 每个 LoRA 单独显示来源和许可，不假设可商用 |

## 模型文件策略

- Git 仓库不保存大模型权重。
- `.gitignore` 已排除当前 Qwen、INT8、GGUF、Turbo、DWPose 等目录。
- 新增模型必须先登记到 `config/model_manifest.json`。
- 自动下载器未来应同时显示来源、预计体积和许可提示。
- 用户手动加入的 LoRA 不应被程序默认上传或再分发。

## Qwen-Image-2.1

官方仓库：
https://github.com/QwenLM/Qwen-Image-2.1

截至 2026-10-02，官方 LICENSE 文件标题为 **Qwen RESEARCH LICENSE AGREEMENT**。
其中对模型材料的授权明确限定为 non-commercial purposes；商业使用需要向 Qwen
取得单独商业许可。

因此 Bibi Image Studio 可以继续把 Qwen-Image-2.1 作为本地研究/个人创作的默认主干，
但不能因为本应用代码采用 MIT，就把 Qwen 模型本身也视为 MIT 或默认可商用。
