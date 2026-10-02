# Bibi Image Studio

> 基于 Qwen-Image-2.1 的本地图像生成与编辑工作台。当前仓库由
> [Fooocus-Qwen-Image-2.1](https://github.com/ogoun/fooocus-qwen-image-2.1)
> 二次开发而来，目标是保留成熟的低显存推理和编辑能力，同时把创作体验做得更适合日常视觉创作。

**当前状态：Foundation 阶段。** 现有底座已经具备文生图、最多 10 张参考图、
Mask / 标注 / 精确区域编辑、Outpainting、姿势参考、画廊、质量预设，以及针对
8–24 GB NVIDIA 显卡的 bf16 / INT8 / GGUF 运行方案。

## 我们要增加什么

Bibi Image Studio 不重新发明推理引擎。核心方向是把已有能力重新组织成更直观的创作流程：

- **Generate**：提示词、比例、质量、Seed、多图批量、参考图。
- **Edit**：指令编辑、Mask、区域编辑、扩图、换背景、透明图。
- **Style Profiles**：保存可复用的风格档案，包含提示词模板、参考图、LoRA 和默认参数。
- **Character Profiles**：为同一角色保存多张身份参考、描述与默认一致性设置。
- **Variations / Remix**：围绕现有结果继续做变体，而不是每次从头输入。
- **Model / LoRA Manager**：本地模型和 LoRA 的发现、启停、权重调整与下载清单。
- **Backend Adapter**：默认继续使用当前 Diffusers 后端，后续可选连接独立运行的 ComfyUI，
  不把 ComfyUI GPL 源码复制进本仓库。
- **中文界面**：在现有英文/俄文基础上增加简体中文。

详细路线见 [docs/BIBI_STUDIO_PLAN.md](docs/BIBI_STUDIO_PLAN.md)。

## 现在能不能直接运行？

底座本身可以运行，但这个 fork 的 Bibi 定制功能尚在开发分支中。首次真正安装、下载模型、
检测 CUDA、验证显存和实际出图，会在本地阶段由 Codex 执行。

Windows 基础运行方式仍沿用上游：

```powershell
.\install.ps1
.\run.ps1
```

模型权重不会提交到 GitHub。

## 模型与许可

生成主干为 **Qwen-Image-2.1**。模型许可与本项目程序代码许可不是一回事。
Qwen-Image-2.1 当前使用 Qwen Research License，非商业使用与商业使用条件不同。
请在发布或商业使用前阅读 [docs/MODELS_AND_LICENSES.md](docs/MODELS_AND_LICENSES.md)。

本 fork 继承的 Fooocus-Qwen 软件代码采用 MIT License；保留原作者版权与许可声明。

## 开发

当前开发分支：

```
bibi-foundation
```

本阶段只增加项目结构、数据格式、文档和后续实现接口，不在没有本地 GPU 验证的情况下
贸然改动核心推理代码。

后续本地接管清单见 [docs/CODEX_HANDOFF.md](docs/CODEX_HANDOFF.md)。
