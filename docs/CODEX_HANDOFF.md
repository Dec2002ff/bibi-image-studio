# Codex Local Handoff

当 GitHub 阶段完成后，Codex 在用户 Windows 电脑上按下面顺序接管。

## A. 只读检查

先不要安装任何东西：

1. Clone `Dec2002ff/bibi-image-studio`。
2. Checkout `bibi-foundation`。
3. 收集：
   - Windows 版本
   - GPU 型号与 VRAM
   - NVIDIA driver
   - 系统 RAM
   - 可用磁盘
   - Python / Git 状态
4. 运行仓库现有静态测试中不需要模型的部分。

## B. 选择权重方案

优先使用上游 `WeightsPlan`，不要手写第二套显存判断。

参考上游当前方案：

- >=20 GB VRAM：bf16 / high
- 12–20 GB：INT8 / low
- 10–12 GB：GGUF Q8_0 / low
- 8 GB：GGUF Q4_K_M / low
- 6 GB：仅作为实验，使用更低 GGUF

实际选择必须以检测到的硬件和当时上游文档为准。

## C. 安装

Windows 优先调用仓库现有：

```powershell
.\install.ps1
```

不要手动把依赖装进系统 Python。

随后：

```powershell
.venv\Scripts\python -m fooocus_qwen --selftest
```

如果失败，先修环境，不进入 UI 重构。

## D. 最小生成验收

至少保存以下结果和日志：

1. 文生图：1:1，一张，固定 seed。
2. 9:16 文生图。
3. 单参考编辑。
4. Mask 局部编辑。
5. Outpainting。
6. 两张以上 reference 的身份/风格保持测试。

记录耗时、峰值 VRAM、RAM 与所用 precision/profile。

## E. 验收后才开始 Bibi 功能

顺序：

1. i18n 重构并加入 zh-CN。
2. Style Profile storage + UI。
3. Character Profile storage + UI。
4. Variation / Remix。
5. LoRA manager。
6. 可选 ComfyUI backend adapter。

每一步都应保持现有生成、编辑、画廊 smoke tests 通过。
