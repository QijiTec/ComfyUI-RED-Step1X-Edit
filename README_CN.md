# ComfyUI_Step1X-Edit

[English](README.md) | [中文文档](README_CN.md)

此自定义节点将 [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit) 图像编辑模型集成到 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 中。Step1X-Edit 是一个先进的图像编辑模型，它接收参考图像和用户的编辑指令，生成新的图像。

## 功能特点

- [x] 支持 FP8 推理
- [x] 支持 flash-attn 加速
- [x] 支持 TeaCache 加速，实现2倍速推理且质量损失极小

## 示例展示

以下是使用 ComfyUI_Step1X-Edit 可以实现的效果示例：

<table>
  <tr>
    <th colspan="2" style="text-align: center">示例 1: "给这个女生的脖子上戴一个带有红宝石的吊坠。"</th>
  </tr>
  <tr>
    <th style="text-align: center">原生版本</th>
    <th style="text-align: center">～1.5倍加速版本（threshold=0.25）</th>
  </tr>
  <tr>
    <td><img src="examples/0000.jpg" alt="Example Image1"></td>
    <td><img src="examples/0000_fast_0.25.jpg" alt="Example Image2"></td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="2" style="text-align: center">示例 2: "让她哭。"</th>
  </tr>
  <tr>
    <th style="text-align: center">原生版本</th>
    <th style="text-align: center">～1.5倍加速版本（threshold=0.25）</th>
  </tr>
  <tr>
    <td><img src="examples/0001.jpg" alt="Example Image1"></td>
    <td><img src="examples/0001_fast_0.25.jpg" alt="Example Image2"></td>
  </tr>
</table>

您可以在[示例目录](examples/step1x_edit_example.json)中找到示例工作流。直接将其加载到 ComfyUI 中即可查看其工作原理。

## 安装方法

1.  **将此仓库克隆到 ComfyUI 的 `custom_nodes` 目录中：**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/raykindle/ComfyUI_Step1X-Edit.git
    ```

2.  **安装所需依赖：**

    *步骤 1: 安装 ComfyUI_Step1X-Edit的依赖*
    ```bash
    cd ComfyUI_Step1X-Edit
    pip install -r requirements.txt
    ```

    *步骤 2: 安装 [`flash-attn`](https://github.com/Dao-AILab/flash-attention)，我们提供了一个脚本，帮助您找到适合您系统的预编译轮子。*
    ```bash
    python utils/get_flash_attn.py
    ```
    该脚本将生成一个轮子名称，例如：`flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`，该名称可在 [flash-attn 的发布页面](https://github.com/Dao-AILab/flash-attention/releases) 中找到。

    然后，您可以下载相应的预编译轮子，并按照 [`flash-attn`](https://github.com/Dao-AILab/flash-attention) 中的说明进行安装。

3.  **下载 Step1X-Edit-FP8 模型**
    ```
    ComfyUI/
    └── models/
        ├── diffusion_models/
        │   └── step1x-edit-i1258-FP8.safetensors
        ├── vae/
        │   └── vae.safetensors
        └── text_encoders/
            └── Qwen2.5-VL-7B-Instruct/
    ```
    - Step1X-Edit 扩散模型：从 [HuggingFace](https://huggingface.co/meimeilook/Step1X-Edit-FP8/tree/main) 下载 `step1x-edit-i1258-FP8.safetensors` 并放置在 ComfyUI 的 `models/diffusion_models` 目录中
    - Step1X-Edit VAE：从 [HuggingFace](https://huggingface.co/meimeilook/Step1X-Edit-FP8/tree/main) 下载 `vae.safetensors` 并放置在 ComfyUI 的 `models/vae` 目录中
    - Qwen2.5-VL 模型：下载 [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main) 并放置在 ComfyUI 的 `models/text_encoders/Qwen2.5-VL-7B-Instruct` 目录中


## 使用方法

1. 启动 ComfyUI 并创建新的工作流。
2. 添加 "Step1X-Edit Model Loader" 节点（或更快的 "Step1X-Edit TeaCache Model Loader" 节点，获得2倍速度提升）到工作流中。
3. 配置模型参数：
    - 选择 `step1x-edit-i1258-FP8.safetensors` 作为扩散模型
    - 选择 `vae.safetensors` 作为 VAE
    - 设置 `Qwen2.5-VL-7B-Instruct` 作为文本编码器
    - 根据需要设置其他参数（`dtype`、`quantized`、`offload`）
    - 如果使用TeaCache，设置合适的阈值
4. 连接 "Step1X-Edit Generate" 节点（或使用TeaCache时连接 "Step1X-Edit TeaCache Generate" 节点）到模型节点。
5. 提供输入图像和编辑提示。
6. 运行工作流生成编辑后的图像。

## 参数说明

### Step1X-Edit Model Loader（Step1X-Edit 模型加载器）

- `diffusion_model`：Step1X-Edit 扩散模型文件（从 diffusion_models 下拉菜单中选择）
- `vae`：Step1X-Edit VAE 文件（从 vae 下拉菜单中选择）
- `text_encoder`：Qwen2.5-VL 模型目录名称（例如 "Qwen2.5-VL-7B-Instruct"）
- `dtype`：模型精度（bfloat16、float16 或 float32）
- `quantized`：是否使用 FP8 量化权重（推荐 开启）
- `offload`：在不使用时是否将模型卸载到 CPU

### Step1X-Edit TeaCache Model Loader（Step1X-Edit TeaCache 模型加载器）（附加参数）

- `teacache_threshold`：控制速度和质量之间的平衡
  - `0.25`：约1.5倍速度提升
  - `0.4`：约1.8倍速度提升
  - `0.6`：2倍速度提升（推荐）
  - `0.8`：约2.25倍速度提升，质量损失极小
- `verbose`：是否打印TeaCache调试信息

### Step1X-Edit Generate / Step1X-Edit TeaCache Generate（Step1X-Edit 生成 / Step1X-Edit TeaCache 生成）

- `model`：Step1X-Edit 模型包
- `input_image`：要编辑的输入图像
- `prompt`：描述所需编辑的文本指令
- `negative_prompt`：描述要避免的内容的文本
- `num_steps`：去噪步数（更多步数 = 更好的质量但更慢）
- `cfg_guidance`：引导系数（控制对提示的遵循程度）
- `size_level`：输出图像大小（推荐 512）
- `seed`：随机种子，用于可重现性

## TeaCache 加速

本实现包含TeaCache加速技术，提供：

- 无质量损失的2倍推理速度
- 无需额外模型微调的免训练加速
- 基于时间步嵌入的自适应缓存
- 通过阈值参数可调节的速度-质量平衡

TeaCache通过智能跳过去噪过程中的冗余计算工作。它分析步骤之间的相对变化并在可能的情况下重用先前计算的结果，显著减少计算需求而不影响输出质量。

基于[TeaCache](https://github.com/LiewFeng/TeaCache)研究，该技术最初是为加速视频扩散模型而开发的，此处已适配用于图像生成。

## 内存需求

Step1X-Edit 模型需要相当大的 GPU 内存：(768 px, 10 步)

|     模型版本   |     峰值 GPU 内存 | 原生版本 | ～1.5倍加速版本（threshold=0.25） | ～2.0倍加速版本（threshold=0.6） |
|:------------:|:------------:|:------------:|:------------:|:------------:|
| Step1X-Edit-FP8(offload=False)   |       31.5GB     | 17.4s | 11.2s | 7.8s |

* 该模型在一张 H20 GPU 上测试。

为了降低显存使用，请在模型加载器节点中启用 `quantized` 和/或 `offload` 选项。

## 故障排除

- 如果遇到 CUDA 内存不足错误，请尝试：
  - 启用 `offload` 选项
  - 使用 FP8 量化模型
  - 减小图像大小
  - 关闭其他占用 GPU 的应用程序
- 如果遇到文件缺失错误：
  - 确保模型路径正确
  - 扩散模型应位于 `models/diffusion_models`
  - VAE 应位于 `models/vae`
  - 文本编码器应位于 `models/text_encoders/Qwen2.5-VL-7B-Instruct`
- 如果遇到导入错误，请确保所有依赖项都正确安装
- 如果遇到TeaCache相关的异常行为：
  - 尝试不同的阈值设置
  - 启用详细输出模式查看哪些步骤被缓存
  - 确认TeaCache模型加载器正确连接到TeaCache生成节点

## 致谢

- 感谢 [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit) 提供原始模型
- 感谢 [TeaCache](https://github.com/LiewFeng/TeaCache) 提供加速技术
- 感谢 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 提供可扩展的 UI 框架