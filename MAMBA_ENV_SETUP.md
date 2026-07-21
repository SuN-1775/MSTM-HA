# Mamba 实验环境构建说明

本文说明本实验（MSTMHA + MSTE 多尺度 Mamba）所用 conda 环境的构建过程。  
日常训练/评测使用的环境名为 **`mstmha`**（不是名为 `mamba` 的那个独立环境）。

---

## 1. 环境概览

| 项目 | 值 |
|------|-----|
| conda 环境名 | `mstmha` |
| Python | 3.8 |
| PyTorch | 2.1.0 + CUDA 12.1 |
| mamba-ssm | 1.1.1（Vision Mamba 的 `mamba-1p1p1`，可编辑安装） |
| causal-conv1d | 1.1.0 |
| flash-attn | 2.5.8（可选，加速用） |
| 核心额外包 | `einops` |

MSTE 分支依赖：

```python
from mamba_ssm import Mamba
from einops import rearrange
```

并使用 `bimamba_type='v2'`（双向 Mamba），因此需要 **Vision Mamba 提供的 `mamba-1p1p1`**，而不是较新的官方 `mamba-ssm` 2.x。

---

## 2. 推荐构建步骤（复现 `mstmha`）

### Step 0. 准备源码与 wheel（本机路径示例）

```text
/home/mkyvkbwh/sun/vision_mamba/          # Vim / mamba-1p1p1 / causal-conv1d 源码
/home/mkyvkbwh/sun/causal_conv1d-1.1.0+cu122torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
/home/mkyvkbwh/sun/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

> wheel 与 **Python 3.8 + Torch 2.1** 绑定；换 Python/Torch 版本需重新编译或换对应 wheel。

### Step 1. 创建 conda 环境

```bash
conda create -n mstmha python=3.8 -y
conda activate mstmha
```

### Step 2. 安装 PyTorch（CUDA 12.1）

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 期望类似: 2.1.0+cu121 12.1 True
```

### Step 3. 安装 MSTMHA 核心依赖

```bash
cd /path/to/MSTMHA-MSTE-3
pip install -r requirements.txt
```

### Step 4. 安装编译工具链依赖

```bash
pip install packaging ninja
```

系统侧通常还需要可用的 CUDA toolkit / `nvcc`（与驱动匹配），用于编译 CUDA 扩展。

### Step 5. 安装 `causal-conv1d`

**方式 A（推荐，本机已有预编译 wheel）：**

```bash
pip install /home/mkyvkbwh/sun/causal_conv1d-1.1.0+cu122torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

**方式 B（从 Vision Mamba 源码可编辑安装）：**

```bash
cd /home/mkyvkbwh/sun/vision_mamba
pip install -e "causal-conv1d>=1.1.0"
# 或:
# pip install -e causal-conv1d --no-build-isolation
```

### Step 6. 安装 `mamba-ssm`（关键：1.1.1 + BiMamba v2）

```bash
cd /home/mkyvkbwh/sun/vision_mamba
pip install -e mamba-1p1p1 --no-build-isolation
```

安装成功后 `pip show mamba_ssm` 应类似：

```text
Name: mamba_ssm
Version: 1.1.1
Editable project location: .../vision_mamba/mamba-1p1p1
```

### Step 7.（可选）安装 FlashAttention

```bash
pip install /home/mkyvkbwh/sun/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

### Step 8. 验证

```bash
conda activate mstmha
python - <<'PY'
import torch
from mamba_ssm import Mamba
from einops import rearrange
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
x = torch.randn(2, 64, 512, device="cuda")
y = Mamba(d_model=512, d_conv=4, bimamba_type="v2", use_fast_path=True, expand=1).cuda()(x)
print("Mamba OK:", y.shape)
PY
```

项目内快速检查：

```bash
cd /path/to/MSTMHA-MSTE-3
python -c "from tvr.models.modeling import Mamba_head, LayerNorm_conv; print('MSTE modules OK')"
```

---

## 3. 一键命令汇总

```bash
conda create -n mstmha python=3.8 -y
conda activate mstmha

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install packaging ninja

pip install /home/mkyvkbwh/sun/causal_conv1d-1.1.0+cu122torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install -e /home/mkyvkbwh/sun/vision_mamba/mamba-1p1p1 --no-build-isolation
# 可选:
# pip install /home/mkyvkbwh/sun/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

训练时：

```bash
conda activate mstmha
bash train_msrvtt.sh
```

---

## 4. 常见问题

### 4.1 为什么不用 `pip install mamba-ssm` 最新版？

本实验的 `Mamba_head` 使用了 **`bimamba_type='v2'`**（双向 Mamba），该接口来自 Vision Mamba 维护的 `mamba-1p1p1`（1.1.1）。  
官方较新的 `mamba-ssm` 2.x API 不同，直接替换可能导致导入或参数报错。

### 4.2 编译失败 / `undefined symbol` / CUDA 不匹配

- 确认 `torch.version.cuda` 与本机驱动、编译时 CUDA 一致  
- 优先使用与 `cp38` + `torch2.1` 匹配的预编译 wheel  
- 源码安装时加 `--no-build-isolation`，并保证 `nvcc` 可用  
- 换机复现时，Python / Torch / CUDA 三者需一起对齐

### 4.3 机器上另有名为 `mamba` 的 conda 环境

本机还存在环境 **`mamba`**（Python 3.10，`mamba-ssm==2.2.4`），用于其它用途。  
**MSTMHA-MSTE 训练请使用 `mstmha`**，不要与之混淆。

---

## 5. 依赖关系简图

```text
conda env: mstmha (py3.8)
 ├── torch 2.1.0+cu121
 ├── MSTMHA 核心包 (requirements.txt)
 │     numpy / pandas / decord / opencv / ftfy / timm / ...
 └── MSTE 扩展
       ├── einops
       ├── causal-conv1d 1.1.0
       └── mamba-ssm 1.1.1  ← editable: vision_mamba/mamba-1p1p1
             └── 支持 bimamba_type='v2'
```
