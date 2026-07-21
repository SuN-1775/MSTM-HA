# MSTM-HA

<p align="center">
  <b>Multi-scale SpatioTemporal Mamba with Hyperbolic Alignment for Video-Text Retrieval</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c">
  <img src="https://img.shields.io/badge/CUDA-12.1-green">
  <img src="https://img.shields.io/badge/Dataset-MSR--VTT-orange">
  <img src="https://img.shields.io/badge/Backbone-CLIP--ViT--B%2F32-purple">
</p>

This is the official code implementation of the paper **“Multi-scale SpatioTemporal Mamba with Hyperbolic Alignment for Video-Text Retrieval”**. More dataset settings and documentation updates will be released soon.

We are continuously organizing the code and documentation. Please stay tuned for the latest updates.

## 🔥 Updates

- [x] Release the MSR-VTT training and evaluation code.
- [x] Release environment and Mamba installation notes.
- [ ] Add more dataset-specific scripts if needed.
- [ ] Improve documentation for additional experimental settings.

## 📁 Repository Structure

```text
MSTM-HA/
├── main_retrieval.py
├── train_msrvtt.sh
├── test_msrvtt.sh
├── requirements.txt
├── MAMBA_ENV_SETUP.md
└── tvr/
    ├── dataloaders/
    ├── models/
    └── utils/
```

## ⚙️ Environment

```bash
conda create -n mstmha python=3.8 -y
conda activate mstmha
```

Install PyTorch first:

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

Then install project dependencies:

```bash
pip install -r requirements.txt
```

MSTE depends on `mamba-ssm` with Bi-Mamba support. See:

```text
MAMBA_ENV_SETUP.md
```

## 🧩 CLIP Pretrained Weights

Download `ViT-B-32.pt` and place it under:

```text
tvr/models/ViT-B-32.pt
```

The CLIP BPE vocabulary file is already included:

```text
tvr/models/bpe_simple_vocab_16e6.txt.gz
```

## 📦 Dataset Preparation

Current scripts are prepared for MSR-VTT.

You may follow the dataset preparation protocol used by existing video-text retrieval repositories such as:

```text
https://github.com/jpthu17/DiCoSA
```

Expected annotation files:

```text
ANNOTATION_PATH/
├── MSRVTT_train.9k.csv
├── MSRVTT_JSFUSION_test.csv
└── MSRVTT_data.json
```

Expected video directory:

```text
YOUR_RAW_VIDEO_PATH/
├── video0.mp4
├── video1.mp4
└── ...
```

Each video file should be named by its `video_id`:

```text
{video_id}.mp4
```

## 🚀 Training

Update paths in `train_msrvtt.sh`:

```bash
--anno_path ANNOTATION_PATH
--video_path YOUR_RAW_VIDEO_PATH
--output_dir YOUR_SAVE_PATH
```

Run:

```bash
bash train_msrvtt.sh
```

Default MSR-VTT setting:

| Item | Value |
|---|---|
| Backbone | CLIP ViT-B/32 |
| Frames | 12 |
| Text length | 32 |
| Epochs | 5 |
| Train batch size | 128 |
| Eval batch size | 64 |

## 🔎 Evaluation

Update paths in `test_msrvtt.sh`:

```bash
--anno_path ANNOTATION_PATH
--video_path YOUR_RAW_VIDEO_PATH
--output_dir YOUR_SAVE_PATH
--init_model YOUR_CKPT_FILE
```

Run:

```bash
bash test_msrvtt.sh
```

## 📝 Notes

- Datasets are not included in this repository.
- Large pretrained files such as `ViT-B-32.pt` are ignored by git.
- Mamba installation can be platform-sensitive; see `MAMBA_ENV_SETUP.md`.

## 📄 License

This project follows the license provided in `LICENSE`.
