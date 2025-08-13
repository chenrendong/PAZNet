# PAZNet

Patient-specific **ablation zone (AZ) prediction** for hepatic microwave ablation (HMWA).  
This repository provides a **simple, fast, and flexible** baseline for training and inference.


## 📦 Environment

- Python >= 3.9
- (Optional) CUDA-enabled GPU for training

```bash
# (Optional) Create and activate a virtual environment
# python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```
> For **PyTorch**, please install the version matching your CUDA from the official website if `pip install torch` doesn't pick the correct build automatically.

---

## 📁 Data Layout

Both training and inference assume a **paired directory layout** with matching filenames across modalities:

```
<DATA_ROOT>/
├─ pre_norm/    # MRI volumes (e.g., .nii ), names like pre_norm-0.nii
├─ tem_norm/    # Temperature volumes (e.g., .nii.)
└─ mask/        # Binary segmentation masks (optional for inference; used in training/metrics)
```

**Examples** (adapt paths for your machine):

```
D:/MyDeepL/traindata_80/
  ├─ pre_norm/
  ├─ tem_norm/
  └─ mask/

D:/MyDeepL/testdata_80/
  ├─ pre_norm/
  ├─ tem_norm/
  └─ mask/
```

---

## 🚀 Training

Use your `train.py` (minimal trainer). Example parameters (Windows paths):


> **Note:** The `resnet_10.pth` pre-trained weights are downloaded from: ([Google Drive](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view?usp=sharing))
```python
para = {
    "training_set_path": "./traindata_80/",
    "batch_size": 8,
    "num_workers": 4,
    "pin_memory": True,
    "cudnn_benchmark": True,
    "epoch": 201,
    # MedicalNet ResNet‑10 pretrain (conv-only will be loaded inside paznet.generate_model)
    "pretrain_path": "./pretrain/resnet_10.pth"
}
```

Then inside `train.py`:

```python
from paznet import generate_model

model, parameters = generate_model(
    training=True,
    no_cuda=False,
    gpu_id=[0],
    phase="train",
    pretrain_path=para["pretrain_path"]
)
```

Typical CLI run (if your trainer supports args):
```bash
python train.py   --data-root D:/MyDeepL/traindata_80   --epochs 200   --batch-size 8   --device cuda:0  
```

## 🔎 Inference

We include a tiny example script: `examples/infer_example.py`.

**Run (example):**
```bash
python examples/infer_example.py   --mri-dir D:/MyDeepL/testdata_80/pre_norm   --tem-dir D:/MyDeepL/testdata_80/tem_norm   --save-dir D:/MyDeepL/PAZNet_Github/test   --weights ./module/epoch200.pth   --device cuda:0
```

- The script tries to load `.nii/.nii.gz` volumes.  
- Predictions will be saved as `.nii.gz` when `nibabel` is available; otherwise as `.npy` files.  
- If you also provide `--mask-dir`, it will compute a simple Dice for sanity-check (`calculate_metrics.py` is used in your main pipeline; this example includes a lightweight dice).

---



---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---


## 🗂️ Suggested Repo Structure

```
PAZNet/
├─ paznet.py
├─ train.py
├─ Dice_Bce_Loss.py
├─ calculate_metrics.py
├─ transforms.py
├─ examples/
│  └─ infer_example.py
├─ requirements.txt
├─ README.md
├─ LICENSE
└─ .gitignore
```
