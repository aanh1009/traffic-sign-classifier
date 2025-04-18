# Traffic‑Sign Classifier

A PyTorch CNN that recognises 43 German road‑sign classes and a **Streamlit** front‑end to test images in your browser.

> **Try the live demo →** https://trafficsigndetector.streamlit.app/

---

## ✨ Key features

| | |
|---|---|
| **State‑of‑the‑art dataset** | Trained on the **GTSRB** benchmark (German Traffic Sign Recognition) – 50 k+ images across 43 classes. |
| **Light CNN** | 3 × Conv‑BN‑ReLU blocks → MaxPool → Dropout 0.25 → 2 × FC; ~840 k parameters. |
| **92 % test accuracy** | Achieved after 24 epochs (10 % random subset of training data). |
| **Optimised TorchScript model** | `model_scripted.pt` loads fast and runs on CPU. |
| **Streamlit UI** | Upload a JPEG/PNG and get the predicted class & probability in <1 s. |
| **Colab notebook friendly** | All training code in `model.py`, runnable on GPU. |

---

## 📊 Model architecture

```text
Input 3×128×128
 ├─ Conv2d 3→32, k3 • BN • ReLU
 ├─ MaxPool 2×2
 ├─ Conv2d 32→64, k3 • BN • ReLU
 ├─ MaxPool 2×2
 ├─ Conv2d 64→128, k3 • BN • ReLU
 ├─ MaxPool 2×2  → 128×16×16
 ├─ Dropout p=0.25
 ├─ Flatten → 32768
 ├─ Linear 32768→512 • ReLU
 └─ Linear 512→43 ─► Softmax
```

---

## 🚀 Quick start

### 1 · Clone & install

```bash
git clone https://github.com/aanh1009/traffic-sign-classifier.git
cd traffic-sign-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch, torchvision, streamlit, numpy, pillow …
```

### 2 · Run Streamlit app

```bash
streamlit run app.py
```

Open <http://localhost:8501>, upload an image, and see the predicted sign.

### 3 · Train from scratch (optional)

```bash
python model.py --epochs 24 --batch 64 --lr 5e-4 \
               --data_root /path/to/GTSRB
```

A scripted model will be saved to `model_scripted.pt` along with `saved_steps.pkl` (label map).

---

## 🗂 Project layout

```
.
├─ app.py              # Streamlit front‑end
├─ classify.py         # CLI inference helper
├─ model.py            # CNN definition + train loop
├─ model_scripted.pt   # Ready‑to‑use TorchScript model
├─ saved_steps.pkl     # Label‑id ↔ class‑name dict
├─ requirements.txt    # pip deps
└─ packages.txt        # apt packages for Streamlit sharing (optional)
```

---

## 🔒 License

MIT © 2025 Tuan Anh Ngo

---

## 🙏 Acknowledgements

* German Traffic Sign Recognition Benchmark (IJCNN 2011) ­– <http://benchmark.ini.rub.de>.
* PyTorch & TorchVision teams.
* Streamlit community for easy sharing.

