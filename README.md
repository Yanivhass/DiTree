# DiTree

[project page](https://sites.google.com/view/ditree/home) [paper](https://arxiv.org/abs/2508.21001)

Official implementation of the paper "Train Once Plan Anywhere Kinodynamic Motion Planning via Diffusion Trees" [CoRL 25]

This repository contains the code and experiment setup for the **DiTree** project. It includes training configuration, experiment execution scripts, and pretrained model checkpoints.

The associated car dataset and model weights are available for download [here](https://drive.google.com/drive/folders/1WiBU2g1qQn_2j6v1ZTB1eU0dAyCGoX7F?usp=sharing).

---

## 📦 Installation

We recommend creating a Python virtual environment before installing dependencies.

1. **Clone the repository**

```bash
git clone https://github.com/Yanivhass/ditree.git
cd ditree
```

2. **Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3. **Install PyTorch**
   Visit [PyTorch.org](https://pytorch.org/get-started/locally/) to find the correct installation command for your system (CPU or GPU).
   Example (CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. **Install other dependencies**

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
.
├── run_scenarios.py      # Runs the experiments
├── train_manager.py      # Configures and launches training sessions
├── checkpoints/          # Contains pretrained model weights
├── data/                 # (optional) Local dataset storage
└── requirements.txt      # Python dependencies
```

---

## 🚀 Usage

### Running Experiments

```bash
python run_scenarios.py
```

### Training

```bash
python train_manager.py
```

---

## 📥 Pretrained Models & Dataset

Pretrained model weights and the car dataset can be downloaded from:
[Google Drive Link](https://drive.google.com/drive/folders/1WiBU2g1qQn_2j6v1ZTB1eU0dAyCGoX7F?usp=sharing)
AntMaze dataset is available using Minari  [Minari]([https://pytorch.org/get-started/locally/](https://minari.farama.org/index.html))
Place the downloaded files into:

```
checkpoints/
data/
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

If you want, I can also make you a **requirements.txt** template for typical PyTorch projects so users can `pip install -r requirements.txt` without guesswork. That way, the README setup will be one command shorter. Would you like me to prepare that?

