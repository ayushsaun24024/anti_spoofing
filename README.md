# 🔍 ASVspoof 2024 Research Pipeline — Spoofing Detection Framework

Welcome to the **ASVspoof 2024 Research Pipeline**, a modular, reproducible, and extensible pipeline designed for **anti-spoofing research** in speaker verification systems.

> ⚠️ **Disclaimer**: This work is an **independent research effort** conducted *after* the challenge concluded.

---

## 🧠 Overview

This repository contains a full pipeline for experimenting with anti-spoofing techniques, structured around the **ASVspoof 2024 (ASVspoof 5)** dataset. The goal is to explore various paradigms for detecting spoofed speech using both traditional and neural approaches.

> ✅ Current best Equal Error Rate (EER): **6.3%**  
> 🚧 Latest code updates are **under research** and will be **released in future commits**.

---

## 📁 Directory Structure

```bash
ASVspoof2024-Pipeline/
├── architecture/      # All model definitions (CNNs, RNNs, Transformers, etc.)
├── dataset/           # Dataset loading, preprocessing, feature extraction
├── metrics/           # Evaluation metrics such as EER, t-DCF
├── paradigms/         # Training strategies, loss functions, learning paradigms
├── utilities/         # Utilities like logger, argument parsers, helper scripts
│   └── requirements.txt  # All required Python dependencies
└── yaml/              # YAML config files to manage experiments
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ayushsaun24024/anti_spoofing.git
   ```

2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r utilities/requirements.txt
   ```

---

## 🔧 How to Use

### 🛠 Configuration

All experiment configurations are stored in the `yaml/` folder. You can create or modify YAML files to control:

- Model architecture
- Dataset path and preprocessing
- Training hyperparameters
- Evaluation settings

Example:
```yaml
experiment_name: cnn_baseline
model: CNN
dataset_path: /path/to/asvspoof2024
learning_rate: 0.001
epochs: 50
```

### 🚀 Training

You can launch training using your configured YAML file:

```bash
python paradigms.train --config=yaml/cnn_baseline.yaml --exp_name=Training
```

### 📊 Evaluation

To evaluate a trained model:

```bash
python paradigms.train --config=yaml/cnn_baseline.yaml --exp_name=Testing
```

---
## 🔬 Under Research

We are actively experimenting with:

- **Transformer-based architectures**
- **Contrastive and self-supervised learning**
- **Advanced augmentation techniques**
- **Score fusion and calibration**

These components will be **released progressively** as the research matures.

---

## 📌 Important Notes

- This repo is **for research purposes only**.
- It is **not affiliated with** ASVspoof organizers or the challenge committee.
- If using the code or methodology in academic work, please cite the original ASVspoof 2024 paper(s), not this repository.

---

## 💡 Contributing

Contributions, suggestions, and feedback are welcome! Feel free to fork this repository or open an issue for bugs or feature requests.

---

## 📜 License

This repository is released under the **MIT License**. See `LICENSE` file for details.

---

## 🤝 Acknowledgments

- ASVspoof 2024 Challenge Team for the dataset and protocol.
- The open-source audio and machine learning communities.

---

📫 **Contact**: You can reach out via GitHub Issues if you encounter any problems or have questions.

