# Reproducing and Extending Vision Transformers (ViT) for Image Classification

This project reproduces the Vision Transformer (ViT) architecture and benchmarks it against classical CNNs on CIFAR-10. Extensions and custom experiments are encouraged!

## 📚 Project Goals

- Implement ViT (with [timm](https://github.com/huggingface/pytorch-image-models) or from scratch)
- Train and evaluate on CIFAR-10
- Compare with ResNet
- Visualize results and attention maps
- Try extensions: data augmentation, transfer learning, or different datasets

## 📦 Structure

```
.
├── src/
│   ├── train_vit.py           # Training script for ViT
│   ├── train_cnn.py           # Training script for CNN baseline
│   └── dataset.py             # Data handling utilities
├── notebooks/
│   └── analysis.ipynb         # For visualization, EDA, and results
├── data/                      # Datasets (downloaded/processed)
├── results/                   # Saved models, plots, metrics
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Vision Transformer**
   ```bash
   python src/train_vit.py
   ```

3. **Train CNN Baseline**
   ```bash
   python src/train_cnn.py
   ```

4. **Analyze Results**
   - Open `notebooks/analysis.ipynb`

## 📊 Results

- Compare accuracy, loss, and other metrics in `results/`
- Visualize attention maps in the notebook

## 📄 References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

---
