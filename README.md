# Math Misconception Ensemble AI

[![Kaggle](https://img.shields.io/badge/Kaggle-Score%200.949-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

> **Competition Result:** 0.949 Score | Top-tier Performance
>
> Advanced multi-model ensemble system for predicting student mathematical misconceptions from open-ended responses.

---

## 📋 Project Overview

This project addresses the **MAP - Charting Student Math Misunderstandings** Kaggle competition, which aims to identify mathematical misconceptions from student explanations. The solution employs a sophisticated 4-model ensemble architecture combining state-of-the-art language models.

### 🎯 Challenge
Predict the affinity between 2,900+ mathematical misconceptions and student open-ended responses across 36,000+ training examples.

### 🏆 Solution
A weighted ensemble of four fine-tuned large language models:
- **Hunyuan-7B** (Weight: 1.2) - Best performer (0.945)
- **DeepSeek-Math-7B** (Weight: 1.0) - Math-specialized model (0.943)
- **Qwen3-8B** (Weight: 1.0) - Robust reasoning (0.943)
- **Gemma2-9B** (Weight: 0.9) - Complementary predictions (0.942)

**Final Ensemble Score:** **0.949** 🚀

---

## 🏗️ Architecture

### Model Pipeline
```
┌─────────────────────────────────────────────────────┐
│  Input: Student Question + Answer + Explanation    │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │  Feature Engineering │
        │  - Correct answer    │
        │  - Format templates  │
        └──────────┬───────────┘
                   │
    ┌──────────────┼──────────────┬──────────────┐
    │              │              │              │
┌───▼───┐    ┌────▼────┐    ┌───▼────┐    ┌────▼────┐
│Hunyuan│    │DeepSeek │    │ Qwen3  │    │ Gemma2  │
│ 0.945 │    │  0.943  │    │ 0.943  │    │  0.942  │
└───┬───┘    └────┬────┘    └───┬────┘    └────┬────┘
    │              │              │              │
    └──────────────┼──────────────┴──────────────┘
                   │
         ┌─────────▼──────────┐
         │ Weighted Ensemble  │
         │ - Probability avg  │
         │ - Agreement voting │
         │ - Confidence boost │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Top-3 Predictions │
         │   Score: 0.949     │
         └────────────────────┘
```

### Ensemble Strategy
1. **Weighted Probability Averaging** (60%): Models vote based on confidence
2. **Agreement Voting** (30%): Bonus for cross-model consensus
3. **Max Confidence Boost** (10%): Amplify high-certainty predictions

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | PyTorch 2.0, Transformers 4.56 |
| **Models** | Hunyuan-7B, DeepSeek-Math-7B, Qwen3-8B, Gemma2-9B |
| **Fine-tuning** | LoRA/QLoRA (PEFT) |
| **Optimization** | Mixed Precision (FP16/BF16) |
| **Hardware** | 2x GPU (Multi-GPU Distribution) |
| **Data** | 36,696 training samples, 65 target classes |

---

## 📊 Key Features

✅ **Multi-Model Fusion**: Combines strengths of 4 diverse LLMs  
✅ **Custom Data Processing**: Correct answer detection and format engineering  
✅ **Memory Optimization**: Sequential loading with aggressive cleanup  
✅ **Robust Ensemble**: Disagreement handling and confidence weighting  
✅ **Production Ready**: Clean, modular, reproducible code  

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch transformers peft datasets scikit-learn scipy tqdm
```

### Data Structure
```
/kaggle/input/
├── map-charting-student-math-misunderstandings/
│   ├── train.csv
│   └── test.csv
├── hunyuan-7b-instruct-bf16/
├── hunyuan-7b-instruct-map/
├── deekseepmath-7b-map-competition/
├── qwen3-8b-map-competition/
└── gemma2-9b-it-cv945/
```

### Run Inference
```python
# Clone and navigate
git clone https://github.com/YOUR_USERNAME/math-misconception-ensemble-ai
cd math-misconception-ensemble-ai

# Run ensemble
python ensemble_hunyuan_deepseek.py
```

---

## 📈 Results

| Model | Individual Score | Ensemble Weight |
|-------|-----------------|----------------|
| Hunyuan-7B | 0.945 | 1.2 |
| DeepSeek-Math-7B | 0.943 | 1.0 |
| Qwen3-8B | 0.943 | 1.0 |
| Gemma2-9B | 0.942 | 0.9 |
| **Final Ensemble** | **0.949** | - |

**Performance Gain:** +0.4% over best single model  
**Competition Rank:** Top-tier performance

---

## 💡 Key Insights

1. **Ensemble > Single Model**: Even small gains matter in competitive ML
2. **Domain Specialization**: Math-focused models (DeepSeek) provide unique value
3. **Agreement Matters**: Cross-model consensus improves reliability
4. **Memory Management**: Critical for multi-GPU large model deployment
5. **Format Engineering**: Input templates significantly impact performance

---

## 📁 Project Structure

```
math-misconception-ensemble-ai/
├── ensemble_hunyuan_deepseek.py  # Main ensemble pipeline
├── README.md                      # This file
├── requirements.txt               # Dependencies
└── notebooks/
    ├── exploratory_analysis.ipynb
    └── model_evaluation.ipynb
```

---

## 🔬 Methodology

### Data Preprocessing
- Identify correct answers per question
- Create boolean "is_correct" feature
- Format inputs with model-specific templates

### Model Training
- Fine-tune base models with LoRA adapters
- Optimize for MAP@3 metric
- Validate on stratified splits

### Inference Pipeline
1. Load models sequentially to manage memory
2. Generate probability distributions (65 classes)
3. Aggregate with weighted voting
4. Select top-3 predictions per sample

---

## 🎓 Educational Impact

This project demonstrates:
- **Educational AI**: Automated misconception detection for personalized learning
- **Scalability**: Processing thousands of student responses efficiently
- **Interpretability**: Understanding common mathematical reasoning errors

---

## 🤝 Acknowledgments

- Kaggle competition organizers
- Hugging Face for transformer models
- Open-source AI community

---

## 📧 Contact

**Muhammad Adel Gabr**  
- GitHub: [@MuhammadAdelGabr]
- Kaggle: [@MuhammadAdelGabr]
- LinkedIn: [[Your LinkedIn](https://www.linkedin.com/in/muhammadadelgabr/)]

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/math-misconception-ensemble-ai?style=social)](https://github.com/YOUR_USERNAME/math-misconception-ensemble-ai)

</div>
