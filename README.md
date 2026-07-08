# ML Model Comparison — News Category Classification

> Comparative analysis of four machine learning algorithms on a synthetic noisy news dataset across five categories.

![All models](https://img.shields.io/badge/models-4-blue) ![Accuracy](https://img.shields.io/badge/accuracy-1.00-brightgreen) ![Categories](https://img.shields.io/badge/categories-5-orange) ![Dataset](https://img.shields.io/badge/dataset-synthetic-yellow)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusion & Recommendations](#conclusion--recommendations)
- [Next Steps](#next-steps)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## Overview

𝗟𝗶𝘃𝗲 𝗔𝗽𝗹: https://news-categori-multiclass-classifier.streamlit.app/

This report presents a comparative analysis of four machine learning classifiers for **news category classification**. The objective is to evaluate algorithm performance on a synthetic, intentionally noisy dataset and identify the most suitable model for real-world deployment — benchmarking against the baseline Multinomial Naive Bayes classifier.

**Key finding:** All four models achieved perfect scores (1.00 accuracy, 1.00 weighted F1) on the test set. While excellent on paper, this outcome warrants careful interpretation — see [Discussion](#discussion).

---

## Dataset

| Property | Detail |
|---|---|
| Source | Synthetic generated dataset |
| Size | 600+ entries (before cleaning) |
| Categories | Sports, Business, Politics, Technology, Entertainment |
| Noise types | HTML tags, inconsistent casing, extra whitespace, special characters, typos, missing values, duplicates |

The dataset was intentionally corrupted to simulate real-world data challenges, making the preprocessing and robustness of each model a key part of the evaluation.

---

## Preprocessing Pipeline

All text data passed through a six-stage cleaning and vectorisation pipeline before model training:

```
Raw Data
   │
   ▼
1. Missing Value Removal     — drop rows with null text or labels
   │
   ▼
2. Category Standardisation  — normalise label casing (e.g. "sports" → "Sports")
   │
   ▼
3. Text Cleaning             — strip HTML tags, lowercase, remove special characters & extra whitespace
   │
   ▼
4. Duplicate Removal         — drop redundant entries
   │
   ▼
5. Stopword Removal          — filter common English stopwords
   │
   ▼
6. TF-IDF Vectorisation      — transform text into numerical features (max 5,000 features)
   │
   ▼
Clean Feature Matrix (X)
```

> **Note:** Capping TF-IDF at 5,000 features balances representational richness with computational efficiency. Raising this limit may marginally improve performance on a real-world corpus.

---

## Models Evaluated

### Multinomial Naive Bayes (MNB)
A probabilistic classifier based on Bayes' theorem with a naive independence assumption. Well-suited to high-dimensional text data and extremely fast to train. Serves as the baseline.

### Logistic Regression (LR)
A linear model that estimates class probabilities via the sigmoid/softmax function. Despite its simplicity, it is one of the strongest baselines for text classification and is highly interpretable.

### Support Vector Machine (SVM)
Finds the optimal separating hyperplane between classes in a high-dimensional feature space. Particularly powerful for text data where TF-IDF produces sparse, high-dimensional vectors.

### Random Forest (RF)
An ensemble method that averages predictions from many decorrelated decision trees. Robust to overfitting and capable of capturing non-linear relationships, though computationally heavier than the linear models.

---

## Results

### Performance Summary

| Model | Accuracy | F1 Score (weighted) | Complexity | Speed |
|---|---|---|---|---|
| Multinomial Naive Bayes | 1.00 | 1.00 | Low | ⚡ Fastest |
| Logistic Regression | 1.00 | 1.00 | Medium | ⚡ Fast |
| Support Vector Machine | 1.00 | 1.00 | Medium | 🔶 Moderate |
| Random Forest | 1.00 | 1.00 | High | 🔴 Slower |

All models achieved perfect classification on the held-out test set (80/20 train/test split, `random_state=42`).

### Evaluation Metrics

**Accuracy** — proportion of correctly classified instances out of all predictions.

**F1 Score (weighted)** — harmonic mean of precision and recall, weighted by class support. More informative than accuracy for imbalanced datasets.

---

## Discussion

### ⚠️ Why perfect scores are a red flag

Achieving 1.00 accuracy on every model is a signal to interrogate the data, not celebrate the result. Three likely causes:

**1. Synthetic data leak**
Synthetically generated datasets — even with introduced noise — tend to produce very strong, category-specific token distributions. After preprocessing strips the noise, the remaining vocabulary may be trivially separable. A real-world article about a politician attending a sports event would challenge every model here.

**2. Aggressive preprocessing amplified signal**
Removing stopwords, HTML, and special characters may have inadvertently amplified domain-specific keywords (e.g. "touchdown", "quarterly earnings", "legislation"). TF-IDF then assigns those terms very high weights, making class boundaries near-perfect.

**3. Small test set**
With ~600 entries and a 20% split, the test set contains roughly 120 samples. Achieving 120/120 correct predictions is far easier than sustaining the same on 10,000+ diverse real-world articles.

### Implications for real-world deployment

Real news data introduces challenges that this benchmark does not capture:

- Articles that span multiple categories (e.g. a tech company's political lobbying)
- Emerging topics with no training precedent
- Sarcasm, irony, and domain jargon
- Class imbalance across categories

---

## Conclusion & Recommendations

Based on the synthetic benchmark, all four models are equally effective. For production use, selection should factor in interpretability, inference speed, and generalisation to real data:

| Recommendation | Model | Reason |
|---|---|---|
| 🥇 Best production baseline | **Logistic Regression** | Interpretable, fast, strong generalisation on text |
| 🥈 Best for noisy real data | **SVM** | Maximises margin in high-dimensional TF-IDF space |
| 🥉 Best for complex patterns | **Random Forest** | Captures non-linear relationships; slower but robust |
| 🔬 Future work | **BERT / DistilBERT** | Contextual embeddings handle ambiguous cross-category content |

---

## Next Steps

To validate these results and build a production-ready classifier:

1. **Use a real-world corpus** — evaluate on AG News, BBC News, or Reuters-21578
2. **Cross-validate** — replace the single 80/20 split with k-fold cross-validation (k=5 or k=10)
3. **Hyperparameter tuning** — grid search over regularisation strength (LR/SVM) and tree depth (RF)
4. **Adversarial testing** — evaluate on articles that genuinely span multiple categories
5. **Deep learning baseline** — fine-tune DistilBERT for a contextual embedding comparison
6. **Error analysis** — inspect confusion matrices and misclassified samples for systematic failure modes

---

## Project Structure

```
news-classification/
├── data/
│   └── news_dataset.csv              ← raw synthetic dataset
├── notebooks/
│   └── model_comparison.ipynb        ← EDA, training, evaluation
├── models/
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   └── random_forest.pkl
├── results/
│   └── model_comparison_plot.png     ← performance bar chart
├── src/
│   ├── preprocess.py                 ← cleaning & TF-IDF pipeline
│   ├── train.py                      ← model training script
│   └── evaluate.py                   ← metrics & visualisation
├── requirements.txt
└── README.md
```

---

## Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
nltk>=3.8.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Download NLTK stopwords (first run only):

```python
import nltk
nltk.download('stopwords')
```

---

## Running the Notebook

```bash
# Clone the repository
git clone https://github.com/zakir-maswani/News-Categori-MultiClass-Classifier.git
cd news-classification

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/model_comparison.ipynb
