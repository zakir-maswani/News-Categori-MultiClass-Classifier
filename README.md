# Machine Learning Model Comparison for News Category Classification

## Introduction
This report presents a comparative analysis of various machine learning models for the task of news category classification. The objective is to evaluate the performance of different algorithms on a synthetic, uncleaned news dataset and identify potentially superior alternatives to the initial Multinomial Naive Bayes classifier.

## Methodology

### Dataset
The analysis utilizes news dataset comprising over 600 entries across five categories: Sports, Business, Politics, Technology, and Entertainment. The dataset was intentionally generated with various forms of noise, including HTML tags, inconsistent casing, extra whitespace, special characters, typos, missing values, and duplicate entries, to simulate real-world data challenges.

### Data Preprocessing
Before model training, the dataset underwent a cleaning and preprocessing pipeline:
1.  **Missing Value Handling**: Rows with missing text or category labels were removed.
2.  **Category Standardization**: Category labels were standardized to a consistent capitalization (e.g., "Sports").
3.  **Text Cleaning**: HTML tags, extra whitespace, and special characters were removed. Text was converted to lowercase.
4.  **Duplicate Removal**: Redundant entries were identified and removed.
5.  **Stopword Removal**: Common English stopwords were removed from the text.
6.  **Vectorization**: Text data was transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, limiting features to 5000 for efficiency.

### Models Evaluated
The following machine learning models were trained and evaluated:
*   **Multinomial Naive Bayes (MNB)**: A probabilistic classifier suitable for text classification.
*   **Logistic Regression (LR)**: A linear model for binary and multiclass classification.
*   **Support Vector Machine (SVM)**: A powerful model that finds an optimal hyperplane to separate classes.
*   **Random Forest (RF)**: An ensemble learning method that constructs multiple decision trees.

### Evaluation Metrics
Model performance was assessed using:
*   **Accuracy**: The proportion of correctly classified instances.
*   **F1-Score (Weighted)**: The harmonic mean of precision and recall, weighted by the support for each class, providing a balanced measure of a model's accuracy.

## Results

### Model Performance Summary
The table below summarizes the performance of each model based on Accuracy and F1-Score.

| Model                     | Accuracy | F1-Score |
|:--------------------------|:---------|:---------|
| Multinomial Naive Bayes   | 1.00     | 1.00     |
| Logistic Regression       | 1.00     | 1.00     |
| Support Vector Machine    | 1.00     | 1.00     |
| Random Forest             | 1.00     | 1.00     |

### Visual Comparison

![Model Performance Comparison](https://private-us-east-1.manuscdn.com/sessionFile/At0L9c20hwnqEgz7gSAHJM/sandbox/NklwSQhPul2xuhNjRII8J7-images_1781763555900_na1fn_L2hvbWUvdWJ1bnR1L21vZGVsX2NvbXBhcmlzb24.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvQXQwTDljMjBod25xRWd6N2dTQUhKTS9zYW5kYm94L05rbHdTUWhQdWwyeHVoTmpSSUk4SjctaW1hZ2VzXzE3ODE3NjM1NTU5MDBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyMXZaR1ZzWDJOdmJYQmhjbWx6YjI0LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=A5c~y2aD0Dd2s~X7lw9OHN-xy-8epM1NCbLUdDQAQ5lOb-wQCO1AZg2XqFWXJhkJWyAZMU8oZkGppV9GmkdCiuOjlyAylXnI9OTCGv2U0~QvE8A2ap-qJGo31qiTP2jr0taW1tFzfEluE~VugrTQ423FMV6a8E~eolSjqTZGFt3r4C4q4SkInCe2lcpHBqW7FUfZk-U7YiAn9y-0FGgKUUxHOQ3JldDu0NQwWF4QNUfsI-tXvmZ-GYaXJLynu-RM59viXiCNpDtA1oXXE0B7TJ7b3aMgIgD-CIz5JFdMvB6p8ZPclIb~I-C6R9INHrVh1do7STCcRXbAONXBbmK-Nw__)

## Discussion
Surprisingly, all evaluated models (Multinomial Naive Bayes, Logistic Regression, Support Vector Machine, and Random Forest) achieved perfect scores (1.00 Accuracy and 1.00 F1-Score) on the test set. This indicates that the synthetic dataset, even with introduced noise, might be too easily separable for these models after the applied cleaning and preprocessing steps. It is possible that the patterns distinguishing the categories are very strong and distinct, leading to straightforward classification.

While these results are excellent, it is important to consider that such perfect scores on a synthetic dataset might not directly translate to real-world performance. Real-world news data often presents more subtle distinctions, higher variability, and more complex noise patterns that could challenge these models further.

## Conclusion
Based on the current synthetic dataset, all tested models performed exceptionally well, achieving perfect classification. For practical applications with real-world data, further experimentation with more complex datasets, hyperparameter tuning, and potentially more advanced models (e.g., deep learning approaches) would be beneficial to assess robustness and generalization capabilities. However, for this specific synthetic dataset, all models are equally effective.
