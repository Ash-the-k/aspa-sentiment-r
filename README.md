# Sentiment Analysis on Customer Reviews in R

[![R Version](https://img.shields.io/badge/R-4.5.1-blue.svg)](https://www.r-project.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Made with text2vec](https://img.shields.io/badge/Made%20with-text2vec-lightgrey.svg)](https://cran.r-project.org/web/packages/text2vec/) [![Caret](https://img.shields.io/badge/ML-caret-orange.svg)](https://cran.r-project.org/web/packages/caret/)

---

## Overview

This project performs **sentiment analysis on TripAdvisor hotel reviews** using **Machine Learning and Lexicon-based techniques in R**.

The workflow combines **lexicon-based sentiment extraction** (`syuzhet`) with **supervised ML classification** (`glmnet`) built on a **TF-IDF feature representation** using `text2vec`.

---

## Features

* Text preprocessing and vocabulary construction using **text2vec**
* TF-IDF feature extraction with vocabulary pruning for memory efficiency
* Lexicon-based sentiment scoring using **Syuzhet**, **Bing**, and **Afinn**
* Emotion categorization via **NRC lexicon**
* Supervised model using **Logistic Regression (GLMNET)**
* Model evaluation using **caret** and **pROC** (Accuracy, ROC AUC)
* Reproducible environment management with **renv**

---

## Project Structure

```
aspa-sentiment-r/
│
├── data/
│   └── tripadvisor.csv                # Dataset (20,491 customer reviews)
│
├── scripts/
│   ├── 00_project_setup.R             # Folder paths + setup
│   ├── 01_exploratory_sentiment.R     # Text cleaning + TF-IDF + Lexicon Analysis
│   └── 02_supervised_sentiment_model.R # Supervised ML training + evaluation
│
├── figures/                           # Output plots
│   ├── top_terms_text2vec.png
│   ├── wordcloud_text2vec.png
│   ├── syuzhet_hist.png
│   └── nrc_emotions.png
│
├── models/                            # Saved trained models + vocab/transformer
│   ├── vocabulary.rds
│   ├── tfidf_transformer.rds
│   └── glmnet_cvfit.rds
│
├── reports/                           # CSV reports for results
│   ├── sentiment_sample_full.csv
│   └── test_predictions_glmnet.csv
│
├── renv.lock                          # Environment dependencies (like requirements.txt)
├── .Rprofile
├── .gitignore
└── aspa-sentiment-r.Rproj
```

---

## Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/Ash-the-k/aspa-sentiment-r
cd aspa-sentiment-r
```

2. Open the project in RStudio.

3. Restore dependencies (like `pip install -r requirements.txt` in Python):

```r
install.packages("renv")
renv::restore()
```

4. Run the analysis scripts:

```r
source("scripts/01_exploratory_sentiment.R")
source("scripts/02_supervised_sentiment_model.R")
```

All output visualizations and reports will be generated automatically inside the `figures/` and `reports/` folders.

---

## Model Performance Summary

| Metric          | Value      | Description                           |
| --------------- | ---------- | ------------------------------------- |
| **Accuracy**    | **94.6%**  | Overall classification accuracy       |
| **AUC (ROC)**   | **0.9836** | Excellent discrimination ability      |
| **Sensitivity** | **98.6%**  | Correctly detects positive sentiments |
| **Specificity** | **75.3%**  | Correctly detects negative sentiments |
| **Kappa**       | **0.798**  | Strong agreement beyond chance        |

The model shows **outstanding predictive performance**, demonstrating the power of **TF-IDF + GLMNET** for textual sentiment classification.

---

## Sample Outputs

| Visualization                 | Description                                   |
| ----------------------------- | --------------------------------------------- |
| `top_terms_text2vec.png`      | Top 20 terms by TF-IDF weight                 |
| `wordcloud_text2vec.png`      | Word frequency visualization                  |
| `syuzhet_hist.png`            | Lexicon-based sentiment distribution          |
| `nrc_emotions.png`            | Emotion breakdown (joy, anger, sadness, etc.) |
| `test_predictions_glmnet.csv` | Model predictions for test data               |

---

## Key Libraries Used

| Category               | Packages                               |
| ---------------------- | -------------------------------------- |
| Text Processing        | `text2vec`, `data.table`, `Matrix`     |
| Visualization          | `ggplot2`, `wordcloud`, `RColorBrewer` |
| Lexicon Sentiment      | `syuzhet`                              |
| Machine Learning       | `caret`, `glmnet`, `pROC`              |
| Environment Management | `renv`, `here`                         |

---

## Methodology Summary

1. **Data Cleaning:** Remove noise, punctuation, and stopwords.
2. **TF-IDF Construction:** Use `text2vec` to create a sparse document-term matrix with pruning.
3. **Exploratory Lexicon Analysis:** Compute sentiment polarity and emotions using Syuzhet and NRC lexicons.
4. **Supervised Model Training:**

   * Convert `Rating` into binary classes (Positive ≥4, Negative ≤2).
   * Train a Logistic Regression model using `glmnet` with 5-fold cross-validation.
5. **Evaluation:** Generate confusion matrix, accuracy, ROC-AUC, and save model artifacts.

---

## Results & Discussion

> The trained model achieved **94.6% accuracy** and **0.9836 AUC**, demonstrating robust performance on the sentiment classification task. The analysis highlights the dominance of positive reviews within the dataset and the effectiveness of sparse TF-IDF features combined with regularized logistic regression.
>
> Lexicon-based methods complemented the machine learning model by offering emotional insights into the review corpus, identifying “joy”, “trust”, and “anticipation” as the most frequent emotions among customer feedback.

---

## Conclusion

This project demonstrates an **end-to-end sentiment analysis pipeline** in R, integrating:

* Efficient text preprocessing with **text2vec**
* Emotional insight extraction using **syuzhet**
* Predictive modeling with **GLMNET Logistic Regression**

It serves as a **reproducible template** for future NLP projects in R that require both **descriptive (lexicon)** and **predictive (ML)** sentiment approaches.

---

## License

This project is shared under the **MIT License** — free to use, modify, and distribute with attribution.
