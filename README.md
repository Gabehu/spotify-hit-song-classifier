# spotify-hit-predictor

This notebook explores what characteristics differentiate hit songs from non-hit tracks using a large Spotify dataset. After cleaning and engineering features, we trained and evaluated two classification models—**Logistic Regression** and **Random Forest**—on a highly imbalanced dataset, rebalanced using **SMOTE**. We then interpreted key contributing features using model-based importance and SHAP values.

---

## 📊 Dataset

**Source:** [Spotify Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data)

Includes 100k+ songs with fields including:

- `popularity` (0-100 score)
- audio features (`danceability`, `energy`, `valence`, `acousticness`, etc.)
- metadata (`explicit`, `duration_ms`, `track_name`, `track_genre`, etc.)

## 🔧 Tools Used

- **Python** (`pandas`, `matplotlib`, `seaborn`, `sklearn`, `imblearn`, `shap`)
- Jupyter Notebook
- GitHub

---

## 🧠 Key Questions Explored

- How can we define a “hit” song using available data?
- Can we build a model that predicts hit songs using audio features alone?
- Which features most strongly contribute to a song’s hit status?
- How do linear and ensemble models perform differently on imbalanced data?

## 📈 Sample Visualizations

![Confusion Matrix](https://github.com/user-attachments/assets/0320a477-2253-4f50-8622-4fdffa7d5006)
![Feature Importance](https://github.com/user-attachments/assets/7ddfdea8-c4ec-40d4-b34e-29bd634f86d9)
![SHAP Sample](https://github.com/user-attachments/assets/7c54c28e-04fb-45a4-9678-f38c50989381)

## ✅ What I Learned

- Applied SMOTE to handle real-world class imbalance
- Compared classical (logistic regression) vs ensemble (random forest) models
- Interpreted model decisions using feature importance and SHAP values
- Discovered that predicting song hits requires non-linear, context-sensitive reasoning
- Practiced complete ML workflow from cleaning to evaluation and interpretation

## 📌 Notes

- `popularity` was removed to prevent data leakage
- SHAP plots were generated on a 100-row subset due to compute limits
- Features like `acousticness` and `instrumentalness` emerged as important, despite weak direct correlation with hit status
- Highlights that success in music may depend on more than just audio traits—context matters

