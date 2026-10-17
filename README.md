# Predicting Student Academic Outcomes with Machine Learning

Summative project comparing traditional ML (scikit-learn) and deep learning (TensorFlow/Keras) on the Open University Learning Analytics Dataset (OULAD).

## Problem

Predict student course outcome (`final_result`: Distinction, Pass, Fail, Withdrawn) from demographics, assessments, and VLE interaction data.

## Dataset

| Property | Value |
|----------|-------|
| Name | Open University Learning Analytics Dataset (OULAD) |
| Source | [UCI ML Repository #349](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset) |
| Rows | 32,593 |
| Target | 4-class classification |

Raw data is **not** stored in this repo (per assignment rules). The notebook downloads it automatically from UCI, or from a GitHub Release if you configure one.

## Repository

**GitHub:** https://github.com/Beulla/model_training_and_evaluation

## Run on Google Colab (recommended)

1. Go to [Google Colab](https://colab.research.google.com/).
2. **File → Open notebook → GitHub**
3. Paste: `Beulla/model_training_and_evaluation`
4. Open `student_outcome_prediction.ipynb`
5. **Runtime → Run all**

The notebook will install dependencies, download OULAD, train all models, and save outputs to Google Drive.

## Project structure

```
model_training_and_evaluation/
├── student_outcome_prediction.ipynb   # Main notebook (run this)
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## Deliverables checklist

| Deliverable | Where |
|-------------|-------|
| Code + notebook | GitHub (this repo) |
| Raw dataset (zipped) | Google Drive |
| Saved models (.joblib, .keras) | Google Drive |
| Written report (IEEE, 3500–5000 words) | PDF with links on page 1 |
| Video (5–10 min) | YouTube / Drive |

## References

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171.
