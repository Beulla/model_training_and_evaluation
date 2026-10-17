# Early Identification of At-Risk Learners Using Machine Learning

**Author:** Beulla Jeanne Francoise Rugero  
**Module:** Introduction to Machine Learning

## Mission

This project connects to my goal of creating safe havens for children who never had loving families and those from unstable or abusive homes. While that mission is about child welfare, this project applies the same core idea: **use machine learning to identify vulnerable people early so support can reach them before it is too late**.

## Problem

Predict whether a distance-learning student will achieve a positive outcome (Distinction/Pass) or a negative one (Fail/Withdrawn), using the Open University Learning Analytics Dataset (OULAD).

## Dataset

| Property | Value |
|----------|-------|
| Name | Open University Learning Analytics Dataset (OULAD) |
| Source | [UCI ML Repository #349](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset) |
| Rows | 32,593 |
| Target | `final_result` — 4-class classification |

OULAD includes deprivation level, disability status, and learning behaviour data — useful for studying vulnerable learners.

## Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. **File → Open notebook → GitHub**
3. Enter: `Beulla/model_training_and_evaluation`
4. Open `student_outcome_prediction.ipynb`
5. **Runtime → Run all**

## Repository

https://github.com/Beulla/model_training_and_evaluation

## References

- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171.
- Children's Bureau. AFCARS Foster Care Data. [NDACAN](https://www.ndacan.acf.hhs.gov/datasets/datasets-list-afcars-foster-care.cfm)
