# ML Internship Project - Employee Salary Prediction

Project done as part of the **AI Pioneers Internship (Machine Learning) - Yuva Intern**.

## About
This project cleans a sample employee dataset and compares two supervised ML models (Linear Regression and Decision Tree) to predict employee salary.

## Libraries Used
Python, Pandas, NumPy, Matplotlib, Scikit-learn

## Files
- `employee_ml_project.py` - main code
- `employee_data.csv` - raw sample dataset
- `employee_data_cleaned.csv` - dataset after preprocessing
- `model_results.csv` - model comparison results
- `*.png` - graphs
- `ML_Internship_Project_Report.docx` - project report

## Steps
1. Create sample data (with missing values and duplicates)
2. Clean data: remove duplicates, fill missing values, one-hot encoding, scaling
3. Split into train and test sets (80/20)
4. Train Linear Regression and Decision Tree
5. Compare using MSE, RMSE and R2 score

## Result
| Model | RMSE | R2 Score |
|---|---|---|
| Linear Regression | 6845.38 | 0.9056 |
| Decision Tree | 9627.03 | 0.8133 |

Linear Regression performed better on this data.

## How to Run
```
pip install pandas numpy matplotlib scikit-learn
python employee_ml_project.py
```
