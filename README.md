# DATA AUTOMATION — ML Pipeline with Experiment Tracking

A machine learning pipeline built with Python that automates data loading,
cleaning, model training and experiment tracking using MLflow.
The project uses the Titanic dataset to predict passenger survival,
and compares multiple experiments to observe how accuracy changes
when different features and settings are used.

---

## Technologies Used

- Python 3.12
- Pandas — data loading and cleaning
- Scikit-learn — machine learning model training
- MLflow — experiment tracking and model logging
- Git & GitHub — version control

---

## How to Install

1. Clone the repository
git clone https://github.com/Princedwane/DATA-AUTOMATION.git

2. Navigate into the project folder
cd DATA-AUTOMATION

3. Create a virtual environment
python3 -m venv venv

4. Activate the virtual environment
source venv/bin/activate

5. Install dependencies
pip install -r requirements.txt

---

## How to Run

Run the full pipeline:
python pipeline.py

Check MLflow experiment results:
python check_runs.py

---

## Experiment Results

| Run | Features Used | Test Size | Accuracy |
|---|---|---|---|
| Run1_Basic | Pclass, Age, Fare | 0.2 | 75.6% |
| Run2_WithSex | Pclass, Age, Fare, Sex | 0.2 | 67.5% |
| Run3_MoreFeatures | Pclass, Age, Fare, Sex, SibSp, Parch | 0.3 | 76.3% |

Run3 achieved the highest accuracy by using more features.

---
## Insights from Experiments

- Adding the Sex column in Run 2 did not improve accuracy because
  train_test_split randomly assigns passengers to test and train sets
  each time it runs, which can produce harder or easier test sets by chance.

- Run 3 achieved the best accuracy of 76.3% by combining more features
  giving the model more information to learn from.

- The most important lesson — more features do not always guarantee
  better accuracy. The quality and relevance of features matters more
  than quantity.

- MLflow allowed us to compare all three runs in one table without
  manually recording any results. This is the power of experiment tracking.

  - train_test_split randomly shuffles data on every run, meaning accuracy
  can change between runs even with identical code and data. This is why
  MLflow experiment tracking is essential — it records every run permanently
  so you can distinguish real improvements from random variation.

---

## Project Structure

pipeline.py — main pipeline, loads, cleans and trains the model
check_runs.py — displays all MLflow experiment runs and metrics
README.md — project documentation
requirements.txt — list of dependencies

---

## Author

Prince John Martine — Information security & Data Science Student
GitHub: https://github.com/Princedwane