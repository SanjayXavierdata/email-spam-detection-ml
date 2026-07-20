# Email Spam Detection using Machine Learning

A machine learning project that classifies email/SMS messages as **Spam** or **Ham** (not spam) using TF-IDF vectorization and multiple classification algorithms.

## Overview

This project builds a text classification pipeline that:
- Cleans and preprocesses a labeled dataset of spam/ham messages
- Converts text into numerical features using **TF-IDF**
- Trains and compares 4 machine learning models
- Selects the best-performing model based on F1 Score
- Provides a simple prediction system to classify new messages

## Dataset

- **Source:** Combined spam/ham message dataset (`combined_dataset.csv`)
- **Size:** 10,961 messages (674 duplicates removed → ~10,287 after cleaning)
- **Columns:** `target` (spam/ham), `text` (message content)

## Models Trained

| Model | Description |
|---|---|
| Naive Bayes | Probabilistic classifier, fast baseline for text |
| Logistic Regression | Linear model, strong general-purpose performer |
| Decision Tree | Rule-based classifier |
| Random Forest | Ensemble of decision trees |

## Workflow

1. **Data loading & exploration** — inspect shape, missing values, duplicates, class distribution
2. **Data cleaning** — remove duplicates, encode target (`ham` → 0, `spam` → 1)
3. **TF-IDF vectorization** — convert text into numerical features (max 3000 features, English stopwords removed)
4. **Train-test split** — 80/20 split, stratified by class
5. **Model training** — train all 4 models on the TF-IDF features
6. **Model evaluation** — compare Accuracy, Precision, Recall, and F1 Score; select the best model
7. **Prediction system** — classify new messages as Spam or Ham

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (`TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`)

## How to Run

1. Clone this repository
2. Place `combined_dataset.csv` in the project folder
3. Open `Email_Spam_Detection_ML.ipynb` in Jupyter/Google Colab
4. Run all cells in order

## Results

The models were evaluated on accuracy, precision, recall, and F1 score, with the best-performing model selected for final predictions. (See notebook output for exact metrics.)

## Future Improvements

- Add SVM and other algorithms for comparison
- Save the trained model and vectorizer for deployment (`.pkl` files)
- Build a simple web app (Streamlit/Flask) for live predictions
- Add cross-validation for more robust model comparison

## Author

Sanjay Xavier
