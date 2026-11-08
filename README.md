# Bank Marketing Prediction

## About the Project

This project aims to predict whether a bank customer is likely to subscribe to a term deposit based on customer information and previous marketing campaign details.

Five machine learning classification algorithms were implemented and evaluated using the same dataset. A Streamlit application was also developed to allow users to make predictions and evaluate the trained models using uploaded test data.

## Dataset

The project uses the Bank Marketing dataset, obtained from the UCI Machine Learning Repository.

The dataset contains 45,211 records and 17 columns. The target column is `y`, which indicates whether a customer subscribed to a term deposit.

- `no` represents no subscription.
- `yes` represents a subscription.

The `duration` column was removed before model training to avoid data leakage, as it can reveal information related to the outcome of the marketing call.

After removing the target and duration columns, 15 input features were used for model training.

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset before training the models.

The analysis included:

- Understanding the dataset structure and data types.
- Identifying numerical and categorical features.
- Checking missing values and duplicate records.
- Examining unknown values in categorical columns.
- Understanding the distribution of the target variable.

The analysis showed that the target classes were imbalanced, with more customers not subscribing than subscribing.

## Data Preprocessing

The dataset was divided into training and testing sets using an 80:20 split.

Stratified sampling was used to maintain the target class distribution in both sets.

The following preprocessing techniques were applied:

- StandardScaler was used to scale numerical features.
- OneHotEncoder was used to convert categorical features into numerical values.
- ColumnTransformer was used to apply the transformations to the appropriate columns.

The preprocessing pipeline was fitted using the training data and then applied to the test data.

## Machine Learning Models

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models were trained and evaluated using the same training and testing split.

## Model Evaluation

The models were evaluated using six metrics:

- Accuracy
- ROC-AUC Score
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

### Comparison of Model Performance.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.3732 | 0.2853 |
| Decision Tree | 0.8401 | 0.6036 | 0.3083 | 0.2949 | 0.3014 | 0.2113 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.8361 | 0.7913 | 0.3753 | 0.6030 | 0.4627 | 0.3867 |

### Observations

**Logistic Regression**

Logistic Regression achieved a recall of 0.6238, which was the highest among the five models. However, its precision was relatively low, indicating that it also classified several non-subscribing customers as potential subscribers.

**Decision Tree**

The Decision Tree achieved an accuracy of 0.8401 but had a comparatively low AUC of 0.6036. Its recall and F1-score were also lower than those of most other models.

**K-Nearest Neighbors**

KNN achieved the highest accuracy of 0.8874 and precision of 0.5529. However, its recall was only 0.1975, meaning it failed to identify many customers who actually subscribed.

**Naive Bayes**

Naive Bayes achieved reasonably balanced results, with an accuracy of 0.8452, recall of 0.4641 and F1-score of 0.4123. Its overall performance was better than the Decision Tree and KNN in terms of F1-score.

**Support Vector Machine**

SVM achieved the highest AUC of 0.7913, F1-score of 0.4627 and MCC of 0.3867. It also achieved a recall of 0.6030, showing a good ability to identify subscribing customers.

### Overall Best Model

Based on the evaluation results, Support Vector Machine (SVM) performed best overall, achieving the highest AUC, F1-score and MCC among the five models.

Although KNN achieved the highest accuracy, its low recall makes it less suitable when identifying potential term deposit subscribers is important.

## Web Application

A web application was developed using Streamlit.

The application provides the following features:

- Selection of any of the five trained machine learning models.
- Customer information input for individual predictions.
- Prediction of whether a customer is likely to subscribe.
- Display of prediction probability.
- CSV upload option for testing the saved models.
- Display of evaluation metrics for the selected model.
- Confusion matrix and classification report.

The application loads the saved models and preprocessing pipeline to evaluate uploaded test data without retraining the models.

## Tools and Technologies

The following tools and libraries were used:

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Running the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

## Conclusion

This project covers the complete machine learning workflow, starting from understanding and exploring the dataset to preprocessing, training classification models and evaluating their performance.

The five models showed different strengths across the evaluation metrics. SVM achieved the best overall performance based on AUC, F1-score and MCC.

The Streamlit application provides an interactive way to make individual predictions and evaluate the trained models using uploaded test data.