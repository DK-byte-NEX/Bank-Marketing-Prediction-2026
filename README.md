# Bank Marketing Prediction

## About the Project

This project focuses on predicting whether a bank customer is likely to subscribe to a term deposit. It uses customer details and previous marketing campaign information to train and compare different machine learning models.

## Dataset

The project uses the Bank Marketing dataset, which contains 45,211 records and 17 columns.

The target column is `y`, which indicates whether the customer subscribed to a term deposit.

- no represents no subscription
- yes represents a subscription

The duration column was removed before training to avoid data leakage.

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset before building the models.

The analysis included checking the dataset structure, numerical and categorical columns, missing values, duplicate records, unknown values and the distribution of the target variable.

## Data Preprocessing

The dataset was divided into training and testing sets using an 80:20 split.

Numerical features were scaled using StandardScaler, while categorical features were converted into numerical values using OneHotEncoder.

A ColumnTransformer was used to apply both preprocessing techniques. Stratified sampling was used to maintain the target distribution in both datasets.

## Machine Learning Models

Five machine learning algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Naive Bayes
5. Support Vector Machine

## Model Evaluation

The models were evaluated using accuracy, precision, recall, F1-score, ROC-AUC and Matthews Correlation Coefficient.

Confusion matrices and classification reports were also generated to understand model performance.

The following are the SVM evaluation results:

| Metric | Score |
|---|---:|
| Accuracy | 0.8361 |
| Precision | 0.3753 |
| Recall | 0.6030 |
| F1-score | 0.4627 |
| ROC-AUC | 0.7913 |
| MCC | 0.3867 |

## Web Application

A web application was developed using Streamlit.

Users can select one of the five trained models, enter customer information and get a subscription prediction along with its probability.

The application also allows users to upload a test dataset and view model evaluation results, including performance metrics, confusion matrix and classification report.

## Tools and Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib and Streamlit were used in this project.

## Running the Application

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Start the application using:

```bash
streamlit run app.py
```

## Conclusion

Through this project, I worked on the complete machine learning process, starting from understanding and preparing the dataset to training different classification models and evaluating their performance.

I also developed a Streamlit application to make the predictions accessible through a simple user interface.