# FraudShield: Fraud Detection in Financial Transactions

## 1.0 Overview

FraudShield is a financial fraud detection system designed to identify fraudulent transactions in mobile banking. The system leverages machine learning models to detect fraudulent activities, ensuring financial security and minimizing losses for financial institutions.

## 2.0 Business Problem

FraudShield is developed for Blocker Fraud Company, which specializes in detecting fraudulent transactions. The company employs a risk-based monetization strategy:

- **25% commission** on successfully detected frauds.
- **5% commission** on false positives (legitimate transactions misclassified as fraud).
- **100% reimbursement** for false negatives (fraudulent transactions misclassified as legitimate).

The profitability of the company depends on the precision and accuracy of the fraud detection model. A well-performing model increases revenue, while a poor model can lead to significant financial losses.

## 3.0 Solution Strategy

The solution involves building a machine learning model to predict fraudulent transactions. The steps include:

1. **Data Description**: Data collection, handling missing values, and basic statistical analysis.
2. **Feature Engineering**: Hypothesis generation and creation of relevant features.
3. **Data Filtering**: Removal of irrelevant columns and outliers.
4. **Exploratory Data Analysis (EDA)**: Univariate, bivariate, and multivariate analysis to understand patterns.
5. **Data Preparation**: Data transformations (encoding, resampling, scaling) for better model training.
6. **Feature Selection**: Identifying the most relevant features using algorithms like Boruta.
7. **Machine Learning Modeling**: Training various models and evaluating performance.
8. **Hyperparameter Tuning**: Optimizing the best-performing model for improved accuracy.
9. **Conclusions**: Evaluating model generalization on unseen data.
10. **Model Deployment**: Creating an API using Flask to integrate the model into the company’s system.

## 4.0 Key Data Insights

- Fraudulent transactions are typically greater than R$ 10,000.
- Fraud is most common in transfer and cash-out transactions.
- High-value transactions (> R$ 100,000) occur across multiple transaction types.

## 5.0 Machine Learning Models Evaluated

The following machine learning models were evaluated:

| Model                  | Balanced Accuracy | Precision | Recall | F1 Score | Kappa  |
|------------------------|-------------------|-----------|--------|----------|--------|
| Dummy Model            | 49.9%            | 0%        | 0%     | 0%       | -0.001 |
| Logistic Regression     | 56.5%            | 100%      | 12.9%  | 22.9%    | 22.8%  |
| K-Nearest Neighbors     | 70.5%            | 94.2%     | 40.9%  | 56.8%    | 56.7%  |
| Support Vector Machine  | 59.5%            | 100%      | 19%    | 31.9%    | 31.9%  |
| Random Forest           | 86.5%            | 97.2%     | 73.1%  | 83.4%    | 83.3%  |
| XGBoost                 | 88.0%            | 96.3%     | 76.1%  | 85.0%    | 85.0%  |
| LightGBM                | 70.1%            | 18.0%     | 40.7%  | 24.1%    | 23.9%  |

## 6.0 Final Model Performance

The **XGBoost** model was selected as the best-performing model and fine-tuned. Its performance metrics are as follows:

| Metric            | Cross-validation | Unseen Data |
|-------------------|------------------|-------------|
| Balanced Accuracy | 88.1%            | 91.5%       |
| Precision         | 96.3%            | 94.4%       |
| Recall            | 76.3%            | 82.9%       |
| F1 Score          | 85.1%            | 88.3%       |
| Kappa             | 85.1%            | 88.3%       |

## 7.0 Deployment

The model is deployed as a **REST API** using Flask. It receives transaction data and returns a fraud prediction.

## 8.0 Future Improvements

- Integrate real-time fraud detection with streaming data.
- Explore deep learning models for improved accuracy.
- Implement adversarial training to counter evolving fraud tactics.

## 9.0 Conclusion

FraudShield significantly improves fraud detection for Blocker Fraud Company, reducing financial risks while maximizing revenue. The XGBoost model achieved **91.5% balanced accuracy** and **94.4% precision** on unseen data, demonstrating its effectiveness in identifying fraudulent transactions.

---

**Note**: This README provides an overview of the project. For detailed implementation and code, refer to the project repository.
