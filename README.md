# Financial Transaction Anomaly and Fraud Detection

## Project Overview
This machine learning project aims to explore, identify anomalies, and detect potential fraud based on customer financial activity patterns. Utilizing a dataset of 2,512 transaction records—which includes transaction attributes, demographics, and device/network usage patterns—this project builds a transaction profile segmentation using **Clustering** models. This unsupervised approach serves as the foundation for developing a predictive **Classification** model focused on financial security.

## Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn, Yellowbrick (*KElbowVisualizer*)
* **Machine Learning:** Scikit-Learn (*KMeans, PCA, Silhouette Score, LabelEncoder, StandardScaler*)
* **Deployment/Model Serialization:** Joblib

## Dataset & Key Features
The dataset contains comprehensive information regarding customer transaction behavior in the United States, including:
* **Entity Information:** `TransactionID`, `AccountID`, `MerchantID`, `CustomerAge`, `CustomerOccupation`
* **Transaction Details:** `TransactionAmount`, `TransactionDate`, `PreviousTransactionDate`, `TransactionType` (Credit/Debit), `AccountBalance`, `TransactionDuration`
* **Digital Footprint & Anomalies:** `Location`, `Channel` (Online/ATM/Branch), `DeviceID`, `IP Address`, `LoginAttempts`

## Project Workflow
1. **Data Ingestion:** Extracted the transaction dataset from a cloud data repository (Google Sheets/CSV).
2. **Exploratory Data Analysis (EDA):** Reviewed data structure, descriptive statistics, and identified initial value distributions.
3. **Data Preprocessing:** Applied label encoding for categorical data and standardization for numerical features.
4. **Dimensionality Reduction:** Utilized Principal Component Analysis (PCA) to optimize pattern processing.
5. **Clustering Analysis:** Evaluated the optimal number of clusters using the *Elbow Method* and *Silhouette Score*, followed by *K-Means* modeling.
6. **Classification Pipeline (Next Step):** Leveraging the clustering results as target labels to train a predictive classification model for automated fraud detection.