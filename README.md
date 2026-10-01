# 📊 Machine Learning & Memory-Optimized Analytics

A comprehensive predictive machine learning pipelines.

## 📌 Project Overview
This project centers on a **Machine Learning Classification**. Categorizing company funding rounds using predictive classification modeling, deployed via a containerized API.


## 🛠️ Tech Stack
* **Machine Learning & Analytics**: Python 3 (Pandas / Scikit-Learn)
* **API Framework**: FastAPI
* **DevOps**: Docker

---

## ✨ Solutions & Implementation Details

### 🔹 Machine Learning Model (Funding Round Classifier)

* **Objective**: Predicts the specific investment round category (`angel`, `seed`, `a`, `b`, `c`) using features from `python_task_data.csv`.
* **Pipeline**: Built with automated feature transformers, string encodings, and robust classifiers designed to prevent class imbalance issues. GridSearchCV and the Pipeline was then used to optimize MLPClassifier, which was the model of choice for this project. Find the classifier with the best hyperparameters as follows:

```MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 100, 50), max_iter=500,
              random_state=42, solver='sgd')
              ```

* **API & Deployment (Bonus)**: Features a dedicated `POST /predict` endpoint exposed via an asynchronous server framework and containerized inside an isolated Docker network.

## ⚙️ Getting Started & Installation

### 1. Project Dependencies Installation
Clone the repository and install the production package requirements locally:
```bash
pip install -r requirements.txt
```

### 2. Launching the App / API Local Server
Execute the application entry point script to boot up the analytics engine and live endpoints:
```bash
 uvicorn main:app --reload
```

### 3. Container Deployment via Docker
Build and serve the containerized infrastructure network instantly:
```bash
cd Investment_Round_Predictor
docker build -t investment-round-prediction-app .
docker run -p 80:80 investment-round-prediction-app
```

### 4. Prediction and Series Analysis endpoints
The prediction endpoints are:

```http://0.0.0.0/predict``` -- For a single prediction,
```http://0.0.0.0/predict_batch``` -- For the batch prediction

The analytics endpoint is:
```http://127.0.0.1:8000/analyse_series```

Examples of the request bodies are:

```{"numEmps": 20,
  "category": "web",
  "city": "Washington",
  "state": "DC",
  "fundedDate": "1-May-01",
  "raisedAmt": 2000000,
  "raisedCurrency": "USD"
}``` - For single prediction,


```{
  "investments": [
    {
  "numEmps": 20,
  "category": "web",
  "city": "Washington",
  "state": "DC",
  "fundedDate": "1-May-01",
  "raisedAmt": 2000000,
  "raisedCurrency": "USD"
}
  ]
}``` -- For batch prediction.


### 5. Endpoints demo

<img width="2000" height="946" alt="image" src="https://github.com/user-attachments/assets/d24fde68-8578-4fcc-9aa4-c532dd981b11" />

<img width="2000" height="982" alt="image" src="https://github.com/user-attachments/assets/3340c4b7-d491-4a1e-a862-a979e7db631e" />

<img width="2002" height="744" alt="image" src="https://github.com/user-attachments/assets/029f7b17-7330-4636-a6a2-596a5c93945c" />


