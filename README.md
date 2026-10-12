# House Price Prediction

A complete, beginner-friendly machine learning project to predict house prices using the California Housing dataset.

## Project Structure

```
house-price-prediction/
├── data/                 # Raw and processed data (if any)
├── notebooks/            # Jupyter notebooks for exploratory data analysis (EDA)
│   └── eda.ipynb
├── plots/                # Generated plots (e.g., feature importance, actual vs predicted)
├── src/                  # Source code for data preparation, training, and evaluation
│   ├── data_prep.py      # Load, clean, feature engineering, and split data
│   ├── train.py          # Train and tune models (Linear Regression, Random Forest, XGBoost)
│   ├── evaluate.py       # Calculate metrics (RMSE, MAE, R²) and generate plots
│   └── predict.py        # Make predictions using the saved model
├── models/               # Saved model files (e.g., best_model.joblib)
├── app.py                # Streamlit web application
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository** (or download the files).
2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Exploratory Data Analysis**:
   Open `notebooks/eda.ipynb` using Jupyter Notebook to explore the dataset and save correlation/distribution plots.

2. **Train the Model**:
   Run the training script to prepare data, train models, tune hyperparameters for XGBoost, and save the best model.
   ```bash
   python -m src.train
   ```

3. **Evaluate the Model**:
   Run the evaluation script to test the model on the test set and generate plots (`actual_vs_predicted.png`, `feature_importance.png`).
   ```bash
   python -m src.evaluate
   ```

4. **Run the Web App**:
   Start the Streamlit application to serve predictions through a user-friendly interface.
   ```bash
   streamlit run app.py
   ```

## Results Summary

- Evaluated models: Linear Regression, Random Forest, XGBoost.
- Best model: XGBoost (tuned with RandomizedSearchCV).
- Target variable (`Price`) was log-transformed (`np.log1p`) to handle skewness and transformed back (`np.expm1`) for evaluation.
- View the `plots/` folder for feature importance and prediction accuracy charts.
"# house-price-prediction" 
