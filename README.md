#Explainable Medical Insurance Cost Intelligence

A redesigned, portfolio-grade version of the original Medical Insurance Cost Prediction internship project.

## Why this version is stronger

The original project proved the basic idea with Linear Regression, Random Forest, a neural network, and a Streamlit form. This version turns that prototype into a complete ML product:

- **Model benchmarking:** Linear Regression, Random Forest, Gradient Boosting, and Extra Trees.
- **Robust preprocessing:** reusable `ColumnTransformer` + `Pipeline` instead of manual one-hot encoding.
- **Evaluation:** 5-fold cross-validation plus holdout MAE, RMSE, and R².
- **Automatic model selection:** the best model is selected by RMSE and persisted with `joblib`.
- **Uncertainty:** an empirical 80% prediction band from out-of-fold residuals.
- **Explainability:** local counterfactual feature impacts + global feature-signal analysis.
- **Interactive product UI:** profile builder, prediction gauge, what-if scenarios, model lab, data explorer, and explainability dashboard.
- **Deployment-friendly stack:** removes the TensorFlow release-candidate dependency that made the earlier app fragile on Streamlit Cloud.

## Dataset

`insurance.csv` contains 1,338 observations with age, sex, BMI, children, smoker, region, and target insurance charges.

> This is a portfolio / educational ML application. Predictions are not insurance quotes, underwriting decisions, or medical advice.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python scripts/train.py
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push the repository to GitHub.
2. Select `app.py` as the main file.
3. Use **Python 3.12** in the deployment environment.
4. `requirements.txt` contains the tested core stack.

The first training run creates `artifacts/model_bundle.joblib`; later app launches load the saved model immediately.

## Verified benchmark snapshot

With the included dataset and fixed random seed, the current training run selects **Gradient Boosting** using 5-fold CV RMSE. Its reported holdout metrics are **MAE ≈ $1,541**, **RMSE ≈ $4,382**, and **R² ≈ 0.876**. These are project-specific benchmark results, not guarantees for unseen datasets or production insurance pricing.

## Resume bullets

**MedPredict AI — Explainable Medical Insurance Cost Intelligence**  
- Engineered an end-to-end insurance-cost regression system with reusable preprocessing pipelines and four benchmark models, using 5-fold cross-validation and RMSE-based model selection.
- Added empirical prediction intervals, what-if scenario analysis, and model-agnostic counterfactual explanations to make regression outputs interpretable.
- Built a production-style Streamlit dashboard with model benchmarking, data exploration, explainability, and persisted artifacts for deployment.

## Suggested GitHub repository description

> Explainable medical insurance cost intelligence platform with multi-model benchmarking, uncertainty estimation, counterfactual explanations, and an interactive Streamlit dashboard.

## Optional REST API

The same trained artifact can be served through FastAPI:

```bash
uvicorn api:app --reload
```

Then use `GET /health` or `POST /predict`.

## Engineering additions

- Dataset validation layer with schema/value checks.
- Persisted model artifact to avoid retraining during normal app startup.
- FastAPI inference endpoint reusing the same model bundle as the dashboard.
- GitHub Actions CI smoke test on Python 3.12.
- Streamlit theme configuration and `.gitignore` for clean repository hygiene.

## Credits

Original internship project by **Saiashish Chauhan** and **Utkarsh Pachpande**. This repository is the enhanced portfolio edition.
