import shap
import pickle
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "../models_artifacts")

def load_xgboost_model():
    with open(os.path.join(ARTIFACTS_DIR, "xgboost_failure_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model

def create_shap_explainer(model):
    return shap.TreeExplainer(model)

def get_shap_values(explainer, X):
    shap_values = explainer.shap_values(X)
    return shap_values
