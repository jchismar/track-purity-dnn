import logging
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
from pathlib import Path

def shap_analysis(model, X, feature_names, output_dir, max_samples=1000, batch_size=32768):
    logging.info(f"\nComputing SHAP values (up to {max_samples} samples)...")

    n_samples = min(max_samples, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_subset = X[indices]

    background = X_subset[:min(100, len(X_subset))]

    def predict_fn(data):
        return model.predict(data, batch_size=batch_size)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_subset)

    output_path = Path(output_dir)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_subset, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path / 'shap_summary.png'}")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_subset, feature_names=feature_names, plot_type='bar', show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path / 'shap_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path / 'shap_importance.png'}")

    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_shap
    }).sort_values('Importance', ascending=False)

    importance_df.to_csv(output_path / 'shap_importance.csv', index=False)
    logging.info(f"Saved: {output_path / 'shap_importance.csv'}")

    return shap_values, importance_df
