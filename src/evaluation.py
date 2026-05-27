import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred, dataset_name="", model_name=""):
    """Tính metrics sau khi inverse transform"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    within5 = np.mean(np.abs(y_true - y_pred) <= 5) * 100
    within10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100

    print(f"\n=== {model_name} - {dataset_name} ===")
    print(f"MAE      : {mae:.3f} µg/m³")
    print(f"RMSE     : {rmse:.3f} µg/m³")
    print(f"R²       : {r2:.4f}")
    print(f"MAPE     : {mape:.2f}%")
    print(f"Within±5 : {within5:.1f}%")
    print(f"Within±10: {within10:.1f}%")
    print("="*60)

    return {
        'Model': model_name,
        'Dataset': dataset_name,
        'MAE': round(mae, 3),
        'RMSE': round(rmse, 3),
        'R²': round(r2, 4),
        'MAPE (%)': round(mape, 2),
        'Within±5 (%)': round(within5, 1),
        'Within±10 (%)': round(within10, 1)
    }


def predict_and_inverse(model, X, y_true_scaled, scaler):
    """Predict và inverse transform về đơn vị µg/m³"""
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(np.array(y_true_scaled).reshape(-1, 1)).flatten()
    return y_pred, y_true