import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import pickle
import joblib
from lime.lime_tabular import LimeTabularExplainer

# 1. Hazır veriyi yükle
veri = np.load('hazir_veri.npz')
X_train = veri['X_train']
X_test = veri['X_test']
y_train = veri['y_train']
y_test = veri['y_test']

# 2. Modelleri tanımla
rf = RandomForestClassifier(random_state=42, n_estimators=100)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgbm = LGBMClassifier(random_state=42)

# 3. Eğitim
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train)

# 4. Test setinde metrikler
def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    print(f"{name} | Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    return acc, f1, auc

print("\n--- Test Sonuçları ---")
evaluate(rf, X_test, y_test, "Random Forest")
evaluate(xgb, X_test, y_test, "XGBoost")
evaluate(lgbm, X_test, y_test, "LightGBM")

# 5. Modelleri kaydet
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
joblib.dump(xgb, 'xgboost_model.pkl')
joblib.dump(lgbm, 'lightgbm_model.pkl')

# 6. SHAP ile açıklanabilirlik (Test setiyle)
print("\n--- SHAP Analizi (Test Seti) ---")
explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X_test)
shap.summary_plot(shap_values_rf[1] if isinstance(shap_values_rf, list) else shap_values_rf, X_test, show=False)
plt.title("Random Forest - SHAP Özellik Önemleri (Test Seti)")
plt.savefig("shap_rf_summary_test.png")
plt.close()

# SHAP örnek açıklama (test setinden rastgele bir örnek)
idx = np.random.randint(0, X_test.shape[0])
shap.initjs()
shap.force_plot(explainer_rf.expected_value[1] if isinstance(explainer_rf.expected_value, np.ndarray) else explainer_rf.expected_value,
                shap_values_rf[1][idx] if isinstance(shap_values_rf, list) else shap_values_rf[idx],
                X_test[idx], matplotlib=True, show=False)
plt.title(f"Random Forest - SHAP Force Plot (Test Seti, Örnek {idx})")
plt.savefig(f"shap_rf_force_test_{idx}.png")
plt.close()

# 7. LIME ile açıklanabilirlik (örnek bir test örneği için)
try:
    from lime.lime_tabular import LimeTabularExplainer
    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    class_names = ["Ret", "Onay"]
    lime_explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names, discretize_continuous=True)
    lime_exp = lime_explainer.explain_instance(X_test[idx], rf.predict_proba, num_features=10)
    fig = lime_exp.as_pyplot_figure()
    plt.title(f"LIME - Random Forest (Test Seti, Örnek {idx})")
    plt.tight_layout()
    plt.savefig(f"lime_rf_explanation_test_{idx}.png")
    plt.close()
    print(f"LIME görseli kaydedildi: lime_rf_explanation_test_{idx}.png")
except Exception as e:
    print(f"LIME görseli üretilemedi: {e}")

print(f"SHAP özet ve örnek görselleri kaydedildi. (Örnek index: {idx})")
print("✅ Model eğitimi, SHAP ve LIME açıklamaları tamamlandı.") 