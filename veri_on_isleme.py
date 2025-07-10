import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib

# 1. Veriyi oku
df = pd.read_csv("kredi_veri_seti.csv")

# 2. Hedef değişkeni 0/1'e çevir
df["kredi_onay"] = df["kredi_onay"].map({"Onay": 1, "Ret": 0})

# 3. Outlier temizliği (örnek: gelir, mevcut_borc, kredi_tutari)
for col in ["gelir", "mevcut_borc", "kredi_tutari"]:
    q1 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    df = df[(df[col] >= q1) & (df[col] <= q99)]

# 4. Yeni özellikler
df["borc_gelir_orani"] = df["mevcut_borc"] / (df["gelir"] + 1)
df["kredi_gelir_orani"] = df["kredi_tutari"] / (df["gelir"] + 1)
df["yas_grubu"] = pd.cut(df["yas"], bins=[17, 30, 45, 60, 100], labels=["Genç", "Orta", "Yetişkin", "Yaşlı"])

# 5. Kategorik ve sayısal değişkenler
num_cols = ["yas", "gelir", "kredi_tutari", "kredi_suresi_ay", "mevcut_borc", "borc_gelir_orani", "kredi_gelir_orani"]
cat_cols = ["calisma_durumu", "egitim", "medeni_durum", "ev_sahibi", "kredi_gecmisi", "yas_grubu"]

# 6. Eğitim/test bölmesi (stratify=y)
X = df[num_cols + cat_cols]
y = df["kredi_onay"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Kategorik değişkenleri one-hot encode et (fit sadece train'de!)
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(X_train[cat_cols])
cat_train = encoder.transform(X_train[cat_cols])
cat_test = encoder.transform(X_test[cat_cols])

# 8. Sayısal değişkenleri MinMaxScaler ile normalleştir (fit sadece train'de!)
scaler = MinMaxScaler()
scaler.fit(X_train[num_cols])
num_train = scaler.transform(X_train[num_cols])
num_test = scaler.transform(X_test[num_cols])

# 9. Tüm feature'ları birleştir
X_train_final = np.concatenate([num_train, cat_train], axis=1)
X_test_final = np.concatenate([num_test, cat_test], axis=1)

# 10. Kaydet
np.savez("hazir_veri.npz", X_train=X_train_final, X_test=X_test_final, y_train=y_train, y_test=y_test)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

print("✅ Gelişmiş veri önişleme tamamlandı. Eğitim ve test setleri, scaler ve encoder kaydedildi.") 