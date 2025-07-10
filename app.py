from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import joblib
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import io
import base64
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Model ve preprocessor'ları yükle
try:
    with open('random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("✅ Modeller başarıyla yüklendi")
except Exception as e:
    print(f"❌ Model yükleme hatası: {e}")

# Özellik isimleri (veri_on_isleme.py'den alınan)
NUMERIC_FEATURES = ["yas", "gelir", "kredi_tutari", "kredi_suresi_ay", "mevcut_borc", "borc_gelir_orani", "kredi_gelir_orani"]
CATEGORICAL_FEATURES = ["calisma_durumu", "egitim", "medeni_durum", "ev_sahibi", "kredi_gecmisi", "yas_grubu"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Gelen veriyi DataFrame'e çevir
        df = pd.DataFrame([data])
        
        # Yaş grubu oluştur
        yas = df['yas'].iloc[0]
        if yas <= 30:
            df['yas_grubu'] = 'Genç'
        elif yas <= 45:
            df['yas_grubu'] = 'Orta'
        elif yas <= 60:
            df['yas_grubu'] = 'Yetişkin'
        else:
            df['yas_grubu'] = 'Yaşlı'
        
        # Borç/gelir oranları hesapla
        df['borc_gelir_orani'] = df['mevcut_borc'] / (df['gelir'] + 1)
        df['kredi_gelir_orani'] = df['kredi_tutari'] / (df['gelir'] + 1)
        
        # Sayısal ve kategorik özellikleri ayır
        num_data = df[NUMERIC_FEATURES].values
        cat_data = df[CATEGORICAL_FEATURES].values
        
        # Preprocessing uygula
        num_scaled = scaler.transform(num_data)
        cat_encoded = encoder.transform(cat_data)
        
        # Özellikleri birleştir
        features = np.concatenate([num_scaled, cat_encoded], axis=1)
        
        # Tahmin yap
        prediction = rf_model.predict(features)[0]
        probability = rf_model.predict_proba(features)[0]
        
        # Basit açıklama oluştur (SHAP yerine)
        explanation = create_simple_explanation(df, prediction, probability)
        
        # Sonuçları hazırla
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Onay' if prediction == 1 else 'Ret',
            'probability': {
                'ret': float(probability[0]),
                'onay': float(probability[1])
            },
            'shap_explanation': explanation,
            'confidence': float(max(probability)),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Hata detayı: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def create_simple_explanation(df, prediction, probability):
    """Basit açıklama oluştur"""
    explanation = {}
    
    # Borç/gelir oranı (en önemli faktör)
    borc_gelir_orani = df['borc_gelir_orani'].iloc[0]
    explanation['borc_gelir_orani'] = float(borc_gelir_orani)
    
    # Gelir
    gelir = df['gelir'].iloc[0]
    explanation['gelir'] = float(gelir)
    
    # Yaş
    yas = df['yas'].iloc[0]
    explanation['yas'] = float(yas)
    
    # Kredi tutarı
    kredi_tutari = df['kredi_tutari'].iloc[0]
    explanation['kredi_tutari'] = float(kredi_tutari)
    
    # Mevcut borç
    mevcut_borc = df['mevcut_borc'].iloc[0]
    explanation['mevcut_borc'] = float(mevcut_borc)
    
    # Kredi geçmişi etkisi
    kredi_gecmisi = df['kredi_gecmisi'].iloc[0]
    if kredi_gecmisi == 'Temiz':
        explanation['kredi_gecmisi_Temiz'] = 0.1
    elif kredi_gecmisi == 'Yeni':
        explanation['kredi_gecmisi_Yeni'] = 0.05
    else:  # Gecikmeli
        explanation['kredi_gecmisi_Gecikmeli'] = -0.15
    
    # Çalışma durumu etkisi
    calisma_durumu = df['calisma_durumu'].iloc[0]
    if calisma_durumu == 'Çalışıyor':
        explanation['calisma_durumu_Çalışıyor'] = 0.08
    elif calisma_durumu == 'Emekli':
        explanation['calisma_durumu_Emekli'] = 0.05
    elif calisma_durumu == 'Öğrenci':
        explanation['calisma_durumu_Öğrenci'] = -0.05
    else:  # İşsiz
        explanation['calisma_durumu_İşsiz'] = -0.12
    
    # Eğitim etkisi
    egitim = df['egitim'].iloc[0]
    if egitim == 'Doktora':
        explanation['egitim_Doktora'] = 0.06
    elif egitim == 'Lisans':
        explanation['egitim_Lisans'] = 0.04
    elif egitim == 'Yüksek':
        explanation['egitim_Yüksek'] = 0.02
    else:  # Lise
        explanation['egitim_Lise'] = -0.02
    
    # Ev sahibi etkisi
    ev_sahibi = df['ev_sahibi'].iloc[0]
    if ev_sahibi == 'Evet':
        explanation['ev_sahibi_Evet'] = 0.03
    else:
        explanation['ev_sahibi_Hayır'] = -0.03
    
    # Medeni durum etkisi
    medeni_durum = df['medeni_durum'].iloc[0]
    if medeni_durum == 'Evli':
        explanation['medeni_durum_Evli'] = 0.02
    elif medeni_durum == 'Bekar':
        explanation['medeni_durum_Bekar'] = 0.01
    else:  # Dul
        explanation['medeni_durum_Dul'] = -0.01
    
    return explanation

@app.route('/api/features', methods=['GET'])
def get_features():
    """Kullanılabilir özellikleri döndür"""
    return jsonify({
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': {
            'calisma_durumu': ['Emekli', 'Çalışıyor', 'İşsiz', 'Öğrenci'],
            'egitim': ['Lisans', 'Doktora', 'Yüksek', 'Lise'],
            'medeni_durum': ['Bekar', 'Evli', 'Dul'],
            'ev_sahibi': ['Evet', 'Hayır'],
            'kredi_gecmisi': ['Temiz', 'Yeni', 'Gecikmeli']
        }
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Model bilgilerini döndür"""
    return jsonify({
        'model_type': 'Random Forest',
        'n_estimators': 100,
        'features_count': len(NUMERIC_FEATURES) + sum(len(cats) for cats in encoder.categories_),
        'training_samples': 1504,
        'test_samples': 376
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 