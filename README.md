# 🏦 Kredi Onay Sistemi - XAI Destekli

Bu proje, yapay zeka destekli kredi onay kararlarını tahmin eden ve bu kararları açıklayabilen (Explainable AI - XAI) kapsamlı bir web uygulamasıdır.

## 📋 Proje Özeti

### 🎯 Amaç
- Kredi başvurularını otomatik olarak değerlendirmek
- Karar verme sürecini şeffaf ve anlaşılır hale getirmek
- SHAP ve LIME gibi XAI teknikleriyle karar açıklamaları sağlamak

### 🔧 Kullanılan Teknolojiler
- **Backend:** Python, Flask, Scikit-learn
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js
- **ML Modelleri:** Random Forest, XGBoost, LightGBM
- **XAI Teknikleri:** SHAP, LIME
- **Veri İşleme:** Pandas, NumPy

## 📊 Veri Seti

### Özellikler
- **Toplam Kayıt:** 2,000
- **Özellik Sayısı:** 13 (7 sayısal + 6 kategorik)
- **Hedef Değişken:** Kredi Onay (Onay/Ret)

### Sayısal Özellikler
- Yaş, Gelir, Kredi Tutarı, Kredi Süresi, Mevcut Borç
- Borç/Gelir Oranı, Kredi/Gelir Oranı

### Kategorik Özellikler
- Çalışma Durumu, Eğitim, Medeni Durum
- Ev Sahibi, Kredi Geçmişi, Yaş Grubu

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
```bash
Python 3.8+
pip
```

### 2. Paket Kurulumu
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Çalıştırma
```bash
python app.py
```

### 4. Web Arayüzüne Erişim
Tarayıcınızda `http://localhost:5000` adresine gidin.

## 📁 Proje Yapısı

```
bitirme_projesi/
├── app.py                      # Flask web uygulaması
├── model_egit.py              # Model eğitimi ve XAI analizi
├── veri_on_isleme.py          # Veri ön işleme
├── kredi_veri_seti.csv        # Orijinal veri seti
├── hazir_veri.npz             # Ön işlenmiş veri
├── random_forest_model.pkl    # Eğitilmiş model
├── scaler.pkl                 # Veri normalleştirme
├── encoder.pkl                # Kategorik kodlama
├── templates/
│   └── index.html             # Web arayüzü
├── requirements.txt           # Python paketleri
└── README.md                  # Bu dosya
```

## 🔍 Özellikler

### 🤖 Makine Öğrenmesi
- **Random Forest:** %85+ doğruluk oranı
- **XGBoost:** Hızlı ve etkili sınıflandırma
- **LightGBM:** Hafif ve verimli model

### 🧠 Açıklanabilir AI (XAI)
- **SHAP Analizi:** Özellik önem dereceleri
- **LIME Açıklamaları:** Yerel tahmin açıklamaları
- **Görselleştirmeler:** Etkileşimli grafikler

### 💻 Web Arayüzü
- **Responsive Tasarım:** Mobil uyumlu
- **Gerçek Zamanlı Tahmin:** Anında sonuçlar
- **Kullanıcı Dostu:** Sezgisel arayüz

## 📈 Model Performansı

### Test Sonuçları
| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 0.856 | 0.823 | 0.891 |
| XGBoost | 0.849 | 0.815 | 0.884 |
| LightGBM | 0.852 | 0.819 | 0.887 |

### En Önemli Özellikler
1. **Borç/Gelir Oranı** - En kritik faktör
2. **Kredi Geçmişi** - Geçmiş performans
3. **Gelir** - Finansal kapasite
4. **Yaş** - Risk profili
5. **Çalışma Durumu** - Gelir istikrarı

## 🔧 API Endpoints

### POST `/api/predict`
Kredi onay tahmini yapar.

**Request Body:**
```json
{
  "yas": 30,
  "gelir": 50000,
  "kredi_tutari": 100000,
  "kredi_suresi_ay": 24,
  "mevcut_borc": 20000,
  "calisma_durumu": "Çalışıyor",
  "egitim": "Lisans",
  "medeni_durum": "Evli",
  "ev_sahibi": "Evet",
  "kredi_gecmisi": "Temiz"
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_text": "Onay",
  "probability": {
    "ret": 0.234,
    "onay": 0.766
  },
  "shap_explanation": {
    "borc_gelir_orani": 0.045,
    "gelir": 0.032,
    "kredi_gecmisi_Temiz": 0.028
  },
  "confidence": 0.766
}
```

### GET `/api/features`
Kullanılabilir özellikleri listeler.

### GET `/api/model-info`
Model bilgilerini döndürür.

## 🎨 Kullanım Örnekleri

### 1. Kredi Başvurusu
1. Web arayüzünü açın
2. Kişisel ve finansal bilgileri girin
3. "Kredi Onayını Kontrol Et" butonuna tıklayın
4. Sonuçları ve açıklamaları inceleyin

### 2. API Kullanımı
```python
import requests

data = {
    "yas": 35,
    "gelir": 75000,
    "kredi_tutari": 150000,
    # ... diğer özellikler
}

response = requests.post("http://localhost:5000/api/predict", json=data)
result = response.json()
print(f"Tahmin: {result['prediction_text']}")
```

## 🔬 Teknik Detaylar

### Veri Ön İşleme
- **Outlier Temizliği:** %1-%99 aralığında filtreleme
- **Özellik Mühendisliği:** Yeni oranlar ve gruplar
- **Normalizasyon:** MinMaxScaler (0-1 aralığı)
- **Kodlama:** One-Hot Encoding

### Model Seçimi
Random Forest modeli seçildi çünkü:
- Yüksek doğruluk oranı
- Açıklanabilirlik avantajı
- Overfitting'e karşı direnç
- SHAP ile uyumluluk

### XAI Uygulaması
- **SHAP:** Global ve yerel açıklamalar
- **LIME:** Bireysel tahmin açıklamaları
- **Görselleştirme:** Etkileşimli grafikler

## 🚨 Güvenlik ve Etik

### Veri Güvenliği
- Kişisel veriler şifrelenir
- HTTPS kullanımı
- Veri saklama süreleri

### Etik İlkeler
- Şeffaf karar verme
- Ayrımcılık önleme
- Kullanıcı kontrolü

## 🔮 Gelecek Geliştirmeler

### Planlanan Özellikler
- [ ] Çoklu model karşılaştırması
- [ ] Gerçek zamanlı model güncelleme
- [ ] Mobil uygulama
- [ ] API rate limiting
- [ ] Kullanıcı kimlik doğrulama

### Teknik İyileştirmeler
- [ ] Model performans optimizasyonu
- [ ] Daha gelişmiş XAI teknikleri
- [ ] Otomatik model retraining
- [ ] A/B testing framework

## 📞 İletişim ve Destek

### Geliştirici
- **Ad:** [Adınız]
- **E-posta:** [e-posta@adres.com]
- **GitHub:** [github.com/kullaniciadi]

### Katkıda Bulunma
1. Fork yapın
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request gönderin

## Arayüzden görseller
Ana Sayfa :
![Ana Sayfa](img/aryz1.png)
Kredi sonucu tahmini ekranı :
![Sonuç Ekranı](img/aryz2.png)
![Sonuç Ekranı](img/aryz3.png)

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🙏 Teşekkürler

- Scikit-learn ekibine
- SHAP ve LIME geliştiricilerine
- Flask ve Bootstrap topluluklarına
- Veri setini sağlayan kuruma

---

**Not:** Bu proje eğitim amaçlı geliştirilmiştir. Gerçek kredi kararları için profesyonel değerlendirme gerekir. 
