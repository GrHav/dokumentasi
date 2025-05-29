# Laporan Proyek Machine Learning - Robert Varian

## Domain Proyek

Dalam industri telekomunikasi, mempertahankan pelanggan lama jauh lebih ekonomis daripada mencari pelanggan baru. Salah satu tantangan utama yang dihadapi perusahaan adalah churn pelanggan, yaitu ketika pelanggan berhenti menggunakan layanan. Churn tidak hanya menyebabkan kehilangan pendapatan tetapi juga meningkatkan biaya pemasaran untuk mendapatkan pelanggan baru. Oleh karena itu, penting bagi perusahaan untuk mengantisipasi perilaku churn agar dapat mengembangkan strategi retensi yang lebih efektif.

Penelitian menunjukkan bahwa pendekatan berbasis machine learning mampu memprediksi churn secara lebih akurat dibandingkan metode konvensional. Menurut Amin et al. (2019), teknik klasifikasi seperti pohon keputusan dan boosting memiliki performa yang tinggi dalam mengklasifikasi pelanggan berdasarkan risiko churn mereka.

Referensi:
Amin, A., Anwar, S., Adnan, A., et al. (2019). Customer churn prediction in the telecommunication sector using a rough set approach. Neural Computing and Applications.

## Business Understanding
### Problem Statements

- Bagaimana cara mengidentifikasi pelanggan yang berisiko tinggi melakukan churn?
- Fitur pelanggan mana yang paling memengaruhi kemungkinan churn?

### Goals

- Membangun model prediksi churn yang akurat untuk membantu perusahaan mengantisipasi kehilangan pelanggan.
- Mengidentifikasi variabel-variabel penting yang paling memengaruhi kemungkinan pelanggan berhenti berlangganan.

### Solution statements

- Menggunakan algoritma Logistic Regression sebagai baseline model karena mudah diinterpretasi.
- Menggunakan algoritma Random Forest dan XGBoost untuk eksplorasi performa dengan metode ensembel dan boosting.
- Melakukan hyperparameter tuning khususnya pada Random Forest dan XGBoost untuk peningkatan performa model.
- Evaluasi model menggunakan metrik F1-score, karena data tidak seimbang (imbalanced dataset) antara pelanggan churn dan non-churn.

## Data Understanding
Dataset: Telco Customer Churn - (Kaggle)[https://www.kaggle.com/datasets/blastchar/telco-customer-churn]

Dataset ini berisi informasi pelanggan dari perusahaan telekomunikasi fiktif, termasuk demografi, layanan yang digunakan, informasi kontrak, dan label churn.

Jumlah data: 7043 baris dan 21 kolom.

### Variabel-variabel pada Telco Customer dataset adalah sebagai berikut:

#### Variabel utama:

- Demografi: gender, SeniorCitizen, Partner, Dependents
- Layanan: PhoneService, InternetService, StreamingTV, StreamingMovies, dll.
- Keuangan & kontrak: MonthlyCharges, TotalCharges, Contract, PaymentMethod
- Target: Churn (Yes/No)

#### Exploratory Data Analysis (EDA):

- Visualisasi distribusi target churn menunjukkan dataset tidak seimbang.
- Histogram fitur numerik menunjukkan distribusi yang bervariasi.
- Korelasi antar fitur numerik divisualisasikan dengan heatmap.

## Data Preparation

Langkah-langkah preprocessing:
- Menghapus nilai null pada TotalCharges (hasil konversi dari string ke float).
- Konversi kolom TotalCharges menjadi numerik karena ada karakter kosong.
- One-hot encoding pada fitur kategorikal agar bisa diproses oleh algoritma ML.
- Standarisasi fitur numerik menggunakan StandardScaler agar memiliki skala yang sama.

Alasan Data Preparation:
- Model ML tidak dapat memproses data kategorikal secara langsung.
- Standarisasi penting untuk algoritma seperti Logistic Regression.
- Menghapus nilai kosong dan tipe data tidak sesuai untuk menghindari error saat training.

## Modeling
Model yang digunakan:
1. Logistic Regression:
  - Kelebihan: Mudah diinterpretasi, baseline yang bagus.
  - Kekurangan: Tidak menangani hubungan non-linear dengan baik.

2. Random Forest:
  - Kelebihan: Menangani data kompleks dan interaksi fitur dengan baik.
  - Kekurangan: Lebih lambat dan kurang transparan dibanding logistic regression.

3. XGBoost:
  - Kelebihan: Akurasi tinggi, cepat, dan dapat menangani overfitting dengan baik.
  - Kekurangan: Lebih kompleks untuk dituning.

## Evaluation
### Metrik Evaluasi:
- Accuracy: Persentase prediksi yang benar.
- Precision: Proporsi prediksi churn yang benar dari semua yang diprediksi churn.
- Recall: Proporsi churn yang benar-benar terdeteksi oleh model.
- F1-score: Harmonik antara precision dan recall.

Formula F1-score: F1 = 2*(Precision*Recall/Precision+Recall)

Hasil Evaluasi:
- XGBoost memberikan performa terbaik berdasarkan F1-score.
- Random Forest sedikit di bawah XGBoost namun lebih baik dari Logistic Regression.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
