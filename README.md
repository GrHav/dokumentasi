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
Dataset: Telco Customer Churn - [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Dataset ini berisi informasi pelanggan dari perusahaan telekomunikasi fiktif, termasuk demografi, layanan yang digunakan, informasi kontrak, dan label churn.

Jumlah data: 7043 baris dan 21 kolom.

### Variabel-variabel pada Telco Customer dataset adalah sebagai berikut:

#### Variabel utama:

- Demografi: gender, SeniorCitizen, Partner, Dependents
- Layanan: PhoneService, InternetService, StreamingTV, StreamingMovies, OnlineSecurity, DeviceProtection, TechSupport, multiplelines, onlinebackup, paperlessbilling.
- Keuangan & kontrak: MonthlyCharges, TotalCharges, Contract, PaymentMethod, tenure
- Target: Churn (Yes/No)

### Penjelasan Variabel:

- Pelanggan yang berhenti berlangganan dalam bulan terakhir – kolom ini disebut Churn
- Layanan yang telah didaftarkan oleh setiap pelanggan – Phone, multiplelines, internet, online security, online protection, device protection, tech support, serta streaming TV dan movies
- Informasi akun pelanggan – berapa lama mereka menjadi customer, contract, payment method, paperless billing, monthly charges, dan total charges
- Informasi demografis tentang pelanggan – gender, age range, dan apakah mereka memiliki partners dan dependents

#### Exploratory Data Analysis (EDA):

- Visualisasi distribusi target churn menunjukkan dataset tidak seimbang.
- Histogram fitur numerik menunjukkan distribusi yang bervariasi.

## Data Preparation

Langkah-langkah preprocessing:
- Konversi 'TotalCharges' menjadi data numerik
- Proses dropna() setelah konversi 'TotalCharges' untuk menghapus nilai yang kosong
- Encoding kolom target 'Churn' dari 'Yes'/'No' menjadi nilai numerik (0/1).
- Penghapusan kolom 'customerID'.
- Melakukan encode dengan pd.get_dummies() untuk mengubah fitur kategorikal menjadi numerik
- Melakukan drop kolom churn karena churn akan digunakan sebagai variabel y
- Melakukan split data dengan test size sebesar 20%
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
  - Cara Kerja
      - Logistic Regression adalah model klasifikasi linier yang digunakan untuk memprediksi probabilitas dari kelas target.
      
      Fungsi utama
      - z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
      
      Nilai z kemudian dipetakan ke rentang 0–1 menggunakan fungsi sigmoid:
      - P(y=1|x) = 1 / (1 + exp(-z))
      Probabilitas diklasifikasikan ke dalam label 0 atau 1 berdasarkan threshold (biasanya 0.5).
    - max_iter adalah parameter yang menentukan jumlah maksimum iterasi yang digunakan oleh solver dalam proses optimasi model Logistic Regression.
  
2. Random Forest:
  - Kelebihan: Menangani data kompleks dan interaksi fitur dengan baik.
  - Kekurangan: Lebih lambat dan kurang transparan dibanding logistic regression.
  - Cara Kerja
    - Random Forest adalah algoritma ensemble berbasis banyak pohon keputusan.
    - Setiap pohon dilatih menggunakan subset acak dari data (bootstrap sampling).
    - Prediksi akhir berdasarkan voting mayoritas (klasifikasi) atau rata-rata (regresi) dari semua pohon.
    - random_state memastikan bahwa proses pelatihan model menjadi reproducible (hasil yang konsisten tiap kali dijalankan).


3. XGBoost:
  - Kelebihan: Akurasi tinggi, cepat, dan dapat menangani overfitting dengan baik.
  - Kekurangan: Lebih kompleks untuk dituning.
  - Cara Kerja
    - XGBoost adalah algoritma boosting berbasis pohon yang melatih pohon berturut-turut untuk memperbaiki kesalahan dari model sebelumnya.
    - Menggunakan residual error dari prediksi sebelumnya sebagai target untuk pohon berikutnya.
    - Menggabungkan boosting dan regularisasi untuk meningkatkan performa dan menghindari overfitting.
      - Logloss (logarithmic loss) adalah metrik yang mengukur seberapa dekat prediksi probabilitas dengan label sebenarnya.
      - random_state memastikan bahwa proses pelatihan model menjadi reproducible (hasil yang konsisten tiap kali dijalankan).
      - use_label_encoder=false untuk mematikan penggunaan label encoder karena sudah diencode sebelumnya jadi tidak perlu melakukan encode lagi 

## Evaluation
### Metrik Evaluasi:
- Accuracy: Persentase prediksi yang benar.
- Precision: Proporsi prediksi churn yang benar dari semua yang diprediksi churn.
- Recall: Proporsi churn yang benar-benar terdeteksi oleh model.
- F1-score: Harmonik antara precision dan recall.

Formula F1-score: F1 = 2*(Precision*Recall/Precision+Recall)

Hasil Evaluasi:
Model                    F1-Score  Accuracy  Precision  Recall 
1. Logistic Regression : 0.5635    0.7875    0.6206     0.5160
2. Random Forest : 0.5410    0.7853       0.6268      0.4759
3. XGBoost : 0.5139     0.7633      0.5659       0.4706
- Model Logistic Regression memiliki F1-score tertinggi (0.563504), diikuti oleh Random Forest (0.541033), dan kemudian XGBoost (0.513869)
- Model Logistic Regression memiliki accuracy tertinggi (0.7875), diikuti oleh Random Forest (0.7853), dan kemudian XGBoost (0.7633)
- Model Random Forest memiliki precision tertinggi (0.6268), diikuti oleh Logistic Regression (0.6206), dan kemudian XGBoost (0.5659)
- Model Logistic Regression memiliki recall tertinggi (0.5160), diikuti oleh Random Forest (0.4759), dan kemudian XGBoost(0.4706)
- Dikarenakan F1-score tertinggi adalah Logistic Regression maka model yang lebih baik digunakan dalam kasus ini adalah Logistic Regression

### Menjawab Problem Statements
1. Pelanggan yang berisiko tinggi untuk melakukan churn dapat diidentifikasi dengan membangun model klasifikasi berbasis machine learning. Dalam proyek ini, dilakukan beberapa tahapan:
- Pra-pemrosesan data untuk membersihkan dan mengubah data ke format numerik.
- Pelatihan model dengan menggunakan algoritma seperti Logistic Regression, Random Forest, dan XGBoost.
- Evaluasi model dilakukan menggunakan F1-score, karena data bersifat tidak seimbang (imbalanced).
Model dengan performa terbaik (Logistic Regression dalam kasus ini, dengan F1-score 0.5635) digunakan untuk memprediksi probabilitas churn setiap pelanggan. Pelanggan dengan probabilitas churn lebih tinggi dari ambang batas tertentu (misalnya 0.5) dikategorikan sebagai pelanggan berisiko tinggi, dan dapat menjadi target dari program retensi perusahaan.

2. Berdasarkan analisis feature importance dan koefisien dari model:
- Contract: Pelanggan dengan kontrak bulanan lebih mungkin untuk churn dibandingkan pelanggan dengan kontrak jangka panjang.
- Tenure (masa berlangganan): Pelanggan dengan masa langganan yang lebih pendek cenderung lebih berisiko untuk churn.
- MonthlyCharges: Tagihan bulanan yang tinggi berkorelasi dengan kemungkinan churn yang lebih besar.
- InternetService dan TechSupport: Pelanggan dengan layanan internet DSL/cable dan yang tidak memiliki dukungan teknis juga menunjukkan kecenderungan lebih tinggi untuk berhenti.
Model Random Forest dan XGBoost secara eksplisit juga menunjukkan fitur-fitur di atas sebagai fitur penting dalam menentukan label churn.

## Referensi
1. Amin, A., Anwar, S., Adnan, A., et al. (2019). Customer churn prediction in the telecommunication sector using a rough set approach. Neural Computing and Applications.
2. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition). O'Reilly Media.
3. Kaggle Dataset. (n.d.). Telco Customer Churn. Retrieved from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
