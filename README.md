# 🎮 Final Project: Competitive Gaming & Esports — Machine Learning

## Fundamen Sains Data | Tugas Besar
---

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan dua pendekatan Machine Learning pada dataset pertandingan profesional **Valorant Champions Tour (VCT) 2025**, dengan tema **Competitive Gaming & Esports**.

| Studi Kasus | Pendekatan | Tujuan |
|------------|-----------|--------|
| **1** | Supervised Learning | Prediksi Kemenangan (Win/Loss) berdasarkan statistik tim |
| **2** | Unsupervised Learning | Segmentasi & Pengelompokan Senjata Valorant berdasarkan Performa (Clustering) |

---

## 📊 Sumber Dataset (Real / Publik)

### Dataset 1 — VCT Champions 2025 Match Data
- **Platform**: [Kaggle](https://www.kaggle.com/datasets/thedevastator/valorant-champion-tour-2021-2026-data)
- **Konten**: Data statistik pertandingan tim (skor, win/loss, economy) dan performa pemain individual (ACS, ADR, rating, KAST, headshot %) dari VCT Champions 2025

### Dataset 2 — Valorant Weapons Statistics
- **Platform**: [Kaggle – Valorant Weapons Dataset](https://www.kaggle.com/datasets/aarishmughal/valorant-weapons-stats-latest)
- **File**: `valorant_weapons.csv`
- **Konten**: Spesifikasi teknis dan statistik performa 20 senjata Valorant (harga, fire rate, amunisi, dan damage head/body/legs di berbagai jarak)

---

## 🗂️ Struktur Direktori

```
Tubes-FSD/
│
├── data/
│   ├── vct_champs_2025/              ← Dataset utama (VCT 2025 - Studi Kasus 1)
│   │   ├── stats.csv                 ← Performa individual pemain
│   │   ├── score.csv                 ← Skor & hasil pertandingan
│   │   ├── economy.csv               ← Data ekonomi tim
│   │   ├── player_id.csv             ← ID pemain
│   │   ├── team_id.csv               ← ID tim
│   │   ├── agent_id.csv              ← ID agent
│   │   ├── match_id.csv              ← ID match
│   │   ├── pick_ban.csv              ← Data pick/ban agent
│   │   ├── 1v1.csv                   ← Data duel 1v1
│   │   └── counter_kill.csv          ← Data counter kill
│   │
│   └── weapons/                      ← Dataset senjata (Studi Kasus 2)
│       └── valorant_weapons.csv
│
├── models/                           ← Model hasil training (dibuat otomatis)
│   ├── best_classifier.pkl           ← Model klasifikasi terbaik
│   ├── scaler.pkl                    ← StandardScaler untuk klasifikasi
│   ├── feature_cols.json             ← Daftar fitur klasifikasi
│   ├── kmeans_weapons.pkl            ← Model K-Means clustering senjata
│   ├── scaler_weapons.pkl            ← Scaler untuk clustering senjata
│   └── pca_weapons.pkl               ← PCA model clustering senjata
│
├── supervised_learning_valorant.ipynb    ← 📓 Notebook Studi Kasus 1 (Match Outcome Prediction)
├── unsupervised_learning_valorant.ipynb  ← 📓 Notebook Studi Kasus 2 (Weapon Clustering)
├── app.py                                ← 🖥️ Gradio App (standalone)
└── README.md                             ← Dokumentasi ini
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn gradio joblib jupyter
```

### 2. Jalankan Notebook Supervised Learning

```bash
jupyter notebook supervised_learning_valorant.ipynb
```

> Jalankan semua sel (`Run All`). Model akan tersimpan otomatis ke folder `models/`.

### 3. Jalankan Notebook Unsupervised Learning

```bash
jupyter notebook unsupervised_learning_valorant.ipynb
```

### 4. Jalankan Gradio App (Standalone)

> ⚠️ **Pastikan notebook Supervised Learning sudah dijalankan terlebih dahulu** agar model tersedia.

```bash
python app.py
```

Buka browser di: [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 📌 Studi Kasus 1 — Supervised Learning

### Objektif
Memprediksi apakah sebuah tim akan **Menang** atau **Kalah** dalam pertandingan VCT 2025 berdasarkan statistik rata-rata tim.

### Fitur Input (X)
| Fitur | Deskripsi |
|-------|-----------|
| `avg_kda` | Rata-rata KDA (Kill+Assist/Death) seluruh pemain |
| `avg_acs` | Rata-rata Average Combat Score |
| `avg_adr` | Rata-rata Average Damage per Round |
| `avg_hs`  | Rata-rata Headshot % tim |
| `avg_kast` | Rata-rata KAST % (Kill/Assist/Survive/Trade) |
| `total_fk` | Total First Kills per game |
| `total_fd` | Total First Deaths per game |
| `avg_rating` | Rating rata-rata pemain |

### Target (y)
- `1` = Menang (Win)
- `0` = Kalah (Loss)

### Algoritma yang Dibandingkan
- ✅ **Logistic Regression**
- ✅ **SVM (RBF Kernel)**
- ✅ **Random Forest** ← *(terbaik)*
- ✅ **Gradient Boosting**

### Evaluasi
- 5-Fold Stratified Cross Validation
- Confusion Matrix, Classification Report
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Curve
- Feature Importance

---

## 📌 Studi Kasus 2 — Unsupervised Learning

### Objektif
Mengelompokkan 20 senjata Valorant ke dalam cluster archetype berdasarkan profil performa (harga, fire rate, amunisi, dan damage di berbagai jarak) menggunakan **K-Means Clustering**.

### Fitur yang Digunakan
`Price`, `FireRate`, `Ammo`, `ReserveAmmo`, `Damage_Head_0-30m`, `Damage_Body_0-30m`, `Damage_Legs_0-30m`, `Damage_Head_30-50m`, `Damage_Body_30-50m`, `Damage_Legs_30-50m`, dll.

### Hasil Clustering (3 Cluster Archetype Senjata)
| Cluster | Archetype | Karakteristik & Contoh Senjata |
|---------|-----------|--------------------------------|
| **Cluster 0** | ⭐ **Precision / High-Damage** | Head damage sangat tinggi, harga mahal, fire rate rendah (*Operator, Vandal, Guardian*) |
| **Cluster 1** | ⚡ **Rapid Fire / Aggressive** | Fire rate sangat tinggi, ammo besar, damage per peluru rendah (*Spectre, Stinger, Frenzy, Ares, Odin*) |
| **Cluster 2** | ⚖️ **Balanced / All-Round** | Keseimbangan antara damage, harga, dan fire rate (*Ghost, Sheriff, Classic, Phantom*) |

### Metode Evaluasi Clustering
- **Elbow Method** (Inertia) — Menentukan jumlah cluster $k=3$ optimal
- **Silhouette Score** — Mengukur tingkat pemisahan antar cluster
- **Davies-Bouldin Index** — Mengukur rasio separasi dan dispersi cluster
- **PCA 2D Visualization** — Reduksi dimensi untuk visualisasi sebaran 20 senjata dalam grafik 2D

---

## 🖥️ Gradio Application

Antarmuka interaktif yang memungkinkan pengguna memasukkan statistik tim secara manual dan mendapatkan prediksi **Win/Loss** secara real-time beserta probabilitasnya.

**Fitur UI:**
- 8 slider input (ACS, ADR, KDA, HS%, KAST%, First Kills, First Deaths, Rating)
- Output: Teks prediksi lengkap + Label keputusan + Probabilitas Win/Loss
- Accordion panduan penggunaan & deskripsi fitur
- Desain bertema Valorant (warna merah-biru-cyan)

---

## 👤 Informasi Proyek

| Detail | Keterangan |
|--------|-----------|
| **Mata Kuliah** | Fundamen Sains Data |
| **Tema** | Competitive Gaming & Esports |
| **Dataset Utama** | Valorant VCT Champions 2025 | Valorant Weapon Stats |
| **Sumber Dataset** | [Kaggle 1] (https://www.kaggle.com/datasets/kierru/valorant-vct-champions-2025-dataset)   [Kaggle 2](https://www.kaggle.com/datasets/aarishmughal/valorant-weapons-stats-latest)) |
| **Teknologi** | Python, scikit-learn, Pandas, Matplotlib, Seaborn, Gradio |
---

*Dataset dikumpulkan dari VLR.gg dan didistribusikan melalui platform Kaggle untuk keperluan edukasi dan penelitian.*
