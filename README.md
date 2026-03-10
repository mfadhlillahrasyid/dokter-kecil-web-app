# Dokter Kecil — Pediatric Disease Predictor

## Stack
- **Frontend**: HTML + Tailwind CSS + Vanilla JS
- **Backend**: PHP (Naive Bayes classifier)
- **Algorithm**: Naive Bayes dengan Laplace Smoothing

## File Structure
```
dokter-kecil/
├── index.html          # Main app UI
├── api.php             # Naive Bayes engine + REST API
├── disease_data.csv    # Training data (8000 rows) — upload sendiri
└── model_cache.json    # Auto-generated setelah training
```

## Setup

### 1. Taruh di server PHP (XAMPP / Laragon / server)
```
htdocs/dokter-kecil/
├── index.html
├── api.php
└── disease_data.csv   ← Rename file CSV kamu ke ini
```

### 2. Training model (sekali saja)
Buka browser: `http://localhost/dokter-kecil/api.php?action=train`

Response sukses:
```json
{"success": true, "samples": 8000, "features": [...]}
```

Model akan di-cache di `model_cache.json` — tidak perlu training ulang setiap request.

### 3. Akses app
`http://localhost/dokter-kecil/index.html`

## CSV Format
Header yang diharapkan:
```
fever,cough,runny_nose,sore_throat,rash,vomiting,diarrhea,ear_pain,breathing_difficulty,headache,appetite_loss,mosquito_bite,duration,label
```

Values:
- `fever`: none | mild | high | severe
- `duration`: short | long
- Symptom lain: Ya | Tidak
- `label`: nama penyakit (Dengue, Flu, Pneumonia, dll)

## Mode Demo
Jika `disease_data.csv` tidak ada dan `api.php` tidak bisa diakses,
app otomatis fallback ke **client-side rule-based prediction** (JS).

## API Endpoints
- `GET api.php?action=train` — Train & cache model dari CSV
- `POST api.php?action=predict` — Predict dari input JSON
  ```json
  {
    "symptoms": {
      "fever": "high",
      "cough": "Ya",
      "headache": "Ya"
    }
  }
  ```
