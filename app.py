"""
app.py — Gradio Standalone Application
VCT Champions 2025 — Match Outcome Predictor

Cara menjalankan:
    pip install gradio scikit-learn joblib
    python app.py
"""

import os
import sys
import json
import warnings
import numpy as np
import gradio as gr
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

# ─── Pastikan working directory benar ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

MODEL_PATH   = os.path.join(BASE_DIR, "models", "best_classifier.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEAT_PATH    = os.path.join(BASE_DIR, "models", "feature_cols.json")

# ─── Validasi keberadaan model ─────────────────────────────────────────────────
def check_models():
    missing = []
    for path in [MODEL_PATH, SCALER_PATH, FEAT_PATH]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("⚠️  Model belum ditemukan! Silakan jalankan notebook 'supervised_learning_valorant.ipynb' terlebih dahulu.")
        print("   File yang belum ada:")
        for m in missing:
            print(f"   - {m}")
        return False
    return True

# ─── Load model ────────────────────────────────────────────────────────────────
if check_models():
    _model  = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    with open(FEAT_PATH) as f:
        _feat = json.load(f)
    MODEL_LOADED = True
    MODEL_NAME   = type(_model).__name__
    print(f"✅ Model '{MODEL_NAME}' berhasil dimuat!")
else:
    MODEL_LOADED = False
    MODEL_NAME   = "N/A"
    _model       = None
    _scaler      = None
    _feat        = ['avg_kda','avg_acs','avg_adr','avg_hs','avg_kast',
                    'total_fk','total_fd','avg_rating']

NEEDS_SCALE = MODEL_NAME in ['LogisticRegression', 'SVC']


# ─── Fungsi Prediksi ───────────────────────────────────────────────────────────
def predict_match(avg_kda, avg_acs, avg_adr, avg_hs, avg_kast,
                  total_fk, total_fd, avg_rating):
    """
    Prediksi hasil pertandingan berdasarkan statistik tim.
    
    Parameters
    ----------
    avg_kda     : Rata-rata KDA tim (Kill+Assist / Death)
    avg_acs     : Rata-rata Average Combat Score per pemain
    avg_adr     : Rata-rata Average Damage per Round
    avg_hs      : Rata-rata persentase headshot tim
    avg_kast    : Rata-rata KAST % (Kill, Assist, Survive, Trade)
    total_fk    : Total First Kills dalam game
    total_fd    : Total First Deaths dalam game
    avg_rating  : Rata-rata rating pemain (dari VLR.gg)
    
    Returns
    -------
    result_text : Teks hasil prediksi beserta probabilitas
    result_label: Label singkat untuk display
    prob_win    : Probabilitas menang (0–100)
    prob_loss   : Probabilitas kalah (0–100)
    """
    if not MODEL_LOADED:
        return ("⚠️ Model belum dimuat. Jalankan notebook terlebih dahulu!",
                "Error", 0, 100)

    X_input = np.array([[avg_kda, avg_acs, avg_adr, avg_hs, avg_kast,
                          total_fk, total_fd, avg_rating]], dtype=float)
    if NEEDS_SCALE:
        X_input = _scaler.transform(X_input)

    pred    = _model.predict(X_input)[0]
    prob    = _model.predict_proba(X_input)[0]
    prob_win  = round(float(prob[1]) * 100, 2)
    prob_loss = round(float(prob[0]) * 100, 2)

    if pred == 1:
        result_text = (
            f"🏆  PREDIKSI: MENANG (WIN)\n\n"
            f"{'─'*40}\n"
            f"  Peluang Menang  :  {prob_win:.2f}%\n"
            f"  Peluang Kalah   :  {prob_loss:.2f}%\n"
            f"{'─'*40}\n"
            f"  Model           :  {MODEL_NAME}\n\n"
            f"  Interpretasi:\n"
            f"  Tim ini memiliki statistik yang kuat.\n"
            f"  Tingkat kepercayaan model: {'Tinggi' if prob_win > 80 else 'Sedang' if prob_win > 60 else 'Rendah'}."
        )
        label = "🏆 MENANG"
    else:
        result_text = (
            f"❌  PREDIKSI: KALAH (LOSS)\n\n"
            f"{'─'*40}\n"
            f"  Peluang Menang  :  {prob_win:.2f}%\n"
            f"  Peluang Kalah   :  {prob_loss:.2f}%\n"
            f"{'─'*40}\n"
            f"  Model           :  {MODEL_NAME}\n\n"
            f"  Interpretasi:\n"
            f"  Statistik tim perlu ditingkatkan.\n"
            f"  Tingkat kepercayaan model: {'Tinggi' if prob_loss > 80 else 'Sedang' if prob_loss > 60 else 'Rendah'}."
        )
        label = "❌ KALAH"

    return result_text, label, prob_win, prob_loss


# ─── Tema CSS Kustom ───────────────────────────────────────────────────────────
CUSTOM_CSS = """
body { background-color: #0F1923; }
.gradio-container { font-family: 'Segoe UI', sans-serif; }
#title-banner {
    background: linear-gradient(135deg, #FF4655 0%, #0F1923 60%, #00B4D8 100%);
    padding: 20px; border-radius: 12px; margin-bottom: 16px;
    text-align: center;
}
.predict-btn { background: #FF4655 !important; color: white !important; font-size: 1.1em !important; }
.result-box textarea { font-family: monospace !important; font-size: 0.95em !important; }
"""


# ─── Build Gradio UI ───────────────────────────────────────────────────────────
def build_interface():
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="red",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
        title="🎮 VCT Match Predictor"
    ) as app:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="title-banner">
            <h1 style="color:white; margin:0; font-size:2em;">🎮 VCT Champions 2025</h1>
            <h2 style="color:#FF4655; margin:4px 0;">Match Outcome Predictor</h2>
            <p style="color:#aaa; margin:0; font-size:0.9em;">
                Prediksi kemenangan tim berdasarkan statistik pertandingan &nbsp;|&nbsp;
                Powered by Machine Learning &amp; Dataset VLR.gg via Kaggle
            </p>
        </div>
        """)

        # ── Input Section ────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Statistik Ofensif")
                avg_acs = gr.Slider(
                    minimum=80, maximum=400, value=200, step=1,
                    label="⚡ Avg Combat Score (ACS)",
                    info="Rata-rata ACS seluruh pemain dalam tim"
                )
                avg_adr = gr.Slider(
                    minimum=60, maximum=260, value=135, step=1,
                    label="💥 Avg Damage per Round (ADR)",
                    info="Rata-rata kerusakan per ronde tim"
                )
                avg_kda = gr.Slider(
                    minimum=0.3, maximum=5.0, value=1.3, step=0.01,
                    label="⚔️  KDA Ratio Rata-rata",
                    info="(Kill + Assist) / Death, rata-rata tim"
                )
                avg_hs = gr.Slider(
                    minimum=0, maximum=70, value=25, step=0.5,
                    label="🎯 Headshot % Rata-rata",
                    info="Persentase headshot rata-rata tim"
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🛡️ Statistik Defensif & Lainnya")
                avg_kast = gr.Slider(
                    minimum=30, maximum=100, value=70, step=0.5,
                    label="📈 KAST % Rata-rata",
                    info="Kill/Assist/Survive/Trade — konsistensi pemain"
                )
                total_fk = gr.Slider(
                    minimum=0, maximum=30, value=5, step=1,
                    label="⚡ Total First Kills per Game",
                    info="Jumlah first kill yang diraih tim dalam satu game"
                )
                total_fd = gr.Slider(
                    minimum=0, maximum=30, value=5, step=1,
                    label="💀 Total First Deaths per Game",
                    info="Jumlah kematian pertama yang diderita tim"
                )
                avg_rating = gr.Slider(
                    minimum=0.3, maximum=3.0, value=1.1, step=0.01,
                    label="⭐ Rating Rata-rata Tim",
                    info="Rating keseluruhan pemain (skala VLR.gg)"
                )

        # ── Predict Button ───────────────────────────────────────────────────
        predict_btn = gr.Button(
            "🔮  Prediksi Hasil Pertandingan!",
            variant="primary",
            size="lg",
            elem_classes=["predict-btn"]
        )

        # ── Output ──────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                result_text = gr.Textbox(
                    label="📋 Hasil Prediksi Lengkap",
                    lines=10,
                    interactive=False,
                    elem_classes=["result-box"]
                )
            with gr.Column(scale=1):
                result_label = gr.Label(label="🏁 Keputusan Akhir")
                with gr.Row():
                    prob_win  = gr.Number(label="✅ Peluang Menang (%)", precision=2)
                    prob_loss = gr.Number(label="❌ Peluang Kalah (%)", precision=2)

        # ── Bind ─────────────────────────────────────────────────────────────
        predict_btn.click(
            fn=predict_match,
            inputs=[avg_kda, avg_acs, avg_adr, avg_hs, avg_kast,
                    total_fk, total_fd, avg_rating],
            outputs=[result_text, result_label, prob_win, prob_loss]
        )

        # ── Info / Footer ─────────────────────────────────────────────────────
        with gr.Accordion("ℹ️ Panduan Penggunaan & Deskripsi Fitur", open=False):
            gr.Markdown("""
            ## 📌 Cara Menggunakan
            1. **Atur slider** di sebelah kiri dan kanan sesuai statistik tim yang ingin Anda evaluasi.
            2. Klik tombol **"Prediksi Hasil Pertandingan!"**
            3. Lihat hasil prediksi beserta probabilitasnya di bagian bawah.

            ## 📊 Deskripsi Fitur
            | Fitur | Deskripsi |
            |-------|-----------|
            | **ACS** | Average Combat Score — skor kontribusi per ronde per pemain |
            | **ADR** | Average Damage per Round — kerusakan rata-rata per ronde |
            | **KDA** | (Kills + Assists) / Deaths — efisiensi dalam duel |
            | **HS%** | Headshot percentage — akurasi dan presisi tembakan ke kepala |
            | **KAST%** | Kill/Assist/Survive/Trade — konsistensi pemain tiap ronde |
            | **First Kills** | Jumlah kemenangan duel pertama per game |
            | **First Deaths** | Jumlah kekalahan duel pertama per game |
            | **Rating** | Skor performa keseluruhan (scale VLR.gg) |

            ## 📂 Sumber Dataset
            - **Dataset**: VCT Champions 2025 Player & Match Statistics
            - **Sumber**: [Kaggle – Valorant Champion Tour Data](https://www.kaggle.com/datasets/kierru/valorant-vct-champions-2025-dataset)
            - **Provider Asli**: [VLR.gg](https://www.vlr.gg)
            """)

        gr.HTML("""
        <div style="text-align:center; margin-top:16px; color:#666; font-size:0.8em;">
            Tugas Besar Fundamen Sains Data &nbsp;|&nbsp; Competitive Gaming &amp; Esports ML Project
        </div>
        """)

    return app


# ─── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   🎮 VCT Champions 2025 — Match Outcome Predictor")
    print("="*60)
    interface = build_interface()
    interface.launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        show_error=True
    )
