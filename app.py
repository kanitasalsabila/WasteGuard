import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet

st.set_page_config(layout="wide")
st.title("📁 Upload Data atau Input Manual untuk Prediksi & Klasifikasi Risiko Limbah Radioaktif")

# =======================
# Load Model
# =======================
rf_model = joblib.load("random_forest_esg_model.pkl")
prophet_models = joblib.load("all_prophet_models.pkl")  # dict dengan key: 'depth', 'ph', 'tds'

# =======================
# Tab Layout
# =======================
tab1, tab2 = st.tabs(["📤 Upload CSV", "✍️ Input Manual"])

# =======================
# Tab 1: Upload CSV
# =======================
with tab1:
    uploaded_file = st.file_uploader("Unggah file CSV (harus mengandung kolom 'ds', 'depth', 'ph', 'tds')", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Validasi kolom wajib
        expected_cols = {"ds", "depth", "ph", "tds"}
        if not expected_cols.issubset(df.columns):
            st.error(f"Kolom wajib tidak lengkap. Diperlukan: {expected_cols}")
        else:
            df["ds"] = pd.to_datetime(df["ds"])
            st.info("📌 Data berhasil dibaca. Menjalankan prediksi...")

            forecast_results = []

            for i, row in df.iterrows():
                date_input = pd.DataFrame({"ds": [row["ds"]]})
                f_depth = prophet_models["depth"].predict(date_input)["yhat"].values[0]
                f_ph = prophet_models["ph"].predict(date_input)["yhat"].values[0]
                f_tds = prophet_models["tds"].predict(date_input)["yhat"].values[0]

                forecast_results.append({
                    "forecast_depth": round(min(f_depth, 20), 2),
                    "forecast_ph": round(max(f_ph, 0), 2),
                    "forecast_tds": round(f_tds, 2),
                })

            forecast_df = pd.DataFrame(forecast_results)
            final_df = pd.concat([df.reset_index(drop=True), forecast_df], axis=1)

            # Klasifikasi risiko
            classify_input = final_df[["forecast_depth", "forecast_ph", "forecast_tds"]]
            final_df["Prediksi_Risiko"] = rf_model.predict(classify_input)

            st.success("✅ Forecast & klasifikasi selesai!")
            st.subheader("📄 Tabel Hasil:")
            st.dataframe(final_df, use_container_width=True)

            csv = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Hasil sebagai CSV",
                data=csv,
                file_name="hasil_forecast_klasifikasi.csv",
                mime="text/csv"
            )

# =======================
# Tab 2: Input Manual
# =======================
with tab2:
    st.write("Masukkan nilai parameter secara manual untuk klasifikasi risiko (tanpa forecasting)")

    depth = st.number_input("Depth", min_value=0.0, max_value=20.0, step=0.1)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
    tds = st.number_input("TDS", min_value=0.0, max_value=5000.0, step=1.0)

    if st.button("Prediksi Risiko"):
        input_df = pd.DataFrame({
            "forecast_depth": [depth],
            "forecast_ph": [ph],
            "forecast_tds": [tds],
        })

        pred_risk = rf_model.predict(input_df)[0]

        st.success(f"✅ Prediksi Risiko ESG: {pred_risk}")
        st.dataframe(input_df.assign(Prediksi_Risiko=pred_risk))

st.caption("Model oleh Kanita Salsabila Dwi Irmanti || WasteGuard")