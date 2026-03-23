import streamlit as st
import pandas as pd
import joblib
import os
import sklearn

st.set_page_config(page_title="Risco de Defasagem", layout="wide")

st.title("📊 Predição de Risco de Defasagem")

# =========================
# Load model seguro
# =========================
@st.cache_resource
def load_model():
    model_path = "modelo_pipeline.pkl"

    if not os.path.exists(model_path):
        st.error("❌ Modelo não encontrado. Execute o modelo.py primeiro.")
        st.stop()

    try:
        data = joblib.load(model_path)

        model = data["model"]
        version_train = data.get("sklearn_version", "desconhecida")
        version_now = sklearn.__version__

        if version_train != version_now:
            st.warning(f"⚠️ Versão sklearn diferente: treino={version_train} | atual={version_now}")

        return model

    except Exception as e:
        st.error("❌ Erro ao carregar modelo")
        st.exception(e)
        st.stop()

model = load_model()

# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("df_combinado.csv")

df = load_data()

# =========================
# Visualização
# =========================
st.subheader("📂 Dados")
st.dataframe(df.head())

# =========================
# Simulação
# =========================
st.subheader("🎯 Simulação de Aluno")

input_data = {}

# Numéricas
for col in df.select_dtypes(exclude="object").columns:
    if col not in ["RA", "AnooRef"]:  # removidos
        input_data[col] = st.number_input(col, value=float(df[col].mean()))

# Categóricas
for col in df.select_dtypes(include="object").columns:
    input_data[col] = st.selectbox(col, df[col].dropna().unique())

input_df = pd.DataFrame([input_data])

# =========================
# Previsão
# =========================
if st.button("Prever"):
    try:
        proba = model.predict_proba(input_df)[0][1]

        st.metric("Probabilidade de Risco", f"{proba:.2%}")

        # threshold fixo (mesmo comportamento padrão anterior)
        if proba >= 0.3:
            st.error("⚠️ Alto risco")
        else:
            st.success("✅ Baixo risco")

    except Exception as e:
        st.error("Erro na previsão")
        st.exception(e)

# =========================
# Rodapé
# =========================
st.markdown("---")
st.markdown("Modelo com pipeline robusto + validação temporal")