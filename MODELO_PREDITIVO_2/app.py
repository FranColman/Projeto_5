# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # =========================================
# # CARREGAR MODELO E FEATURES
# # =========================================
# model = joblib.load('modelo_rf_v2.pkl')
# features = joblib.load('features_v2.pkl')

# # =========================================
# # INTERFACE
# # =========================================
# st.title("📊 Predição de Risco de Defasagem")

# st.write("Preencha os dados do aluno:")

# # =========================================
# # DADOS PRINCIPAIS
# # =========================================
# Fase = st.number_input("Fase do aluno", 1, 10, 7)

# Defasagem = st.number_input("Defasagem atual", -2.0, 5.0, 0.0, step=0.1)

# IAN = st.number_input("IAN (Indicador de Adequação ao Nível)", 0.0, 10.0, 5.0, step=0.1)
# IDA = st.number_input("IDA (Indicador de Aprendizagem)", 0.0, 10.0, 5.0, step=0.1)
# IEG = st.number_input("IEG (Indicador de Engajamento)", 0.0, 10.0, 5.0, step=0.1)
# IAA = st.number_input("IAA (Indicador de Aproveitamento Acadêmico)", 0.0, 10.0, 5.0, step=0.1)
# IPS = st.number_input("IPS (Indicador Psicossocial)", 0.0, 10.0, 5.0, step=0.1)
# IPV = st.number_input("IPV (Indicador de Ponto de Virada)", 0.0, 10.0, 5.0, step=0.1)
# INDE = st.number_input("INDE (Índice de Desenvolvimento Educacional)", 0.0, 10.0, 5.0, step=0.1)
# IPP = st.number_input("IPP (Indicador de Permanência/Participação)", 0.0, 10.0, 5.0, step=0.1)

# # =========================================
# # CATEGÓRICAS
# # =========================================
# genero = st.selectbox("Gênero", ["Feminino", "Masculino"])

# pedra = st.selectbox("Pedra", ["Quartzo", "Ágata", "Topázio"])

# # =========================================
# # BOTÃO
# # =========================================
# if st.button("🔍 Prever risco"):

#     # =========================================
#     # DATAFRAME BASE
#     # =========================================
#     data = pd.DataFrame(columns=features)
#     data.loc[0] = 0

#     # =========================================
#     # NUMÉRICAS
#     # =========================================
#     data['Fase'] = Fase
#     data['Defasagem'] = Defasagem
#     data['IAN'] = IAN
#     data['IDA'] = IDA
#     data['IEG'] = IEG
#     data['IAA'] = IAA
#     data['IPS'] = IPS
#     data['IPV'] = IPV
#     data['INDE'] = INDE
#     data['IPP'] = IPP

#     # =========================================
#     # GÊNERO
#     # =========================================
#     if 'Genero_Masculino' in data.columns:
C:\Users\Franco\Documents\Fiap\Datathon Fase 5\data\Modelo_preditivo\MODELO_PREDITIVO_2\DATATHON-ANALISE.ipynb
#         data['Genero_Masculino'] = 1 if genero == "Masculino" else 0

#     # =========================================
#     # PEDRA (ONE HOT)
#     # =========================================
#     for col in data.columns:
#         if "Pedra_" in col:
#             data[col] = 0

#     if pedra == "Quartzo" and 'Pedra_Quartzo' in data.columns:
#         data['Pedra_Quartzo'] = 1

#     if pedra == "Topázio" and 'Pedra_Topázio' in data.columns:
#         data['Pedra_Topázio'] = 1

#     if pedra == "Ágata" and 'Pedra_Ágata' in data.columns:
#         data['Pedra_Ágata'] = 1

#     # =========================================
#     # GARANTIR ORDEM DAS FEATURES
#     # =========================================
#     data = data[features]

#     # =========================================
#     # PREDIÇÃO
#     # =========================================
#     prob = model.predict_proba(data)[0][1]

#     threshold = 0.2

#     # =========================================
#     # RESULTADO
#     # =========================================
#     st.subheader("Resultado")

#     st.write(f"Probabilidade de risco: **{prob:.2%}**")

#     if prob >= threshold:
#         st.error("⚠️ Risco de aumento de defasagem")
#     else:
#         st.success("✅ Sem risco relevante")

#     # =========================================
#     # DEBUG (OPCIONAL)
#     # =========================================
#     with st.expander("🔍 Debug"):
#         st.write("Input enviado ao modelo:")
#         st.dataframe(data)
#         st.write("Features esperadas:")
#         st.write(features)




# ________________________________________________________________

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