import pandas as pd
import numpy as np
import joblib
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# =========================
# 1. Carregar dados
# =========================
df = pd.read_csv("df_combinado.csv")

# =========================
# 2. Ordenação temporal
# =========================
df = df.sort_values(by=["RA", "AnoRef"])

# =========================
# 3. Feature Engineering
# =========================
df["Defasagem_proximo_ano"] = df.groupby("RA")["Defasagem"].shift(-1)

df["Delta_Defasagem"] = (
    df["Defasagem_proximo_ano"] - df["Defasagem"]
)

df["Risco_Defasagem"] = (df["Delta_Defasagem"] >= 1).astype("Int64")

# =========================
# 4. Limpeza
# =========================
df_modelo = df.dropna(subset=["Risco_Defasagem"]).copy()
df_modelo["Risco_Defasagem"] = df_modelo["Risco_Defasagem"].astype(int)

df_modelo["AnoRef"] = pd.to_numeric(df_modelo["AnoRef"], errors="coerce")

# =========================
# 5. Remover colunas
# =========================
cols_remover = [
    "INDE",
    "RA",
    "Defasagem_proximo_ano",
    "Delta_Defasagem"
]

df_modelo = df_modelo.drop(columns=cols_remover, errors="ignore")

# =========================
# 6. Split ROBUSTO
# =========================
anos = sorted(df_modelo["AnoRef"].dropna().unique())

print("Anos encontrados:", anos)

if len(anos) >= 2:
    ano_teste = anos[-1]
    train = df_modelo[df_modelo["AnoRef"] < ano_teste].copy()
    test = df_modelo[df_modelo["AnoRef"] == ano_teste].copy()
else:
    print("⚠️ Apenas um ano disponível — usando split aleatório")
    train, test = train_test_split(
        df_modelo,
        test_size=0.3,
        random_state=42,
        stratify=df_modelo["Risco_Defasagem"]
    )

print("Train:", train.shape)
print("Test:", test.shape)

if train.shape[0] == 0 or test.shape[0] == 0:
    raise ValueError("❌ Split inválido")

# =========================
# 7. Separar X e y
# =========================
y_train = train["Risco_Defasagem"]
y_test = test["Risco_Defasagem"]

X_train = train.drop(columns=["Risco_Defasagem"])
X_test = test.drop(columns=["Risco_Defasagem"])

# =========================
# 8. Pipeline
# =========================
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    ))
])

# =========================
# 9. Treinar
# =========================
model.fit(X_train, y_train)

# =========================
# 10. Avaliação
# =========================
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.3).astype(int)

print("\n=== Avaliação ===")
print("AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# =========================
# 11. Salvar modelo + versão
# =========================
joblib.dump({
    "model": model,
    "sklearn_version": sklearn.__version__
}, "modelo_pipeline.pkl")

print(f"\n✅ Modelo salvo! Versão sklearn: {sklearn.__version__}")







# antigo++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# # =========================================
# # 1. IMPORTS E DRIVE
# # =========================================

# import pandas as pd
# import numpy as np


# # =========================================
# # 2. CARREGAR BASE
# # =========================================
# arquivo = fr'C:\Users\Franco\Documents\Fiap\Datathon Fase 5\data\Modelo_preditivo\MODELO_PREDITIVO_2\df_combinado.csv'

# df_analise = pd.read_csv(arquivo)

# print("Pré-visualização da base:")
# print(df_analise.head())

# # =========================================
# # 3. FEATURE ENGINEERING (TARGET FUTURO)
# # =========================================
# df_risco = df_analise.copy()

# # Ordenação temporal
# df_risco = df_risco.sort_values(by=['RA', 'AnoRef'])

# # Defasagem do próximo ano
# df_risco['Defasagem_proximo_ano'] = (
#     df_risco.groupby('RA')['Defasagem'].shift(-1)
# )

# # Conversões
# df_risco['Defasagem'] = df_risco['Defasagem'].astype(float)
# df_risco['Defasagem_proximo_ano'] = df_risco['Defasagem_proximo_ano'].astype(float)

# # Target: piora de defasagem
# df_risco['Risco_Defasagem'] = (
#     df_risco['Defasagem_proximo_ano'] > df_risco['Defasagem']
# ).astype('Int64')

# print("\nFeature engineering concluída!")
# print(df_risco.head())

# # =========================================
# # 4. PREPARAÇÃO PARA MODELO
# # =========================================
# df_modelo = df_risco.copy()

# # Remover nulos do target
# df_modelo = df_modelo.dropna(subset=['Risco_Defasagem']).copy()
# df_modelo['Risco_Defasagem'] = df_modelo['Risco_Defasagem'].astype(int)

# # Separar target
# y = df_modelo['Risco_Defasagem']

# # Separar features
# X = df_modelo.drop(columns=[
#     'RA',
#     'AnoRef',
#     'Defasagem_proximo_ano',
#     'Risco_Defasagem',
#     'IAN_grupo'
# ], errors='ignore')

# # One-hot encoding automático
# X = pd.get_dummies(X, drop_first=True)
# print(fr'o x aqui {X}')
# print(X.describe())
# print("\nDados preparados!")
# print("\nDistribuição do target:")
# print(y.value_counts(normalize=True))

# # =========================================
# # 5. TRAIN TEST SPLIT
# # =========================================
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.3,
#     random_state=42,
#     stratify=y
# )

# print("\nSplit concluído!")

# # =========================================
# # 6. BALANCEAMENTO COM SMOTE
# # =========================================
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state=42)

# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# print("\nDistribuição após SMOTE:")
# print(y_train_res.value_counts())

# # =========================================
# # 7. TREINAMENTO DO MODELO
# # =========================================
# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(
#     random_state=42,
#     class_weight='balanced',
#     n_estimators=200,
#     max_depth=10
# )

# model.fit(X_train_res, y_train_res)

# print("\nModelo treinado!")

# # =========================================
# # 8. AVALIAÇÃO DO MODELO
# # =========================================
# from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# # Probabilidades
# y_proba = model.predict_proba(X_test)[:, 1]

# # Threshold ajustado
# threshold = 0.2
# y_pred = (y_proba >= threshold).astype(int)

# # Métricas
# accuracy = accuracy_score(y_test, y_pred)
# auc = roc_auc_score(y_test, y_proba)

# print("\n=== Avaliação do Modelo ===\n")
# print(f"Acurácia: {accuracy:.4f}")
# print(f"AUC-ROC: {auc:.4f}")
# print(f"Threshold: {threshold}\n")

# print("Classification Report:\n")
# print(classification_report(y_test, y_pred))

# # =========================================
# # 9. TESTE DE THRESHOLDS
# # =========================================
# print("\n=== Teste de Threshold ===")

# for t in [0.1, 0.2, 0.3]:
#     y_pred_t = (y_proba >= t).astype(int)

#     print(f"\nThreshold: {t}")
#     print(f"Acurácia: {accuracy_score(y_test, y_pred_t):.4f}")
#     print(classification_report(y_test, y_pred_t))

# # =========================================
# # 10. FEATURE IMPORTANCE
# # =========================================
# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Importancia': model.feature_importances_
# }).sort_values(by='Importancia', ascending=False)

# print("\n=== Top 10 Variáveis Mais Importantes ===\n")
# print(feature_importance.head(10))

# # =========================================
# # 11. SALVAR MODELO (PARA O APP)
# # =========================================
# import joblib

# joblib.dump(model, fr'MODELO_PREDITIVO_2\modelo_rf_v2.pkl')
# joblib.dump(X.columns.tolist(), fr'MODELO_PREDITIVO_2\features_v2.pkl')

# print("\nModelo e features salvos com sucesso!")