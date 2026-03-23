# 📊 Predição de Risco de Defasagem Escolar

Este projeto tem como objetivo prever o **risco de aumento de defasagem escolar** de alunos com base em indicadores educacionais e comportamentais.

A solução foi desenvolvida utilizando técnicas de **Machine Learning** e disponibilizada através de uma interface interativa com **Streamlit**.


Links: https://projeto5-vncnrhztajiqb5pkpsmfyk.streamlit.app/
---

## 🚀 Objetivo

Identificar alunos com maior probabilidade de **piora na defasagem escolar no próximo ano**, permitindo ações preventivas e mais assertivas.

---

## 🧠 Como funciona o modelo

O modelo foi treinado para prever se a **defasagem do aluno irá aumentar no próximo período**, considerando o histórico dos dados.

### 🎯 Variável alvo (target)

```python
Risco_Defasagem = Defasagem_proximo_ano > Defasagem_atual
```

Ou seja:

* `1` → O aluno piora
* `0` → O aluno mantém ou melhora

---

## 📊 Features utilizadas

O modelo utiliza as seguintes variáveis:

### 📚 Indicadores educacionais

* **IAN** — Indicador de Adequação ao Nível
* **IDA** — Indicador de Aprendizagem
* **IEG** — Indicador de Engajamento
* **IAA** — Indicador de Auto Avaliação
* **IPS** — Indicador Psicossocial
* **IPV** — Indicador de Ponto de Virada
* **INDE** — Índice de Desenvolvimento Educacional
* **IPP** — Indicador Psicopedagógico

### 🏫 Dados complementares

* **Fase** — Etapa escolar do aluno
* **Defasagem** — Nível atual de defasagem
* **Gênero** — Representado como variável binária
* **Pedra** — Classificação categórica do aluno

---

## ⚙️ Tecnologias utilizadas

* Python
* Pandas
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Random Forest
* Streamlit

---

## 🔄 Pipeline do projeto

1. **Tratamento dos dados**
2. **Criação do target (defasagem futura)**
3. **Balanceamento com SMOTE**
4. **Treinamento com Random Forest**
5. **Ajuste de threshold**
6. **Deploy com Streamlit**

---

## 🧪 Avaliação do modelo

O modelo foi avaliado utilizando:

* Accuracy
* Precision
* Recall
* F1-score
* AUC-ROC

Além disso, foram realizados testes com diferentes **thresholds de decisão** para otimizar a sensibilidade do modelo.

---

## 🖥️ Interface (App)

O projeto conta com uma interface interativa onde é possível:

✔️ Inserir dados do aluno
✔️ Simular cenários
✔️ Visualizar a probabilidade de risco
✔️ Entender o comportamento do modelo

---

## ▶️ Como executar o projeto

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o app

```bash
streamlit run app.py
```

### 3. Acessar no navegador

```
http://localhost:8501
```

---

## 📁 Estrutura do projeto

```
├── app.py
├── modelo_pipeline.pkl
├── notebook_modelo.py
├── requirements.txt
├── data/
└── README.md
```

---

## 🧠 Principais aprendizados

* A definição do **target é crítica** para o sucesso do modelo
* Dados desbalanceados impactam fortemente a performance
* O alinhamento entre **treino e produção** é essencial
* Nem sempre modelos mais complexos são melhores na prática

---

## 📌 Próximos passos

* Implementar explicabilidade do modelo (SHAP)
* Melhorar a interface do usuário
* Integrar com bases reais
* Criar versão com histórico do aluno

---

## 👨‍💻 Autores

Projeto desenvolvido por 
**Franco Colmán**
**Hugo Duran**

---

## ⭐ Considerações finais

Este projeto demonstra como transformar dados educacionais em insights acionáveis, apoiando decisões mais inteligentes e proativas no acompanhamento de alunos.

---
