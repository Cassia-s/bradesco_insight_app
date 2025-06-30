import streamlit as st
import pandas as pd
import joblib
import os
import json
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime

# --- Configurações do Google Cloud ---
project_id = "bradesco-insight"
dataset_id = "bradesco"
location = "southamerica-east1"

@st.cache_resource
def get_bigquery_client():
    st.info("Conectando ao BigQuery usando Streamlit Secrets.")
    key_dict = json.loads(st.secrets["gcp_key"]["json"])
    credentials = service_account.Credentials.from_service_account_info(key_dict)
    return bigquery.Client(credentials=credentials, project=key_dict["project_id"])

client = get_bigquery_client()

@st.cache_resource
def load_models():
    model_dir = "models"
    try:
        model_fraud_detection = joblib.load(os.path.join(model_dir, "fraud_detection_model.joblib"))
        kmeans_model = joblib.load(os.path.join(model_dir, "kmeans_segmentation_model.joblib"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        fraud_encoders = joblib.load(os.path.join(model_dir, "fraud_label_encoders.joblib"))
        customer_encoders = joblib.load(os.path.join(model_dir, "customer_label_encoders.joblib"))
        fraud_features_names = joblib.load(os.path.join(model_dir, "fraud_features_names.joblib"))

        return model_fraud_detection, kmeans_model, scaler, fraud_encoders, customer_encoders, fraud_features_names
    except FileNotFoundError as e:
        st.error(f"Erro: Modelos não encontrados na pasta '{model_dir}'. Detalhe: {e}.")
        st.stop()

model_fraud_detection, kmeans_model, scaler, fraud_encoders, customer_encoders, fraud_features_names = load_models()

@st.cache_data(ttl=3600)
def get_customers_data():
    query = f"SELECT * FROM `{project_id}.{dataset_id}.customers_segmented`"
    return client.query(query).to_dataframe()

@st.cache_data(ttl=3600)
def get_transactions_data():
    query = f"SELECT * FROM `{project_id}.{dataset_id}.transactions_with_fraud_score`"
    return client.query(query).to_dataframe()

customers_df = get_customers_data()
transactions_df = get_transactions_data()

st.set_page_config(layout="wide", page_title="Bradesco Insight: Detecção de Fraudes e Segmentação de Clientes")
st.title("🛡️ Bradesco Insight: Detecção de Fraudes e Segmentação de Clientes")
st.markdown("Explore riscos, identifique comportamentos suspeitos e conheça o perfil dos clientes. Ferramenta preditiva com foco em segurança e inteligência de negócio.")

st.sidebar.title("🔎 Navegação")
page = st.sidebar.radio("Escolha uma opção:", ["Visão Geral do Dashboard", "Análise de Transação (Simulação)", "Perfil do Cliente"])

if page == "Visão Geral do Dashboard":
    st.header("📊 Visão Geral do Sistema")
    st.divider()
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔐 Análise de Fraudes")
        fraud_counts = transactions_df['is_fraudulent'].value_counts()
        total_transacoes = len(transactions_df)
        trans_fraud = fraud_counts.get(True, 0)
        taxa_fraude = (trans_fraud / total_transacoes * 100) if total_transacoes > 0 else 0

        st.metric("💳 Total de Transações", value=total_transacoes)
        st.metric("🚨 Transações Fraudulentas", value=trans_fraud, delta=f"{taxa_fraude:.1f}%")
        st.metric("📈 Média da Pontuação de Fraude", value=f"{transactions_df['fraud_score'].mean():.4f}")

        st.markdown("#### Distribuição da Pontuação de Fraude")
        st.bar_chart(transactions_df['fraud_score'].value_counts(bins=10).sort_index())

        top_merchant_fraud = transactions_df[transactions_df['is_fraudulent'] == True]['merchant_category'].value_counts().idxmax()
        st.markdown(f"📌 Categoria mais associada à fraude: **{top_merchant_fraud}**")

    with col2:
        st.subheader("👥 Segmentação de Clientes")
        segment_counts = customers_df['customer_segment'].value_counts().sort_index()
        st.metric("🧑‍💼 Clientes Segmentados", value=len(customers_df))
        st.metric("🧮 Média de Transações/Cliente", value=f"{total_transacoes / len(customers_df):.1f}")

        st.markdown("#### Distribuição por Segmento")
        st.bar_chart(segment_counts)

        st.markdown("#### Médias por Segmento")
        features_for_segmentation = [
            'age', 'income', 'avg_balance', 'num_accounts', 'total_spent',
            'avg_transaction_amount', 'num_transactions', 'total_fraud_score',
            'num_fraudulent_transactions', 'num_products_held'
        ]
        if 'marital_status_encoded' in customers_df.columns:
            features_for_segmentation.append('marital_status_encoded')
        if 'profession_encoded' in customers_df.columns:
            features_for_segmentation.append('profession_encoded')

        existing = [f for f in features_for_segmentation if f in customers_df.columns]
        segment_analysis = customers_df.groupby('customer_segment')[existing].mean().round(2)

        st.dataframe(segment_analysis)

    st.divider()
    st.subheader("🔎 Top 10 Transações com Maior Risco de Fraude")
    top10 = transactions_df.sort_values(by='fraud_score', ascending=False).head(10)
    st.dataframe(top10[['transaction_date', 'amount', 'merchant_category', 'fraud_score', 'is_fraudulent']])


elif page == "Análise de Transação (Simulação)":
    st.header("🔍 Simulador de Transações")
    st.markdown("Simule uma transação para verificar a probabilidade de fraude com base nas características fornecidas.")

    top_professions = customers_df['profession'].value_counts().head(20).index.tolist()
    top_categories = transactions_df['merchant_category'].value_counts().head(20).index.tolist()
    top_locations = [loc for loc in transactions_df['location'].value_counts().index.tolist() if loc and isinstance(loc, str) and len(loc) >= 3 and 'unknown' not in loc.lower()][:20]

    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("💵 Valor da Transação", min_value=0.0, value=1000.0)
            transaction_type = st.selectbox("Tipo de Transação", transactions_df['transaction_type'].unique())
            merchant_category = st.selectbox("Categoria do Comerciante", top_categories)
        with col2:
            location = st.selectbox("Localização", top_locations)
            device_info = st.selectbox("Dispositivo", transactions_df['device_info'].unique())
            transaction_hour = st.slider("Hora da Transação", 0, 23, 15)
        with col3:
            transaction_day_of_week = st.slider("Dia da Semana", 0, 6, 2)
            income = st.number_input("Renda do Cliente", min_value=0.0, value=5000.0)
            balance = st.number_input("Saldo da Conta", min_value=0.0, value=20000.0)
            customer_age_at_transaction = st.number_input("Idade do Cliente", min_value=0, value=30)
            marital_status = st.selectbox("Estado Civil", customers_df['marital_status'].unique())
            profession = st.selectbox("Profissão", top_professions)

        submitted = st.form_submit_button("🔎 Analisar Risco de Fraude")

        if submitted:
            input_data = pd.DataFrame([{
                'amount': amount,
                'income': income,
                'balance': balance,
                'transaction_hour': transaction_hour,
                'transaction_day_of_week': transaction_day_of_week,
                'customer_age_at_transaction': customer_age_at_transaction,
                'transaction_type': transaction_type,
                'merchant_category': merchant_category,
                'location': location,
                'device_info': device_info,
                'account_type': 'Unknown',
                'marital_status': marital_status,
                'profession': profession,
                'customer_segment': 0
            }])

            input_data['amount_per_income'] = input_data['amount'] / (input_data['income'] + 1e-6)

            for col, encoder in fraud_encoders.items():
                if col in input_data.columns:
                    try:
                        input_data[f'{col}_encoded'] = encoder.transform([input_data[col].iloc[0]])[0]
                    except ValueError:
                        input_data[f'{col}_encoded'] = -1
                else:
                    input_data[f'{col}_encoded'] = -1

            X = input_data[[f for f in fraud_features_names if f in input_data.columns]]
            X = X[fraud_features_names]

            score = model_fraud_detection.predict_proba(X)[:, 1][0]

            st.subheader("📋 Resultado da Avaliação")
            st.metric("Pontuação de Fraude", f"{score:.4f}")
            st.progress(score)

            if score >= 0.8:
                st.error("🔴 ALTO RISCO DE FRAUDE!")
            elif score >= 0.4:
                st.warning("🟠 MÉDIO RISCO DE FRAUDE")
            else:
                st.success("🟢 BAIXO RISCO DE FRAUDE")

            st.divider()
            st.markdown("### 🧠 Por que essa transação foi considerada suspeita?")
            for col in X.columns:
                valor = input_data[col].values[0]
                st.write(f"- **{col}**: {valor}")

if page == "Perfil do Cliente":
    st.header("👤 Perfil do Cliente")
    customer_id_input = st.text_input("ID do Cliente", value="1")

    if customer_id_input:
        try:
            customer_id = int(customer_id_input)
            customer_profile = customers_df[customers_df['customer_id'] == customer_id]

            if not customer_profile.empty:
                st.subheader(f"🧾 Dados do Cliente ID: {customer_id}")
                st.dataframe(
                    customer_profile.drop(columns=['customer_id'])
                    .T.rename(columns={customer_profile.index[0]: 'Valor'})
                    .astype(str)
                )

                segment = customer_profile['customer_segment'].iloc[0]
                st.write(f"Segmento: `{segment}`")

                features_for_segmentation = [
                    'age', 'income', 'avg_balance', 'num_accounts', 'total_spent',
                    'avg_transaction_amount', 'num_transactions', 'total_fraud_score',
                    'num_fraudulent_transactions', 'num_products_held'
                ]
                if 'marital_status_encoded' in customers_df.columns:
                    features_for_segmentation.append('marital_status_encoded')
                if 'profession_encoded' in customers_df.columns:
                    features_for_segmentation.append('profession_encoded')

                existing = [f for f in features_for_segmentation if f in customers_df.columns]
                segment_analysis = customers_df.groupby('customer_segment')[existing].mean()

                if segment in segment_analysis.index:
                    segment_data = (
                        segment_analysis.loc[segment].to_frame().T.rename(columns={
                            'age': 'Idade Média', 'income': 'Renda Média', 'avg_balance': 'Saldo Médio',
                            'num_accounts': 'Nº de Contas', 'total_spent': 'Total Gasto',
                            'avg_transaction_amount': 'Valor Médio Transação', 'num_transactions': 'Nº Transações',
                            'total_fraud_score': 'Pontuação Fraude Total',
                            'num_fraudulent_transactions': 'Nº Transações Fraudulentas',
                            'num_products_held': 'Nº Produtos', 'marital_status_encoded': 'Status Civil Codificado',
                            'profession_encoded': 'Profissão Codificada'
                        })
                    )
                    st.dataframe(segment_data.round(2))

                st.subheader("📂 Últimas Transações do Cliente")
                customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id].sort_values(by='transaction_date', ascending=False).head(10)
                if not customer_transactions.empty:
                    st.dataframe(customer_transactions[['transaction_date', 'amount', 'transaction_type', 'merchant_category', 'fraud_score', 'is_fraudulent']])
                else:
                    st.write("Nenhuma transação encontrada para este cliente.")
            else:
                st.warning("Cliente não encontrado. Verifique o ID.")
        except ValueError:
            st.warning("ID inválido. Insira um número inteiro.")
