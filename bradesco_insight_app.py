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
st.markdown("Bem-vindo ao sistema de análise preditiva do Bradesco. Explore insights sobre transações e clientes.")

st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha uma opção:", ["Visão Geral do Dashboard", "Análise de Transação (Simulação)", "Perfil do Cliente"])

if page == "Visão Geral do Dashboard":
    st.header("Visão Geral do Sistema")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Análise de Fraudes")
        fraud_counts = transactions_df['is_fraudulent'].value_counts()
        st.metric(label="Total de Transações", value=len(transactions_df))
        st.metric(label="Transações Fraudulentas Identificadas", value=fraud_counts.get(True, 0))
        st.metric(label="Pontuação Média de Fraude", value=f"{transactions_df['fraud_score'].mean():.4f}")
        st.bar_chart(transactions_df['fraud_score'].value_counts(bins=10).sort_index())

    with col2:
        st.subheader("Segmentação de Clientes")
        segment_counts = customers_df['customer_segment'].value_counts().sort_index()
        st.metric(label="Total de Clientes Segmentados", value=len(customers_df))
        st.bar_chart(segment_counts)

    st.subheader("Transações Fraudulentas Identificadas (Top 10 por Pontuação)")
    fraudulent_transactions_display = transactions_df[transactions_df['is_fraudulent'] == True].sort_values(by='fraud_score', ascending=False)
    if not fraudulent_transactions_display.empty:
        st.dataframe(fraudulent_transactions_display[['transaction_date', 'amount', 'transaction_type', 'merchant_category', 'fraud_score', 'is_fraudulent']].head(10))
    else:
        st.info("Nenhuma transação fraudulenta explícita encontrada. Exibindo as maiores pontuações.")
        st.dataframe(transactions_df.sort_values(by='fraud_score', ascending=False)[['transaction_date', 'amount', 'transaction_type', 'merchant_category', 'fraud_score']].head(10))

elif page == "Análise de Transação (Simulação)":
    st.header("Simulador de Detecção de Fraudes")
    top_professions = customers_df['profession'].value_counts().head(20).index.tolist()
    top_merchant_categories = transactions_df['merchant_category'].value_counts().head(15).index.tolist()
    top_locations = transactions_df['location'].value_counts().head(15).index.tolist()

    with st.form("transaction_form"):
        st.subheader("Dados da Transação e do Cliente")
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Valor da Transação (R$)", min_value=0.0, value=1000.0, format="%.2f")
            transaction_type = st.selectbox("Tipo de Transação", transactions_df['transaction_type'].unique())
            merchant_category = st.selectbox("Categoria do Comerciante", top_merchant_categories)
        with col2:
            location = st.selectbox("Localização", top_locations)
            device_info = st.selectbox("Informações do Dispositivo", transactions_df['device_info'].unique())
            transaction_hour = st.slider("Hora da Transação (0-23h)", 0, 23, 15)
        with col3:
            transaction_day_of_week = st.slider("Dia da Semana (0=Segunda, 6=Domingo)", 0, 6, 2)
            income = st.number_input("Renda do Cliente (R$)", min_value=0.0, value=5000.0, format="%.2f")
            balance = st.number_input("Saldo da Conta (R$)", min_value=0.0, value=20000.0, format="%.2f")
            customer_age = st.number_input("Idade do Cliente", min_value=0, value=30)
            marital_status = st.selectbox("Estado Civil", customers_df['marital_status'].unique())
            profession = st.selectbox("Profissão", top_professions)

        submitted = st.form_submit_button("Analisar Risco de Fraude")

        if submitted:
            new_tx = pd.DataFrame([{
                'amount': amount,
                'income': income,
                'balance': balance,
                'transaction_hour': transaction_hour,
                'transaction_day_of_week': transaction_day_of_week,
                'customer_age_at_transaction': customer_age,
                'transaction_type': transaction_type,
                'merchant_category': merchant_category,
                'location': location,
                'device_info': device_info,
                'account_type': 'Unknown',
                'marital_status': marital_status,
                'profession': profession,
                'customer_segment': 0
            }])

            new_tx['amount_per_income'] = new_tx['amount'] / (new_tx['income'] + 1e-6)

            for col, encoder in fraud_encoders.items():
                if col in new_tx.columns:
                    try:
                        new_tx[f'{col}_encoded'] = encoder.transform([new_tx[col].iloc[0]])[0]
                    except ValueError:
                        new_tx[f'{col}_encoded'] = -1
                else:
                    new_tx[f'{col}_encoded'] = -1

            X_tx = new_tx[[f for f in fraud_features_names if f in new_tx.columns]]

            if not all(f in X_tx.columns for f in fraud_features_names):
                missing = [f for f in fraud_features_names if f not in X_tx.columns]
                st.error(f"Faltam colunas: {missing}")
                st.stop()

            X_tx = X_tx[fraud_features_names]
            score = model_fraud_detection.predict_proba(X_tx)[:, 1][0]

            st.subheader("Resultado da Análise:")
            st.write(f"**Pontuação de Fraude:** `{score:.4f}`")
            if score >= 0.8:
                st.error("🔴 ALTO RISCO DE FRAUDE")
            elif score >= 0.4:
                st.warning("🟠 MÉDIO RISCO DE FRAUDE")
            else:
                st.success("🟢 BAIXO RISCO DE FRAUDE")

elif page == "Perfil do Cliente":
    st.header("Consulta de Perfil e Segmento do Cliente")
    customer_id_input = st.text_input("ID do Cliente (Ex: 1, 5, 10)", value="1")

    if customer_id_input:
        try:
            customer_id = int(customer_id_input)
            customer_profile = customers_df[customers_df['customer_id'] == customer_id]

            if not customer_profile.empty:
                st.subheader(f"Perfil Detalhado do Cliente ID: {customer_id}")
                st.dataframe(customer_profile.drop(columns=['customer_id']).T.rename(columns={customer_profile.index[0]: 'Valor'}).astype(str))
                segment = customer_profile['customer_segment'].iloc[0]
                st.write(f"**Segmento do Cliente:** `{segment}`")

                st.subheader(f"Características Médias do Segmento {segment}:")
                features_for_segmentation = [
                    'age', 'income', 'avg_balance', 'num_accounts', 'total_spent',
                    'avg_transaction_amount', 'num_transactions', 'total_fraud_score',
                    'num_fraudulent_transactions', 'num_products_held',
                ]
                if 'marital_status_encoded' in customers_df.columns:
                    features_for_segmentation.append('marital_status_encoded')
                if 'profession_encoded' in customers_df.columns:
                    features_for_segmentation.append('profession_encoded')

                existing = [f for f in features_for_segmentation if f in customers_df.columns]
                segment_analysis = customers_df.groupby('customer_segment')[existing].mean()

                if segment in segment_analysis.index:
                    st.dataframe(segment_analysis.loc[[segment]].rename(columns={
                        'age': 'Idade Média',
                        'income': 'Renda Média',
                        'avg_balance': 'Saldo Médio',
                        'num_accounts': 'Nº de Contas',
                        'total_spent': 'Total Gasto',
                        'avg_transaction_amount': 'Valor Médio Transação',
                        'num_transactions': 'Nº de Transações',
                        'total_fraud_score': 'Pontuação Fraude Total',
                        'num_fraudulent_transactions': 'Nº Transações Fraudulentas',
                        'num_products_held': 'Nº Produtos',
                        'marital_status_encoded': 'Status Civil (Codificado)',
                        'profession_encoded': 'Profissão (Codificado)'
                    }).round(2))

                st.subheader("Últimas Transações do Cliente:")
                tx = transactions_df[transactions_df['customer_id'] == customer_id].sort_values(by='transaction_date', ascending=False).head(10)
                if not tx.empty:
                    st.dataframe(tx[['transaction_date', 'amount', 'transaction_type', 'merchant_category', 'fraud_score', 'is_fraudulent']])
                else:
                    st.info("Nenhuma transação encontrada.")
            else:
                st.warning("Cliente não encontrado. Verifique o ID.")
        except ValueError:
            st.warning("ID inválido. Use um número inteiro.")
