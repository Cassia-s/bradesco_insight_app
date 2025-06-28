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

# Visão Geral e Simulação omitidos aqui por clareza
# Inclua normalmente os blocos anteriores se quiser toda a lógica completa

if page == "Perfil do Cliente":
    st.header("Consulta de Perfil e Segmento do Cliente")
    st.write("Insira o ID de um cliente para ver seu perfil detalhado e segmento de cliente.")

    customer_id_input = st.text_input("ID do Cliente (Ex: 1, 5, 10)", value="1")

    if customer_id_input:
        try:
            customer_id = int(customer_id_input)
            customer_profile = customers_df[customers_df['customer_id'] == customer_id]

            if not customer_profile.empty:
                st.subheader(f"Perfil Detalhado do Cliente ID: {customer_id}")
                st.dataframe(
                    customer_profile.drop(columns=['customer_id'])
                    .T.rename(columns={customer_profile.index[0]: 'Valor'})
                    .astype(str)
                )

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

                existing_features = [f for f in features_for_segmentation if f in customers_df.columns]
                segment_analysis = customers_df.groupby('customer_segment')[existing_features].mean()

                if segment in segment_analysis.index:
                    segment_data = (
                        segment_analysis.loc[segment]
                        .to_frame()
                        .T.rename(columns={
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
                        })
                    )
                    st.dataframe(segment_data.round(2))
                else:
                    st.write("Não foi possível encontrar as características médias para este segmento.")

                st.subheader("Últimas Transações do Cliente:")
                customer_transactions = transactions_df[
                    transactions_df['customer_id'] == customer_id
                ].sort_values(by='transaction_date', ascending=False).head(10)

                if not customer_transactions.empty:
                    st.dataframe(
                        customer_transactions[[
                            'transaction_date', 'amount', 'transaction_type',
                            'merchant_category', 'fraud_score', 'is_fraudulent'
                        ]]
                    )
                else:
                    st.write("Nenhuma transação encontrada para este cliente.")
            else:
                st.warning("Cliente não encontrado. Por favor, verifique o ID.")
        except ValueError:
            st.warning("ID do Cliente inválido. Por favor, insira um número inteiro.")
