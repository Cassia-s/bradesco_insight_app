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

# Inicializar o cliente BigQuery
@st.cache_resource
def get_bigquery_client():
    st.info("Conectando ao BigQuery usando Streamlit Secrets.")
    
    # Este bloco de código agora está corretamente indentado DENTRO da função.
    # Lembre-se que seu secrets.toml no Streamlit Cloud DEVE estar assim:
    # [gcp_key]
    # json = """
    # {
    #   "type": "service_account",
    #   "project_id": "bradesco-insight",
    #   "private_key_id": "...",
    #   "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
    #   "client_email": "...",
    #   "client_id": "...",
    #   "auth_uri": "...",
    #   "token_uri": "...",
    #   "auth_provider_x509_cert_url": "...",
    #   "client_x509_cert_url": "...",
    #   "universe_domain": "googleapis.com"
    # }
    # """
    # (Ou seja, a chave "gcp_key" dentro de st.secrets deve conter o JSON COMPLETO como uma string)

    key_dict = json.loads(st.secrets["gcp_key"]["json"])
    credentials = service_account.Credentials.from_service_account_info(key_dict)
    return bigquery.Client(credentials=credentials, project=key_dict["project_id"])

# A chamada da função get_bigquery_client() deve estar FORA da definição da função,
# e sem indentação.
client = get_bigquery_client()

# --- Carregar Modelos e Transformadores ---
# Certifique-se de que a pasta 'models' está no mesmo diretório que este script
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
        st.error(f"Erro: Modelos não encontrados na pasta '{model_dir}'. Detalhe: {e}. Certifique-se de que você baixou a pasta 'models' do Colab e a colocou no mesmo diretório deste script. Para o deploy, a pasta 'models' deve estar no repositório GitHub.")
        st.stop()

model_fraud_detection, kmeans_model, scaler, fraud_encoders, customer_encoders, fraud_features_names = load_models()

# --- Funções para buscar dados do BigQuery ---
@st.cache_data(ttl=3600) # Cache por 1 hora
def get_customers_data():
    query = f"SELECT * FROM `{project_id}.{dataset_id}.customers_segmented`"
    return client.query(query).to_dataframe()

@st.cache_data(ttl=3600) # Cache por 1 hora
def get_transactions_data():
    query = f"SELECT * FROM `{project_id}.{dataset_id}.transactions_with_fraud_score`"
    return client.query(query).to_dataframe()

customers_df = get_customers_data()
transactions_df = get_transactions_data()

# --- Título do Aplicativo ---
st.set_page_config(layout="wide", page_title="Bradesco Insight: Detecção de Fraudes e Segmentação de Clientes")
st.title("🛡️ Bradesco Insight: Detecção de Fraudes e Segmentação de Clientes")
st.markdown("Bem-vindo ao sistema de análise preditiva do Bradesco. Explore insights sobre transações e clientes.")

# --- Sidebar para Navegação ---
st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha uma opção:", ["Visão Geral do Dashboard", "Análise de Transação (Simulação)", "Perfil do Cliente"])

# --- Visão Geral do Dashboard ---
if page == "Visão Geral do Dashboard":
    st.header("Visão Geral do Sistema")
    st.write("Aqui você pode ver os principais indicadores de fraude e a distribuição dos segmentos de clientes.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Análise de Fraudes")
        fraud_counts = transactions_df['is_fraudulent'].value_counts()
        st.metric(label="Total de Transações", value=len(transactions_df))
        st.metric(label="Transações Fraudulentas Identificadas", value=fraud_counts.get(True, 0)) # Usando .get para lidar com caso onde não há True
        st.metric(label="Pontuação Média de Fraude", value=f"{transactions_df['fraud_score'].mean():.4f}")

        st.write("### Distribuição da Pontuação de Fraude")
        st.bar_chart(transactions_df['fraud_score'].value_counts(bins=10).sort_index())
        st.markdown("Este gráfico mostra a frequência das transações em diferentes faixas de pontuação de fraude. Pontuações mais altas indicam maior risco. Um modelo ideal concentraria a maioria das fraudes nas faixas de pontuação mais altas.")

    with col2:
        st.subheader("Segmentação de Clientes")
        segment_counts = customers_df['customer_segment'].value_counts().sort_index()
        st.metric(label="Total de Clientes Segmentados", value=len(customers_df))
        st.write("### Distribuição de Clientes por Segmento")
        st.bar_chart(segment_counts)
        st.markdown("Este gráfico exibe a quantidade de clientes em cada segmento identificado pelo modelo de clusterização (K-Means). Cada segmento agrupa clientes com características similares.")

        # --- INÍCIO DA MELHORIA: Características Médias por Segmento ---
        st.subheader("Características Médias por Segmento")
        st.markdown("Esta tabela mostra as características médias de cada segmento de cliente. Valores como 'Status Civil (Codificado)' e 'Profissão (Codificado)' são médias dos valores numéricos atribuídos pelos LabelEncoders durante o pré-processamento.")

        # Definir as features que queremos analisar para cada segmento
        features_for_segmentation = [
            'age', 'income', 'avg_balance', 'num_accounts', 'total_spent',
            'avg_transaction_amount', 'num_transactions', 'total_fraud_score',
            'num_fraudulent_transactions', 'num_products_held',
        ]
        # Adicionar colunas codificadas se elas existirem no DataFrame customers_df
        if 'marital_status_encoded' in customers_df.columns:
            features_for_segmentation.append('marital_status_encoded')
        if 'profession_encoded' in customers_df.columns:
            features_for_segmentation.append('profession_encoded')

        # Filtrar apenas as features que realmente existem no DataFrame customers_df
        existing_features_for_segmentation = [f for f in features_for_segmentation if f in customers_df.columns]

        # Calcular as médias por segmento
        segment_analysis = customers_df.groupby('customer_segment')[existing_features_for_segmentation].mean()
        
        # Renomear as colunas para melhor visualização na interface do Streamlit
        segment_analysis_display = segment_analysis.rename(columns={
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
        
        # Exibir a tabela no Streamlit, arredondando para 2 casas decimais para clareza
        st.dataframe(segment_analysis_display.round(2))
        # --- FIM DA MELHORIA: Características Médias por Segmento ---

    st.subheader("Transações Fraudulentas Identificadas (Top 10 por Pontuação)")
    # Assume que 'is_fraudulent' é True/False baseada no seu dataset ou um limite definido.
    # Se 'is_fraudulent' não estiver no BigQuery, você pode criá-la aqui:
    # transactions_df['is_fraudulent'] = transactions_df['fraud_score'] >= 0.8 # Exemplo de limite

    fraudulent_transactions_display = transactions_df[transactions_df['is_fraudulent'] == True].sort_values(by='fraud_score', ascending=False)
    if not fraudulent_transactions_display.empty:
        st.dataframe(fraudulent_transactions_display[['transaction_date', 'amount', 'transaction_type', 'merchant_category', 'fraud_score', 'is_fraudulent']].head(10))
    else:
        st.info("Nenhuma transação marcada explicitamente como fraudulenta encontrada nos dados. Exibindo as 10 transações com maior pontuação de fraude geral.")
        st.dataframe(transactions_df.sort_values(by='fraud_score', ascending=False)[['transaction_date', 'amount', 'transaction_type', 'merchant_category', 'fraud_score', 'is_fraudulent']].head(10))


# --- Análise de Transação (Simulação) ---
elif page == "Análise de Transação (Simulação)":
    st.header("Simulador de Detecção de Fraudes")
    st.write("Insira os detalhes de uma transação para prever sua pontuação de fraude em tempo real.")

    # Limitar as opções dos selectboxes aos N mais frequentes
    # Ajuste os valores (20, 15) conforme achar melhor para a demo
    top_professions = customers_df['profession'].value_counts().head(20).index.tolist()
    top_merchant_categories = transactions_df['merchant_category'].value_counts().head(15).index.tolist()
    top_locations = transactions_df['location'].value_counts().head(15).index.tolist()
    
    # Adicionar "Outros" ou "Não Definido" para as opções, se desejar
    # Ex: if 'Outros' not in top_professions: top_professions.append('Outros')

    with st.form("transaction_form"):
        st.subheader("Dados da Transação e do Cliente")
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Valor da Transação (R$)", min_value=0.0, value=1000.0, format="%.2f")
            transaction_type = st.selectbox("Tipo de Transação", transactions_df['transaction_type'].unique())
            merchant_category = st.selectbox("Categoria do Comerciante", top_merchant_categories) # USAR A LISTA FILTRADA
        with col2:
            location = st.selectbox("Localização", top_locations) # USAR A LISTA FILTRADA
            device_info = st.selectbox("Informações do Dispositivo", transactions_df['device_info'].unique())
            transaction_hour = st.slider("Hora da Transação (0-23h)", 0, 23, 15)
        with col3:
            transaction_day_of_week = st.slider("Dia da Semana (0=Segunda, 6=Domingo)", 0, 6, 2)
            income = st.number_input("Renda do Cliente (R$)", min_value=0.0, value=5000.0, format="%.2f")
            balance = st.number_input("Saldo da Conta (R$)", min_value=0.0, value=20000.0, format="%.2f")
            customer_age = st.number_input("Idade do Cliente", min_value=0, value=30)
            marital_status = st.selectbox("Estado Civil", customers_df['marital_status'].unique())
            profession = st.selectbox("Profissão", top_professions) # USAR A LISTA FILTRADA

        submitted = st.form_submit_button("Analisar Risco de Fraude")

        if submitted:
            # Criar DataFrame para a nova transação
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
                'account_type': 'Unknown', # Placeholder se não tiver no input direto (assumindo que não é um campo de input)
                'marital_status': marital_status,
                'profession': profession,
                'customer_segment': 0 # Placeholder: o modelo de fraude foi treinado sem customer_segment. Se você re-treinar para incluir, precisará calcular o segmento aqui. Por agora, 0 é um placeholder.
            }])

            # Calcular amount_per_income
            new_tx['amount_per_income'] = new_tx['amount'] / (new_tx['income'] + 1e-6)

            # Codificar variáveis categóricas usando os encoders salvos
            for col, encoder in fraud_encoders.items():
                if col in new_tx.columns:
                    try:
                        # O reshape(-1, 1) é necessário para que encoder.transform aceite um único valor
                        # e retorna um array 1D, por isso o [0] no final
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
