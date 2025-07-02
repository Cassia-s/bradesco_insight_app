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
    df = client.query(query).to_dataframe()
    df['customer_id'] = df['customer_id'].astype(str)
    df['age'] = df['age'].round(0).astype(int)  # Garante idade inteira
    return df

@st.cache_data(ttl=3600)
def get_transactions_data():
    query = f"SELECT * FROM `{project_id}.{dataset_id}.transactions_with_fraud_score`"
    df = client.query(query).to_dataframe()
    df['customer_id'] = df['customer_id'].astype(str)
    df['account_id'] = df['account_id'].astype(str)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df = df.dropna(subset=['transaction_date']) # Remove transações com data inválida
    return df

customers_df = get_customers_data()
transactions_df = get_transactions_data()

st.set_page_config(layout="wide", page_title="Bradesco Insight: Detecção de Fraudes e Segmentação de Clientes")
st.title("🛡️ Bradesco Insight: Detecção de Fraudes e Segmentação de Clientes")
st.markdown("Explore riscos, identifique comportamentos suspeitos e conheça o perfil dos clientes. Ferramenta preditiva com foco em segurança e inteligência de negócio.")

st.sidebar.title("🔎 Navegação")
page = st.sidebar.radio("Escolha uma opção:", ["Visão Geral do Dashboard", "Análise de Transação (Simulação)", "Perfil do Cliente"])

# Mapeamento de features para nomes em português para exibição no simulador
feature_translation_map = {
    'amount': 'Valor da Transação',
    'income': 'Renda do Cliente',
    'balance': 'Saldo da Conta',
    'transaction_hour': 'Hora da Transação',
    'transaction_day_of_week': 'Dia da Semana (0=Segunda, 6=Domingo)',
    'customer_age_at_transaction': 'Idade do Cliente',
    'amount_per_income': 'Valor da Transação por Renda',
    'transaction_type_encoded': 'Tipo de Transação (Codificado)',
    'merchant_category_encoded': 'Categoria do Comerciante (Codificado)',
    'location_encoded': 'Localização (Codificado)',
    'device_info_encoded': 'Dispositivo (Codificado)',
    'account_type_encoded': 'Tipo de Conta (Codificado)',
    'marital_status_encoded': 'Estado Civil (Codificado)',
    'profession_encoded': 'Profissão (Codificado)',
    'customer_segment': 'Segmento do Cliente (Placeholder)'
}

if page == "Visão Geral do Dashboard":
    st.header("📊 Visão Geral do Sistema")
    st.divider()
    
    # Filtros na lateral
    with st.sidebar:
        st.subheader("Filtros do Dashboard")
        min_date, max_date = transactions_df['transaction_date'].min(), transactions_df['transaction_date'].max()
        
        # Exibe o range de datas disponível
        st.info(f"**Datas disponíveis para filtro:**\n"
                f"De: {min_date.date().strftime('%d/%m/%Y')}\n"
                f"Até: {max_date.date().strftime('%d/%m/%Y')}")

        date_range = st.date_input("Filtrar por período:", [min_date.date(), max_date.date()])

        # Garante que os segmentos exibidos são 1, 2, 3 (se o Colab foi atualizado)
        segmentos = customers_df['customer_segment'].sort_values().unique().tolist()
        selected_segmentos = st.multiselect("Filtrar por segmento:", segmentos, default=segmentos)

    # Aplica filtros
    filtered_tx = transactions_df.copy()
    filtered_tx = filtered_tx[
        (filtered_tx['transaction_date'].dt.date >= date_range[0]) &
        (filtered_tx['transaction_date'].dt.date <= date_range[1])
    ]
    
    # Filtra clientes com base nos segmentos selecionados
    filtered_customer_ids = customers_df[customers_df['customer_segment'].isin(selected_segmentos)]['customer_id']
    filtered_tx = filtered_tx[filtered_tx['customer_id'].isin(filtered_customer_ids)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔐 Análise de Fraudes")
        fraud_counts = filtered_tx['is_fraudulent'].value_counts()
        total_transacoes = len(filtered_tx)
        trans_fraud = fraud_counts.get(True, 0)
        taxa_fraude = (trans_fraud / total_transacoes * 100) if total_transacoes > 0 else 0

        st.metric("💳 Total de Transações (Filtradas)", value=total_transacoes)
        st.metric("🚨 Transações Fraudulentas (Filtradas)", value=trans_fraud, delta=f"{taxa_fraude:.1f}%")
        st.metric("📈 Média da Pontuação de Fraude (Filtrada)", value=f"{filtered_tx['fraud_score'].mean():.4f}")

        st.markdown("#### Distribuição da Pontuação de Fraude")
        score_bins = pd.cut(filtered_tx['fraud_score'], bins=10)
        bin_counts = score_bins.value_counts().sort_index()
        bin_counts.index = bin_counts.index.astype(str)  # converte Interval para string
        st.bar_chart(bin_counts)

        if trans_fraud > 0:
            top_merchant_fraud = filtered_tx[filtered_tx['is_fraudulent'] == True]['merchant_category'].value_counts().idxmax()
            st.markdown(f"📌 Categoria mais associada à fraude: **{top_merchant_fraud}**")

    with col2:
        st.subheader("👥 Segmentação de Clientes")
        
        # Total de clientes na base original
        st.metric("Total de Clientes na Base", value=len(customers_df))

        # Clientes no filtro atual
        num_filtered_customers = len(filtered_customer_ids)
        st.metric("🧑‍💼 Clientes Únicos", value=num_filtered_customers)
        
        # Média de transações por cliente filtrado
        avg_tx_per_customer_filtered = total_transacoes / max(num_filtered_customers, 1)
        st.metric("🧮 Média de Transações/Cliente", value=f"{avg_tx_per_customer_filtered:.1f}")

        st.markdown("#### Distribuição por Segmento")
        # Conta os segmentos dos clientes filtrados
        segment_counts_filtered = customers_df[customers_df['customer_id'].isin(filtered_customer_ids)]['customer_segment'].value_counts().sort_index()
        st.bar_chart(segment_counts_filtered)

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
        
        # Calcula as médias apenas para os clientes filtrados por segmento
        segment_analysis = customers_df[customers_df['customer_id'].isin(filtered_customer_ids)].groupby('customer_segment')[existing].mean().round(2)
        st.dataframe(segment_analysis)

    st.divider()
    st.subheader("🔎 Top 10 Transações com Maior Risco de Fraude (Filtradas)")
    top10 = filtered_tx.sort_values(by='fraud_score', ascending=False).head(10)
    st.dataframe(top10[['transaction_id', 'transaction_date', 'amount', 'merchant_category', 'fraud_score', 'is_fraudulent']])

    st.markdown("""
    > A **pontuação de fraude** (fraud_score) representa a **probabilidade de uma transação ser fraudulenta**, com base no modelo Random Forest. 
    > Pontuações próximas de 1.0 indicam maior risco.
    """)


elif page == "Análise de Transação (Simulação)":
    st.header("🔍 Simulador de Transações")
    st.markdown("Simule uma transação para verificar a probabilidade de fraude com base nas características fornecidas.")

    # Listas de opções para os selectbox, assumindo que os dados do BQ já estão em PT-BR
    # Top 20 profissões
    top_professions = customers_df['profession'].value_counts().head(20).index.tolist()
    # Top 20 categorias de comerciante
    top_categories = transactions_df['merchant_category'].value_counts().head(20).index.tolist()
    # Todas as localizações únicas, excluindo 'Unknown' e vazios, e ordenando
    all_locations = sorted([
        loc for loc in transactions_df['location'].unique() 
        if pd.notna(loc) and isinstance(loc, str) and loc.strip().lower() not in ['unknown', '']
    ])
    
    # Mapeamento de tipos de transação para português (se o BQ ainda tiver em inglês)
    transaction_types_map = {
        'Purchase': 'Compra', 'Withdrawal': 'Saque', 'Deposit': 'Depósito',
        'Transfer': 'Transferência', 'Online Payment': 'Pagamento Online',
        'Bill Payment': 'Pagamento de Conta', 'Unknown': 'Desconhecido'
    }
    # Obtém os tipos de transação únicos do DataFrame e os mapeia
    transaction_types_pt = [transaction_types_map.get(tt, tt) for tt in transactions_df['transaction_type'].unique()]

    # Mapeamento de estado civil para português (se o BQ ainda tiver em inglês)
    marital_status_map = {
        'Single': 'Solteiro(a)', 'Married': 'Casado(a)', 'Divorced': 'Divorciado(a)',
        'Widowed': 'Viúvo(a)', 'Unknown': 'Desconhecido'
    }
    # Obtém os estados civis únicos do DataFrame e os mapeia
    marital_status_pt = [marital_status_map.get(ms, ms) for ms in customers_df['marital_status'].unique()]
    
    # Mapeamento de dispositivo para português (se o BQ ainda tiver em inglês)
    device_info_map = {
        'Mobile': 'Celular', 'Desktop': 'Computador', 'POS Terminal': 'Terminal POS',
        'ATM': 'Caixa Eletrônico', 'Tablet': 'Tablet', 'Unknown': 'Desconhecido'
    }
    device_info_pt = [device_info_map.get(di, di) for di in transactions_df['device_info'].unique()]


    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("💵 Valor da Transação", min_value=0.0, value=1000.0)
            transaction_type = st.selectbox("Tipo de Transação", transaction_types_pt)
            merchant_category = st.selectbox("Categoria do Comerciante", top_categories)
        with col2:
            location = st.selectbox("Localização", all_locations)
            device_info = st.selectbox("Dispositivo", device_info_pt)
            transaction_hour = st.slider("Hora da Transação", 0, 23, 15)
        with col3:
            transaction_day_of_week = st.slider("Dia da Semana (0=Segunda, 6=Domingo)", 0, 6, 2)
            income = st.number_input("Renda do Cliente", min_value=0.0, value=5000.0)
            balance = st.number_input("Saldo da Conta", min_value=0.0, value=20000.0)
            customer_age_at_transaction = st.number_input("Idade do Cliente", min_value=0, value=30)
            marital_status = st.selectbox("Estado Civil", marital_status_pt)
            profession = st.selectbox("Profissão", top_professions)

        submitted = st.form_submit_button("🔎 Analisar Risco de Fraude")

        if submitted:
            # Reverte as seleções para o formato original (se houver mapeamento) para o encoder
            # Se os encoders foram treinados com os termos em português, não precisa reverter.
            # Assumindo que os encoders foram treinados com os termos que vêm do BigQuery.
            # Se o BigQuery tem "Compra", o encoder foi treinado com "Compra".
            # Se o BigQuery tem "Purchase", o encoder foi treinado com "Purchase".
            # O mapeamento aqui é apenas para a exibição no selectbox.
            
            # Reverte para o nome original para passar para o encoder, se houver mapeamento
            original_transaction_type = next((k for k, v in transaction_types_map.items() if v == transaction_type), transaction_type)
            original_marital_status = next((k for k, v in marital_status_map.items() if v == marital_status), marital_status)
            original_device_info = next((k for k, v in device_info_map.items() if v == device_info), device_info)


            input_data = pd.DataFrame([{
                'amount': amount,
                'income': income,
                'balance': balance,
                'transaction_hour': transaction_hour,
                'transaction_day_of_week': transaction_day_of_week,
                'customer_age_at_transaction': customer_age_at_transaction,
                'transaction_type': original_transaction_type, # Usar o original para o encoder
                'merchant_category': merchant_category,
                'location': location,
                'device_info': original_device_info, # Usar o original para o encoder
                'account_type': 'Unknown', # Não há input para isso, mantém como Unknown
                'marital_status': original_marital_status, # Usar o original para o encoder
                'profession': profession,
                'customer_segment': 0 # Este valor é um placeholder e não é usado no modelo de fraude
            }])

            input_data['amount_per_income'] = input_data['amount'] / (input_data['income'] + 1e-6)

            for col, encoder in fraud_encoders.items():
                if col in input_data.columns:
                    try:
                        # Garante que o valor a ser transformado é uma lista, mesmo que seja um único item
                        input_data[f'{col}_encoded'] = encoder.transform([str(input_data[col].iloc[0])])[0]
                    except ValueError:
                        # Se a categoria não for conhecida pelo encoder, atribui -1 (ou um valor default)
                        input_data[f'{col}_encoded'] = -1
                else:
                    input_data[f'{col}_encoded'] = -1

            X = input_data[[f for f in fraud_features_names if f in input_data.columns]]
            X = X[fraud_features_names] # Garante a ordem correta das features

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
                translated_col_name = feature_translation_map.get(col, col) # Traduz o nome da coluna
                valor = input_data[col].values[0]
                st.write(f"- **{translated_col_name}**: {valor}")

elif page == "Perfil do Cliente":
    st.header("👤 Perfil do Cliente")
    # Não converte para int aqui, mantém como string para corresponder ao DataFrame
    customer_id_input = st.text_input("ID do Cliente (Ex: CUST_00001)", value="CUST_00001")

    if customer_id_input:
        # Busca no DataFrame usando a string diretamente
        customer_profile = customers_df[customers_df['customer_id'] == customer_id_input]

        if not customer_profile.empty:
            st.subheader(f"🧾 Dados do Cliente ID: {customer_id_input}")
            # Exibe o perfil do cliente, garantindo que 'age' seja inteiro na exibição
            display_profile = customer_profile.drop(columns=['customer_id']).T.rename(columns={customer_profile.index[0]: 'Valor'})
            
            # Mapeamento para nomes de colunas no perfil do cliente
            profile_col_translation_map = {
                'name': 'Nome',
                'birth_date': 'Data de Nascimento',
                'age': 'Idade',
                'gender': 'Gênero',
                'marital_status': 'Estado Civil',
                'profession': 'Profissão',
                'income': 'Renda',
                'customer_segment': 'Segmento do Cliente'
            }
            display_profile.index = display_profile.index.map(profile_col_translation_map).fillna(display_profile.index)
            
            st.dataframe(display_profile.astype(str)) # Converte para string para exibição consistente

            segment = customer_profile['customer_segment'].iloc[0]
            st.write(f"Segmento: `{segment}`")

            st.subheader("📊 Características Médias do Segmento")
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
                        'num_products_held': 'Nº Produtos', 'marital_status_encoded': 'Estado Civil (Codificado)',
                        'profession_encoded': 'Profissão (Codificada)'
                    })
                )
                st.dataframe(segment_data.round(2))

            st.subheader("📂 Últimas Transações do Cliente")
            # Garante que a comparação é entre strings
            customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id_input].sort_values(by='transaction_date', ascending=False).head(10)
            if not customer_transactions.empty:
                st.dataframe(customer_transactions[['transaction_date', 'amount', 'transaction_type', 'merchant_category', 'fraud_score', 'is_fraudulent']])
            else:
                st.write("Nenhuma transação encontrada para este cliente.")
        else:
            st.warning("Cliente não encontrado. Verifique o ID.")

