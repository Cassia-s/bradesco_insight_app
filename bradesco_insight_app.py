import streamlit as st
import pandas as pd
import joblib
import os
from google.cloud import bigquery
from datetime import datetime
import json # Importar json para trabalhar com as credenciais

# --- Configurações do Google Cloud ---
project_id = "bradesco-insight"
dataset_id = "bradesco"
location = "southamerica-east1" # Certifique-se que esta é a localização correta do seu dataset

client = None # Inicializa client como None

# Tenta carregar as credenciais do Streamlit secrets
if "gcp_key" in st.secrets:
    try:
        # Parseia a string JSON das credenciais
        credentials_info = json.loads(st.secrets["gcp_key"]["json"])

        # Cria um arquivo temporário com as credenciais para o BigQuery Client
        # Isso é necessário porque GOOGLE_APPLICATION_CREDENTIALS espera um caminho de arquivo.
        temp_credentials_path = "gcp_credentials_temp.json"
        with open(temp_credentials_path, "w") as f:
            json.dump(credentials_info, f)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials_path
        client = bigquery.Client(project=project_id) # Esta linha instancia o cliente e atribui a 'client'
        st.success("Credenciais do GCP carregadas com sucesso via Streamlit Secrets!")
        
        # Opcional: remover o arquivo temporário quando o script terminar (para limpeza)
        # import atexit
        # atexit.register(lambda: os.remove(temp_credentials_path) if os.path.exists(temp_credentials_path) else None)

    except Exception as e:
        st.error(f"Erro ao carregar credenciais do Streamlit Secrets: {e}")
        st.info("Verifique o formato JSON em .streamlit/secrets.toml e se as chaves estão corretas.")
else:
    st.warning("Segredos do GCP não encontrados no Streamlit Secrets. Tentando autenticação local (gcloud CLI)...")
    try:
        # Tenta autenticação padrão do gcloud CLI para desenvolvimento local
        client = bigquery.Client(project=project_id) # Esta linha também!
        st.success("Autenticado no Google Cloud via gcloud CLI!")
    except Exception as e:
        st.error(f"Falha na autenticação do Google Cloud via gcloud CLI: {e}")
        st.info("Por favor, verifique se suas credenciais estão configuradas corretamente para o gcloud CLI (execute 'gcloud auth application-default login').")

if client is None:
    st.error("Não foi possível autenticar no Google Cloud. O aplicativo não pode continuar.")
    st.stop() # Interrompe a execução do Streamlit se não houver autenticação

# A partir daqui, o 'client' deve estar autenticado e pronto para ser usado.
# REMOVEMOS AQUI O BLOCO DUPLICADO DE get_bigquery_client() que causava o SyntaxError.

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
        # Carregar os nomes das features do modelo de fraude (garante alinhamento)
        fraud_features_names = joblib.load(os.path.join(model_dir, "fraud_features_names.joblib"))

        return model_fraud_detection, kmeans_model, scaler, fraud_encoders, customer_encoders, fraud_features_names
    except FileNotFoundError as e:
        st.error(f"Erro: Modelos não encontrados na pasta '{model_dir}'. Detalhe: {e}. Certifique-se de que você baixou a pasta 'models' do Colab e a colocou no mesmo diretório deste script. Para o deploy, a pasta 'models' deve estar no repositório GitHub.")
        st.stop()

# Desempacota os valores retornados por load_models
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

        st.write("### Características Médias por Segmento")
        features_for_segmentation = [
            'age', 'income', 'avg_balance', 'num_accounts', 'total_spent',
            'avg_transaction_amount', 'num_transactions', 'total_fraud_score',
            'num_fraudulent_transactions', 'num_products_held',
        ]
        # Garantir que as colunas codificadas existam para a análise
        if 'marital_status_encoded' in customers_df.columns:
            features_for_segmentation.append('marital_status_encoded')
        if 'profession_encoded' in customers_df.columns:
            features_for_segmentation.append('profession_encoded')

        existing_features_for_segmentation = [f for f in features_for_segmentation if f in customers_df.columns]
        segment_analysis = customers_df.groupby('customer_segment')[existing_features_for_segmentation].mean()
        
        # Renomear colunas para melhor visualização
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
        st.dataframe(segment_analysis_display.round(2))
        st.markdown("Esta tabela mostra as características médias de cada segmento de cliente. Valores como 'Status Civil (Codificado)' e 'Profissão (Codificado)' são médias dos valores numéricos atribuídos pelos LabelEncoders durante o pré-processamento.")

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
            customer_age_at_transaction = st.number_input("Idade do Cliente", min_value=0, value=30)
            marital_status = st.selectbox("Estado Civil do Cliente", customers_df['marital_status'].unique())
            profession = st.selectbox("Profissão do Cliente", top_professions) # USAR A LISTA FILTRADA

        submitted = st.form_submit_button("Analisar Risco de Fraude")

        if submitted:
            # Criar DataFrame para a nova transação
            new_transaction = pd.DataFrame([{
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
                'account_type': 'Unknown', # Placeholder se não tiver no input direto (assumindo que não é um campo de input)
                'marital_status': marital_status,
                'profession': profession,
                'customer_segment': 0 # Placeholder: o modelo de fraude foi treinado sem customer_segment. Se você re-treinar para incluir, precisará calcular o segmento aqui. Por agora, 0 é um placeholder.
            }])

            # Calcular amount_per_income
            new_transaction['amount_per_income'] = new_transaction['amount'] / (new_transaction['income'] + 1e-6)

            # Codificar variáveis categóricas usando os encoders salvos
            for col, encoder in fraud_encoders.items():
                if col in new_transaction.columns:
                    try:
                        # O reshape(-1, 1) é necessário para que encoder.transform aceite um único valor
                        # e retorna um array 1D, por isso o [0] no final
                        new_transaction[f'{col}_encoded'] = encoder.transform([new_transaction[col].iloc[0]])[0]
                    except ValueError as e:
                        st.warning(f"Aviso: Valor '{new_transaction[col].iloc[0]}' na coluna '{col}' não foi visto durante o treinamento do encoder. Usando fallback -1. Detalhe: {e}")
                        new_transaction[f'{col}_encoded'] = -1 # Um valor que o modelo possa interpretar como "desconhecido"
                else:
                    # Se a coluna não está no input de simulação e o encoder espera, defina um fallback
                    new_transaction[f'{col}_encoded'] = -1 


            # Selecionar e ordenar as features usando a lista carregada do modelo
            # Isso garante que as colunas de entrada para o predict_proba são EXATAMENTE as que o modelo espera.
            X_new_transaction = new_transaction[[f for f in fraud_features_names if f in new_transaction.columns]]

            # Verificar se X_new_transaction tem todas as features esperadas pelo modelo
            if not all(f in X_new_transaction.columns for f in fraud_features_names):
                missing_features = [f for f in fraud_features_names if f not in X_new_transaction.columns]
                st.error(f"Erro: As seguintes features esperadas pelo modelo de fraude estão faltando no input: {missing_features}. Por favor, verifique se todas as colunas de entrada foram corretamente processadas e se a lista de features no Colab está correta.")
                st.stop()

            # Garantir que a ordem das colunas esteja correta
            X_new_transaction = X_new_transaction[fraud_features_names]
            
            # Prever a pontuação de fraude
            fraud_score = model_fraud_detection.predict_proba(X_new_transaction)[:, 1][0]

            st.subheader("Resultado da Análise:")
            st.write(f"**Pontuação de Fraude:** `{fraud_score:.4f}`")

            if fraud_score >= 0.8:
                st.error("🔴 **ALTO RISCO DE FRAUDE!**")
                st.write("Esta transação apresenta um padrão de alto risco e pode ser fraudulenta. Recomenda-se investigação imediata.")
            elif fraud_score >= 0.4:
                st.warning("🟠 **MÉDIO RISCO DE FRAUDE!**")
                st.write("Esta transação exige atenção e pode precisar de verificação adicional antes da aprovação.")
            else:
                st.success("🟢 **BAIXO RISCO DE FRAUDE.**")
                st.write("Esta transação parece segura com base nos padrões atuais do modelo.")

# --- Perfil do Cliente ---
elif page == "Perfil do Cliente":
    st.header("Consulta de Perfil e Segmento do Cliente")
    st.write("Insira o ID de um cliente para ver seu perfil detalhado e segmento de cliente.")

    customer_id_input = st.text_input("ID do Cliente (Ex: 1, 5, 10)", value="1")

    if customer_id_input:
        try:
            customer_id = int(customer_id_input) # Tenta converter para int
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

                existing_features_for_segmentation = [f for f in features_for_segmentation if f in customers_df.columns]
                segment_analysis = customers_df.groupby('customer_segment')[existing_features_for_segmentation].mean()

                if segment in segment_analysis.index:
                    # Renomear colunas para melhor visualização na tabela de segmento
                    segment_analysis_display_single = segment_analysis.loc[segment].to_frame().T.rename(columns={
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
                    st.dataframe(segment_analysis_display_single.round(2))
                    st.markdown(f"Esta tabela exibe as características médias que definem os clientes do **Segmento {segment}**.")
                else:
                    st.write("Não foi possível encontrar as características médias para este segmento.")


                st.subheader("Últimas Transações do Cliente:")
                customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id].sort_values(by='transaction_date', ascending=False).head(10)
                if not customer_transactions.empty:
                    st.dataframe(customer_transactions[['transaction_date', 'amount', 'transaction_type', 'merchant_category', 'fraud_score', 'is_fraudulent']])
                else:
                    st.write("Nenhuma transação encontrada para este cliente.")

            else: # <--- Este é o 'else' que precisa ser corrigido
                st.warning("Cliente não encontrado. Por favor, verifique o ID.")
            except ValueError: # <--- Este é o bloco que o erro aponta.
                st.warning("ID do Cliente inválido. Por favor, insira um número inteiro.")

# Corrigido acesso ao segredo do BigQuery via gcp_key
```
Obrigado por me mostrar o código exato que você colou!

Sim, eu confirmei que **este código que você colou ainda tem o `SyntaxError`**. A linha `else:` que está causando o erro ainda está com a indentação incorreta, e o `except ValueError:` também está mal posicionado devido a isso.

Vou te mostrar a diferença exata na parte final do código, para que você possa entender o que está acontecendo:

**Seu código (com erro):**

```python
            else: # ESTÁ AQUI
                st.warning("Cliente não encontrado. Por favor, verifique o ID.")
            except ValueError: # E AQUI
                st.warning("ID do Cliente inválido. Por favor, insira um número inteiro.")
```

**O código CORRETO (como deveria estar):**

```python
            else: # AQUI A INDENTAÇÃO É DENTRO DO try
                st.write("Nenhuma transação encontrada para este cliente.") # Esta linha é do 'else' anterior

        # Este 'else' deveria estar aqui, na mesma coluna do 'if not customer_profile.empty' (linha 316 no seu código)
        # e é um 'else' para o 'if not customer_profile.empty' (lá na linha 316)
        else:
            st.warning("Cliente não encontrado. Por favor, verifique o ID.")
    except ValueError: # E este 'except' deveria estar aqui, na mesma coluna do 'try' (lá na linha 315)
        st.warning("ID do Cliente inválido. Por favor, insira um número inteiro.")
```

A razão pela qual isso continua acontecendo é porque a indentação (os espaços no início da linha) é **CRUCIAL** em Python para definir blocos de código. Um simples espaço a mais ou a menos muda o significado.

### **O Que Fazer AGORA (Última Tentativa de Cola e Verificação):**

Eu vou te dar o código **exatamente como ele deve ser**, focando mais uma vez na cópia perfeita.

1.  **Abra o seu arquivo `bradesco_insight_app.py` novamente no seu editor de texto.**

2.  **Selecione TODO o conteúdo e APAGUE-O**, deixando o arquivo completamente vazio.

3.  **Copie TODO o código do bloco `<immersive>` abaixo.** Use o botão "Copiar código" no canto superior direito do bloco para garantir que a indentação seja preservada.


```python
import streamlit as st
import pandas as pd
import joblib
import os
from google.cloud import bigquery
from datetime import datetime
import json # Importar json para trabalhar com as credenciais

# --- Configurações do Google Cloud ---
project_id = "bradesco-insight"
dataset_id = "bradesco"
location = "southamerica-east1" # Certifique-se que esta é a localização correta do seu dataset

client = None # Inicializa client como None

# Tenta carregar as credenciais do Streamlit secrets
if "gcp_key" in st.secrets:
    try:
        # Parseia a string JSON das credenciais
        credentials_info = json.loads(st.secrets["gcp_key"]["json"])

        # Cria um arquivo temporário com as credenciais para o BigQuery Client
        # Isso é necessário porque GOOGLE_APPLICATION_CREDENTIALS espera um caminho de arquivo.
        temp_credentials_path = "gcp_credentials_temp.json"
        with open(temp_credentials_path, "w") as f:
            json.dump(credentials_info, f)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials_path
        client = bigquery.Client(project=project_id) # Esta linha instancia o cliente e atribui a 'client'
        st.success("Credenciais do GCP carregadas com sucesso via Streamlit Secrets!")
        
        # Opcional: remover o arquivo temporário quando o script terminar (para limpeza)
        # import atexit
        # atexit.register(lambda: os.remove(temp_credentials_path) if os.path.exists(temp_credentials_path) else None)

    except Exception as e:
        st.error(f"Erro ao carregar credenciais do Streamlit Secrets: {e}")
        st.info("Verifique o formato JSON em .streamlit/secrets.toml e se as chaves estão corretas.")
else:
    st.warning("Segredos do GCP não encontrados no Streamlit Secrets. Tentando autenticação local (gcloud CLI)...")
    try:
        # Tenta autenticação padrão do gcloud CLI para desenvolvimento local
        client = bigquery.Client(project=project_id) # Esta linha também!
        st.success("Autenticado no Google Cloud via gcloud CLI!")
    except Exception as e:
        st.error(f"Falha na autenticação do Google Cloud via gcloud CLI: {e}")
        st.info("Por favor, verifique se suas credenciais estão configuradas corretamente para o gcloud CLI (execute 'gcloud auth application-default login').")

if client is None:
    st.error("Não foi possível autenticar no Google Cloud. O aplicativo não pode continuar.")
    st.stop() # Interrompe a execução do Streamlit se não houver autenticação

# A partir daqui, o 'client' deve estar autenticado e pronto para ser usado.
# REMOVEMOS AQUI O BLOCO DUPLICADO DE get_bigquery_client() que causava o SyntaxError.

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
        # Carregar os nomes das features do modelo de fraude (garante alinhamento)
        fraud_features_names = joblib.load(os.path.join(model_dir, "fraud_features_names.joblib"))

        return model_fraud_detection, kmeans_model, scaler, fraud_encoders, customer_encoders, fraud_features_names
    except FileNotFoundError as e:
        st.error(f"Erro: Modelos não encontrados na pasta '{model_dir}'. Detalhe: {e}. Certifique-se de que você baixou a pasta 'models' do Colab e a colocou no mesmo diretório deste script. Para o deploy, a pasta 'models' deve estar no repositório GitHub.")
        st.stop()

# Desempacota os valores retornados por load_models
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

        st.write("### Características Médias por Segmento")
        features_for_segmentation = [
            'age', 'income', 'avg_balance', 'num_accounts', 'total_spent',
            'avg_transaction_amount', 'num_transactions', 'total_fraud_score',
            'num_fraudulent_transactions', 'num_products_held',
        ]
        # Garantir que as colunas codificadas existam para a análise
        if 'marital_status_encoded' in customers_df.columns:
            features_for_segmentation.append('marital_status_encoded')
        if 'profession_encoded' in customers_df.columns:
            features_for_segmentation.append('profession_encoded')

        existing_features_for_segmentation = [f for f in features_for_segmentation if f in customers_df.columns]
        segment_analysis = customers_df.groupby('customer_segment')[existing_features_for_segmentation].mean()
        
        # Renomear colunas para melhor visualização
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
        st.dataframe(segment_analysis_display.round(2))
        st.markdown("Esta tabela mostra as características médias de cada segmento de cliente. Valores como 'Status Civil (Codificado)' e 'Profissão (Codificado)' são médias dos valores numéricos atribuídos pelos LabelEncoders durante o pré-processamento.")

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
            customer_age_at_transaction = st.number_input("Idade do Cliente", min_value=0, value=30)
            marital_status = st.selectbox("Estado Civil do Cliente", customers_df['marital_status'].unique())
            profession = st.selectbox("Profissão do Cliente", top_professions) # USAR A LISTA FILTRADA

        submitted = st.form_submit_button("Analisar Risco de Fraude")

        if submitted:
            # Criar DataFrame para a nova transação
            new_transaction = pd.DataFrame([{
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
                'account_type': 'Unknown', # Placeholder se não tiver no input direto (assumindo que não é um campo de input)
                'marital_status': marital_status,
                'profession': profession,
                'customer_segment': 0 # Placeholder: o modelo de fraude foi treinado sem customer_segment. Se você re-treinar para incluir, precisará calcular o segmento aqui. Por agora, 0 é um placeholder.
            }])

            # Calcular amount_per_income
            new_transaction['amount_per_income'] = new_transaction['amount'] / (new_transaction['income'] + 1e-6)

            # Codificar variáveis categóricas usando os encoders salvos
            for col, encoder in fraud_encoders.items():
                if col in new_transaction.columns:
                    try:
                        # O reshape(-1, 1) é necessário para que encoder.transform aceite um único valor
                        # e retorna um array 1D, por isso o [0] no final
                        new_transaction[f'{col}_encoded'] = encoder.transform([new_transaction[col].iloc[0]])[0]
                    except ValueError as e:
                        st.warning(f"Aviso: Valor '{new_transaction[col].iloc[0]}' na coluna '{col}' não foi visto durante o treinamento do encoder. Usando fallback -1. Detalhe: {e}")
                        new_transaction[f'{col}_encoded'] = -1 # Um valor que o modelo possa interpretar como "desconhecido"
                else:
                    # Se a coluna não está no input de simulação e o encoder espera, defina um fallback
                    new_transaction[f'{col}_encoded'] = -1 


            # Selecionar e ordenar as features usando a lista carregada do modelo
            # Isso garante que as colunas de entrada para o predict_proba são EXATAMENTE as que o modelo espera.
            X_new_transaction = new_transaction[[f for f in fraud_features_names if f in new_transaction.columns]]

            # Verificar se X_new_transaction tem todas as features esperadas pelo modelo
            if not all(f in X_new_transaction.columns for f in fraud_features_names):
                missing_features = [f for f in fraud_features_names if f not in X_new_transaction.columns]
                st.error(f"Erro: As seguintes features esperadas pelo modelo de fraude estão faltando no input: {missing_features}. Por favor, verifique se todas as colunas de entrada foram corretamente processadas e se a lista de features no Colab está correta.")
                st.stop()

            # Garantir que a ordem das colunas esteja correta
            X_new_transaction = X_new_transaction[fraud_features_names]
            
            # Prever a pontuação de fraude
            fraud_score = model_fraud_detection.predict_proba(X_new_transaction)[:, 1][0]

            st.subheader("Resultado da Análise:")
            st.write(f"**Pontuação de Fraude:** `{fraud_score:.4f}`")

            if fraud_score >= 0.8:
                st.error("🔴 **ALTO RISCO DE FRAUDE!**")
                st.write("Esta transação apresenta um padrão de alto risco e pode ser fraudulenta. Recomenda-se investigação imediata.")
            elif fraud_score >= 0.4:
                st.warning("🟠 **MÉDIO RISCO DE FRAUDE!**")
                st.write("Esta transação exige atenção e pode precisar de verificação adicional antes da aprovação.")
            else:
                st.success("🟢 **BAIXO RISCO DE FRAUDE.**")
                st.write("Esta transação parece segura com base nos padrões atuais do modelo.")

# --- Perfil do Cliente ---
elif page == "Perfil do Cliente":
    st.header("Consulta de Perfil e Segmento do Cliente")
    st.write("Insira o ID de um cliente para ver seu perfil detalhado e segmento de cliente.")

    customer_id_input = st.text_input("ID do Cliente (Ex: 1, 5, 10)", value="1")

    if customer_id_input:
        try:
            customer_id = int(customer_id_input) # Tenta converter para int
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

                existing_features_for_segmentation = [f for f in features_for_segmentation if f in customers_df.columns]
                segment_analysis = customers_df.groupby('customer_segment')[existing_features_for_segmentation].mean()

                if segment in segment_analysis.index:
                    # Renomear colunas para melhor visualização na tabela de segmento
                    segment_analysis_display_single = segment_analysis.loc[segment].to_frame().T.rename(columns={
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
                    st.dataframe(segment_analysis_display_single.round(2))
                    st.markdown(f"Esta tabela exibe as características médias que definem os clientes do **Segmento {segment}**.")
                else:
                    st.write("Não foi possível encontrar as características médias para este segmento.")

  if customer_id_input:

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

            existing_features = [f for f in features_for_segmentation if f in customers_df.columns]
            segment_analysis = customers_df.groupby('customer_segment')[existing_features].mean()

            if segment in segment_analysis.index:
                segment_data = segment_analysis.loc[segment].to_frame().T.rename(columns={
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
                st.dataframe(segment_data.round(2))
            else:
                st.write("Não foi possível encontrar as características médias para este segmento.")

            st.subheader("Últimas Transações do Cliente:")
            customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id].sort_values(by='transaction_date', ascending=False).head(10)
            if not customer_transactions.empty:
                st.dataframe(customer_transactions[['transaction_date', 'amount', 'transaction_type', 'merchant_category', 'fraud_score', 'is_fraudulent']])
            else:
                st.write("Nenhuma transação encontrada para este cliente.")
        else:
            st.warning("Cliente não encontrado. Por favor, verifique o ID.")
    except ValueError:
        st.warning("ID do Cliente inválido. Por favor, insira um número inteiro.")

# Corrigido acesso ao segredo do BigQuery via gcp_key
