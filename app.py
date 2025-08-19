# ==============================================================================
# APLICATIVO WEB DE ANÁLISE ESPACIAL DO IDEB (VERSÃO 4.2 - CORREÇÃO FINAL)
# Ferramenta: Streamlit
# Autor: Edson (com fluxo de análise e correções por Gemini)
# ==============================================================================

# --- Importação das Bibliotecas ---
import streamlit as st
import pandas as pd
import geopandas as gpd
import geobr
import libpysal
from libpysal.weights import Queen, Rook, KNN, higher_order
from esda.moran import Moran, Moran_Local
import matplotlib.pyplot as plt
import numpy as np
import copy

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Análise Espacial do IDEB")

# ==============================================================================
# FUNÇÕES DE CACHE E ANÁLISE
# ==============================================================================

@st.cache_data
def carregar_dados_geograficos(uf_sigla):
    """Carrega os dados geográficos para a UF selecionada."""
    try:
        gdf = geobr.read_municipality(code_muni=uf_sigla, year=2020).to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        st.error(f"Não foi possível carregar os dados geográficos para {uf_sigla}. Erro: {e}")
        return None

@st.cache_data
def carregar_dados_ideb():
    """Carrega os dados nacionais do IDEB."""
    try:
        df = pd.read_csv("https://edsonmatias.com.br/data/ideb_escola_2021.txt", delimiter="\t")
        df = df[df['ideb'] != 0].iloc[:, :-7]
        return df
    except Exception as e:
        st.error(f"Não foi possível carregar os dados do IDEB da fonte original. Erro: {e}")
        return None

@st.cache_data
def processar_e_juntar_dados(uf_sigla):
    """Função única para carregar, processar e juntar todos os dados para um estado."""
    gdf = carregar_dados_geograficos(uf_sigla)
    ideb_df = carregar_dados_ideb()

    if gdf is None or ideb_df is None:
        return None

    ideb_uf = ideb_df[ideb_df['UF'] == uf_sigla]

    if ideb_uf.empty:
        st.warning(f"Não foram encontrados dados do IDEB para o estado {uf_sigla}.")
        return None

    media_mat = ideb_uf.groupby('cod_mun')['nota_matem'].mean().reset_index(name='media_mat')
    media_por = ideb_uf.groupby('cod_mun')['nota_portugues'].mean().reset_index(name='media_por')
    media_ideb = ideb_uf.groupby('cod_mun')['ideb'].mean().reset_index(name='media_ideb')

    notas_uf = pd.merge(media_mat, media_por, on='cod_mun', how='outer')
    notas_uf = pd.merge(notas_uf, media_ideb, on='cod_mun', how='outer')

    gdf_final = gdf.merge(notas_uf, left_on='code_muni', right_on='cod_mun', how='left')

    for col in ['media_mat', 'media_por', 'media_ideb']:
        if not gdf_final[col].isna().all():
            media_estado = gdf_final[col].mean()
            gdf_final[col].fillna(media_estado, inplace=True)

    gdf_final = gdf_final.reset_index(drop=True)
    return gdf_final

@st.cache_resource
def calcular_pesos(uf_sigla, k):
    """Calcula e armazena em cache os pesos para a UF e k selecionados."""
    gdf = carregar_dados_geograficos(uf_sigla)
    if gdf is None:
        return None
    
    pesos = {
        "Rainha": Queen.from_dataframe(gdf, use_index=True),
        "Torre": Rook.from_dataframe(gdf, use_index=True),
        f"KNN (k={k})": KNN.from_dataframe(gdf, k=k, use_index=True)
    }
    return pesos

def calculate_spatial_correlogram(weights, values, max_lag, binaria='r', permutations=999):
    """Calcula o correlograma espacial de forma segura, evitando erros com lags vazios."""
    moran_values = []
    p_values = []
    
    for lag in range(1, max_lag + 1):
        lag_W = higher_order(weights, lag)
        
        # --- CORREÇÃO DO ERRO ---
        # A verificação agora soma os VALORES do dicionário `cardinalities`
        if sum(lag_W.cardinalities.values()) > 0:
            moran = Moran(values, lag_W, transformation=binaria, permutations=permutations)
            moran_values.append(moran.I)
            p_values.append(moran.p_sim)
        else:
            moran_values.append(0)
            p_values.append(1.0)
            
    return moran_values, p_values

# ==============================================================================
# INTERFACE DO USUÁRIO (UI)
# ==============================================================================

st.title("🗺️ Análise de Autocorrelação Espacial do IDEB 2021")

st.sidebar.header("Parâmetros da Análise")
estados_br = {
    'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas', 'BA': 'Bahia',
    'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo', 'GO': 'Goiás',
    'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul', 'MG': 'Minas Gerais',
    'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná', 'PE': 'Pernambuco', 'PI': 'Piauí',
    'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte', 'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina', 'SP': 'São Paulo',
    'SE': 'Sergipe', 'TO': 'Tocantins'
}
uf_selecionada = st.sidebar.selectbox("Selecione um Estado:", options=list(estados_br.keys()), format_func=lambda x: estados_br[x], index=3)
k_selecionado = st.sidebar.slider('Valor de K para vizinhança KNN', 1, 10, 5)
lags_selecionados = st.sidebar.slider('Número de Lags para o Correlograma', 2, 10, 6)


# ==============================================================================
# LÓGICA PRINCIPAL DO APLICATIVO
# ==============================================================================

if uf_selecionada:
    with st.spinner(f"Processando dados para {estados_br[uf_selecionada]}..."):
        dados_completos = processar_e_juntar_dados(uf_selecionada)
        ideb_nacional = carregar_dados_ideb()

    if dados_completos is not None and not dados_completos.empty:
        st.header(f"Análise para: {estados_br[uf_selecionada]}")

        # Seções 1 e 2
        st.subheader("1. Análise Exploratória de Dados (EDA)")
        y = dados_completos['media_ideb']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Municípios", f"{len(dados_completos)}")
            st.metric(f"Média do IDEB no Estado", f"{y.mean