# ==============================================================================
# APLICATIVO WEB DE ANÁLISE ESPACIAL DO IDEB
# Ferramenta: Streamlit
# Autor: Edson (adaptado para app por Gemini)
# ==============================================================================

# --- Importação das Bibliotecas ---
import streamlit as st
import pandas as pd
import geopandas as gpd
import geobr
import libpysal
from esda.moran import Moran, Moran_Local
import matplotlib.pyplot as plt
import numpy as np

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Análise Espacial do IDEB")

# ==============================================================================
# FUNÇÕES DE CACHE E ANÁLISE
# Usar @st.cache_data é CRUCIAL para o desempenho. Ele evita que os dados
# sejam baixados e processados toda vez que o usuário interage com o app.
# ==============================================================================

@st.cache_data
def carregar_dados_geograficos(uf_sigla):
    """Carrega os dados geográficos para a UF selecionada."""
    try:
        gdf = geobr.read_municipality(code_muni=uf_sigla, year=2019)
        gdf = gdf.to_crs("EPSG:4326")
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

def processar_e_juntar_dados(gdf, ideb_df, uf_sigla):
    """Filtra o IDEB para a UF, calcula médias e junta com o GeoDataFrame."""
    ideb_uf = ideb_df[ideb_df['UF'] == uf_sigla]

    if ideb_uf.empty:
        st.warning(f"Não foram encontrados dados do IDEB para o estado {uf_sigla}.")
        return None

    # Calcula médias por município
    media_mat = ideb_uf.groupby('cod_mun')['nota_matem'].mean().reset_index(name='media_mat')
    media_por = ideb_uf.groupby('cod_mun')['nota_portugues'].mean().reset_index(name='media_por')
    media_ideb = ideb_uf.groupby('cod_mun')['ideb'].mean().reset_index(name='media_ideb')

    # Junta as médias
    notas_uf = pd.merge(media_mat, media_por, on='cod_mun', how='outer')
    notas_uf = pd.merge(notas_uf, media_ideb, on='cod_mun', how='outer')

    # Junta com dados geográficos
    gdf_final = gdf.merge(notas_uf, left_on='code_muni', right_on='cod_mun', how='left')

    # Tratamento de valores ausentes (imputação pela média)
    for col in ['media_mat', 'media_por', 'media_ideb']:
        media_estado = gdf_final[col].mean()
        gdf_final[col].fillna(media_estado, inplace=True)

    return gdf_final

# ==============================================================================
# INTERFACE DO USUÁRIO (UI)
# ==============================================================================

# --- Título do Aplicativo ---
st.title("🗺️ Análise de Autocorrelação Espacial do IDEB 2021")
st.markdown("Esta ferramenta interativa permite analisar a distribuição espacial do desempenho educacional (IDEB) nos municípios brasileiros.")

# --- Barra Lateral para Controles ---
st.sidebar.header("Parâmetros da Análise")

# Dicionário de estados para seleção
estados_br = {
    'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas', 'BA': 'Bahia',
    'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo', 'GO': 'Goiás',
    'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul', 'MG': 'Minas Gerais',
    'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná', 'PE': 'Pernambuco', 'PI': 'Piauí',
    'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte', 'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina', 'SP': 'São Paulo',
    'SE': 'Sergipe', 'TO': 'Tocantins'
}
# O format_func transforma a sigla (ex: 'AM') no nome completo (ex: 'Amazonas') no dropdown
uf_selecionada = st.sidebar.selectbox(
    "Selecione um Estado:",
    options=list(estados_br.keys()),
    format_func=lambda x: estados_br[x],
    index=3 # Padrão para Amazonas
)

variavel_analise = st.sidebar.selectbox(
    "Selecione a Métrica para Análise:",
    options=['media_ideb', 'media_mat', 'media_por'],
    format_func=lambda x: {'media_ideb': 'IDEB Médio', 'media_mat': 'Nota de Matemática', 'media_por': 'Nota de Português'}[x]
)

# ==============================================================================
# LÓGICA PRINCIPAL DO APLICATIVO
# ==============================================================================

if uf_selecionada:
    # --- Carregamento e Processamento dos Dados ---
    with st.spinner(f"Carregando e processando dados para {estados_br[uf_selecionada]}..."):
        geodados_uf = carregar_dados_geograficos(uf_selecionada)
        ideb_nacional = carregar_dados_ideb()
        if geodados_uf is not None and ideb_nacional is not None:
            dados_completos = processar_e_juntar_dados(geodados_uf, ideb_nacional, uf_selecionada)
        else:
            dados_completos = None

    if dados_completos is not None:
        st.header(f"Análise para: {estados_br[uf_selecionada]}")
        st.subheader(f"Métrica: {variavel_analise.replace('media_', '').capitalize()}")

        # Extrai a variável de interesse
        y = dados_completos[variavel_analise]

        # --- Análise de Autocorrelação Global (I de Moran) ---
        st.markdown("### 1. Autocorrelação Espacial Global (I de Moran)")
        st.markdown("O I de Moran mede o grau de clusterização dos dados. Um valor positivo indica que municípios vizinhos tendem a ter valores semelhantes.")

        weights = libpysal.weights.Queen.from_dataframe(dados_completos)
        weights.transform = 'r'
        moran_global = Moran(y, weights, permutations=999)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Índice I de Moran", f"{moran_global.I:.4f}")
            st.metric("P-valor", f"{moran_global.p_sim:.4f}")
            if moran_global.p_sim < 0.05:
                st.success("O resultado é estatisticamente significativo, indicando a presença de autocorrelação espacial.")
            else:
                st.warning("O resultado não é estatisticamente significativo. A distribuição dos valores pode ser aleatória.")

        with col2:
            # Diagrama de Espalhamento de Moran
            fig, ax = plt.subplots()
            lag_y = libpysal.weights.lag_spatial(weights, y)
            ax.scatter(y, lag_y, alpha=0.6)
            m, b = np.polyfit(y, lag_y, 1)
            ax.plot(y, m*y + b, color='red')
            ax.axvline(y.mean(), color='k', linestyle='--')
            ax.axhline(lag_y.mean(), color='k', linestyle='--')
            ax.set_title("Diagrama de Espalhamento de Moran")
            ax.set_xlabel("Valor da Métrica no Município")
            ax.set_ylabel("Valor Médio nos Vizinhos")
            st.pyplot(fig)

        # --- Análise de Autocorrelação Local (LISA) ---
        st.markdown("### 2. Clusters Espaciais Locais (LISA)")
        st.markdown("A análise LISA identifica a localização de clusters estatisticamente significativos, mostrando **onde** os agrupamentos acontecem.")

        lisa = Moran_Local(y, weights)
        dados_completos['quadrante'] = lisa.q
        dados_completos['valor_p'] = lisa.p_sim

        # Mapa de Clusters
        fig, ax = plt.subplots(figsize=(15, 10))
        dados_completos.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)

        significativos = dados_completos[dados_completos['valor_p'] < 0.05]
        if not significativos.empty:
            quad_colors = {1: 'red', 2: 'lightblue', 3: 'blue', 4: 'pink'}
            quad_labels = {1: 'Alto-Alto', 2: 'Baixo-Alto', 3: 'Baixo-Baixo', 4: 'Alto-Baixo'}
            colors = significativos['quadrante'].map(quad_colors)
            significativos.plot(ax=ax, color=colors, edgecolor='black', linewidth=0.7)

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=quad_colors[i], edgecolor='k', label=quad_labels[i]) for i in significativos['quadrante'].unique()]
            ax.legend(handles=legend_elements, title="Tipos de Cluster (p < 0.05)")
        else:
            st.info("Não foram encontrados clusters locais estatisticamente significativos para os dados selecionados.")

        ax.set_title(f"Clusters LISA para '{variavel_analise}' em {estados_br[uf_selecionada]}")
        ax.set_axis_off()
        st.pyplot(fig)

        # --- Exibição dos Dados ---
        with st.expander("Ver Tabela de Dados Completa"):
            st.dataframe(dados_completos.drop(columns='geometry'))