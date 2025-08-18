# ==============================================================================
# APLICATIVO WEB DE ANÁLISE ESPACIAL DO IDEB (VERSÃO 2.1 - CORRIGIDA)
# Ferramenta: Streamlit
# Autor: Edson (com novas features e correções por Gemini)
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
import copy # <--- ADICIONADO PARA CORRIGIR O ERRO

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Análise Espacial do IDEB")

# ==============================================================================
# FUNÇÕES DE CACHE E ANÁLISE
# Usar @st.cache_data e @st.cache_resource é CRUCIAL para o desempenho.
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

@st.cache_data
def processar_e_juntar_dados(_gdf, _ideb_df, uf_sigla):
    """Filtra o IDEB para a UF, calcula médias e junta com o GeoDataFrame."""
    gdf = _gdf.copy()
    ideb_df = _ideb_df.copy()

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
        media_estado = gdf_final[col].mean()
        gdf_final[col].fillna(media_estado, inplace=True)

    return gdf_final

@st.cache_resource
def calcular_pesos(_gdf, k):
    """Calcula e armazena em cache os diferentes tipos de matrizes de pesos."""
    pesos = {
        "Rainha": Queen.from_dataframe(_gdf),
        "Torre": Rook.from_dataframe(_gdf),
        f"KNN (k={k})": KNN.from_dataframe(_gdf, k=k)
    }
    return pesos

def calcular_correlograma(weights, values, max_lag, binaria='r', permutations=999):
    """Calcula os valores do I de Moran para múltiplos lags espaciais."""
    moran_values = []
    p_values = []

    # O objeto de pesos original não deve ser modificado
    # CORREÇÃO: Usando copy.deepcopy() em vez do método inexistente .clone()
    w_copy = copy.deepcopy(weights)
    w_copy.transform = binaria

    # Lag 1
    moran = Moran(values, w_copy, permutations=permutations)
    moran_values.append(moran.I)
    p_values.append(moran.p_sim)

    # Lags > 1
    for lag in range(2, max_lag + 1):
        lag_w = higher_order(w_copy, lag)
        moran = Moran(values, lag_w, permutations=permutations)
        moran_values.append(moran.I)
        p_values.append(moran.p_sim)

    return moran_values, p_values

# ==============================================================================
# INTERFACE DO USUÁRIO (UI)
# ==============================================================================

st.title("🗺️ Análise de Autocorrelação Espacial do IDEB 2021")
st.markdown("Esta ferramenta interativa permite analisar a distribuição espacial do desempenho educacional (IDEB) nos municípios brasileiros.")

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
uf_selecionada = st.sidebar.selectbox(
    "Selecione um Estado:",
    options=list(estados_br.keys()),
    format_func=lambda x: estados_br[x],
    index=3
)

variavel_analise = st.sidebar.selectbox(
    "Selecione a Métrica para Análise:",
    options=['media_ideb', 'media_mat', 'media_por'],
    format_func=lambda x: {'media_ideb': 'IDEB Médio', 'media_mat': 'Nota de Matemática', 'media_por': 'Nota de Português'}[x]
)

k_selecionado = st.sidebar.slider('Valor de K para vizinhança KNN', 1, 10, 5)
lags_selecionados = st.sidebar.slider('Número de Lags para o Correlograma', 2, 10, 6)

# ==============================================================================
# LÓGICA PRINCIPAL DO APLICATIVO
# ==============================================================================

if uf_selecionada:
    with st.spinner(f"Carregando e processando dados para {estados_br[uf_selecionada]}..."):
        geodados_uf = carregar_dados_geograficos(uf_selecionada)
        ideb_nacional = carregar_dados_ideb()
        if geodados_uf is not None and ideb_nacional is not None:
            dados_completos = processar_e_juntar_dados(geodados_uf, ideb_nacional, uf_selecionada)
        else:
            dados_completos = None

    if dados_completos is not None:
        st.header(f"Análise para: {estados_br[uf_selecionada]}")

        # --- 1. Análise Exploratória de Dados (EDA) ---
        st.markdown("### 1. Análise Exploratória de Dados (EDA)")

        y = dados_completos[variavel_analise]
        media_nacional = ideb_nacional[variavel_analise.replace('media_', 'nota_') if 'nota' in variavel_analise else 'ideb'].mean()

        municipio_maior_valor = dados_completos.loc[y.idxmax()]
        municipio_menor_valor = dados_completos.loc[y.idxmin()]

        col1, col2, col3 = st.columns(3)
        col1.metric(f"Média no Estado", f"{y.mean():.2f}")
        col2.metric("Média no Brasil", f"{media_nacional:.2f}", delta=f"{y.mean() - media_nacional:.2f}")
        col3.metric("Nº de Municípios", f"{len(dados_completos)}")

        st.info(f"📍 **Maior valor:** {municipio_maior_valor['name_muni']} ({municipio_maior_valor[variavel_analise]:.2f})")
        st.info(f"📍 **Menor valor:** {municipio_menor_valor['name_muni']} ({municipio_menor_valor[variavel_analise]:.2f})")


        # --- 2. Comparativo de Autocorrelação Global (I de Moran) ---
        st.markdown("### 2. Autocorrelação Espacial Global (Comparativo)")
        st.markdown("O I de Moran mede a clusterização geral. Abaixo, comparamos os resultados com diferentes definições de vizinhança e pesos.")

        pesos_dict = calcular_pesos(dados_completos, k_selecionado)

        resultados_moran = []
        for nome, w in pesos_dict.items():
            # Matriz Padronizada
            # CORREÇÃO: Usando copy.deepcopy() em vez do método inexistente .clone()
            w_r = copy.deepcopy(w); w_r.transform = 'r'
            moran_r = Moran(y, w_r, permutations=999)
            resultados_moran.append([nome, "Padronizada ('r')", moran_r.I, moran_r.p_sim])

            # Matriz Binária
            # CORREÇÃO: Usando copy.deepcopy() em vez do método inexistente .clone()
            w_b = copy.deepcopy(w); w_b.transform = 'b'
            moran_b = Moran(y, w_b, permutations=999)
            resultados_moran.append([nome, "Binária ('b')", moran_b.I, moran_b.p_sim])

        df_moran = pd.DataFrame(resultados_moran, columns=["Tipo de Vizinhança", "Tipo de Matriz", "I de Moran", "P-valor"])
        st.dataframe(df_moran.style.format({'I de Moran': '{:.4f}', 'P-valor': '{:.4f}'}))

        # --- 3. Correlograma Espacial ---
        st.markdown("### 3. Correlograma Espacial")
        st.markdown("O correlograma mostra como a autocorrelação muda à medida que consideramos vizinhos mais distantes (lags).")

        w_base = pesos_dict["Rainha"]

        try:
            with st.spinner("Calculando correlogramas..."):
                moran_r, p_r = calcular_correlograma(w_base, y, lags_selecionados, binaria='r')
                moran_b, p_b = calcular_correlograma(w_base, y, lags_selecionados, binaria='b')

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            lags = np.arange(1, lags_selecionados + 1)

            axes[0].plot(lags, moran_r, 'o-')
            axes[0].axhline(y=0, color='gray', linestyle='--')
            axes[0].set_title("Proximidade Padronizada")
            axes[0].set_xlabel("Ordem de Vizinhança (Lag)")
            axes[0].set_ylabel("Moran's I")

            axes[1].plot(lags, moran_b, 'o-')
            axes[1].axhline(y=0, color='gray', linestyle='--')
            axes[1].set_title("Proximidade Binária")
            axes[1].set_xlabel("Ordem de Vizinhança (Lag)")

            st.pyplot(fig)
        except Exception as e:
            st.error(f"Não foi possível gerar o correlograma. Pode não haver vizinhos suficientes para os lags solicitados. Erro: {e}")

        # --- 4. Clusters Espaciais Locais (LISA) ---
        st.markdown("### 4. Clusters Espaciais Locais (LISA)")
        st.markdown("A análise LISA identifica a localização de clusters estatisticamente significativos, mostrando **onde** os agrupamentos acontecem.")

        w_lisa = copy.deepcopy(pesos_dict["Rainha"])
        w_lisa.transform = 'r'
        lisa = Moran_Local(y, w_lisa)
        dados_completos['quadrante'] = lisa.q
        dados_completos['valor_p'] = lisa.p_sim

        fig, ax = plt.subplots(figsize=(15, 10))
        dados_completos.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)

        significativos = dados_completos[dados_completos['valor_p'] < 0.05]
        if not significativos.empty:
            quad_colors = {1: 'red', 2: 'lightblue', 3: 'blue', 4: 'pink'}
            quad_labels = {1: 'Alto-Alto', 2: 'Baixo-Alto', 3: 'Baixo-Baixo', 4: 'Alto-Baixo'}
            colors = significativos['quadrante'].map(quad_colors)
            significativos.plot(ax=ax, color=colors, edgecolor='black', linewidth=0.7)

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=quad_colors[i], edgecolor='k', label=quad_labels[i]) for i in sorted(significativos['quadrante'].unique())]
            ax.legend(handles=legend_elements, title="Tipos de Cluster (p < 0.05)")
        else:
            st.info("Não foram encontrados clusters locais estatisticamente significativos para os dados selecionados.")

        ax.set_title(f"Clusters LISA para '{variavel_analise.replace('media_', '').capitalize()}' em {estados_br[uf_selecionada]}")
        ax.set_axis_off()
        st.pyplot(fig)

        with st.expander("Ver Tabela de Dados Completa"):
            st.dataframe(dados_completos.drop(columns='geometry'))