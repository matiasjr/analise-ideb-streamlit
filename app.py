# ==============================================================================
# APLICATIVO WEB DE ANÁLISE ESPACIAL DO IDEB (VERSÃO 4.0 - CORREÇÃO DEFINITIVA)
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
        # A função é chamada diretamente no objeto de pesos original (não transformado)
        lag_W = higher_order(weights, lag)
        
        if lag_W.cardinalities.sum() > 0:
            # A transformação é feita temporariamente dentro da função Moran
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
            st.metric(f"Média do IDEB no Estado", f"{y.mean():.2f}")
        with col2:
            st.metric("Mediana do IDEB no Estado", f"{y.median():.2f}")
            st.metric("Moda do IDEB no Estado", f"{y.mode()[0]:.2f}")
        with col3:
            st.metric("Média do IDEB no Brasil", f"{ideb_nacional['ideb'].mean():.2f}", delta=f"{y.mean() - ideb_nacional['ideb'].mean():.2f}")
            st.metric("Desvio Padrão no Estado", f"{y.std():.2f}")
        
        municipio_maior_valor = dados_completos.loc[y.idxmax()]
        municipio_menor_valor = dados_completos.loc[y.idxmin()]
        st.info(f"📍 **Maior IDEB:** {municipio_maior_valor['name_muni']} ({municipio_maior_valor['media_ideb']:.2f})")
        st.info(f"📍 **Menor IDEB:** {municipio_menor_valor['name_muni']} ({municipio_menor_valor['media_ideb']:.2f})")

        st.subheader("2. Análise de Autocorrelação Espacial Global (I de Moran)")
        pesos_dict = calcular_pesos(uf_selecionada, k_selecionado)
        
        resultados_moran = []
        for nome, w in pesos_dict.items():
            moran_r = Moran(y, w) # Padrão já é 'r'
            moran_b = Moran(y, w, transformation='b')
            resultados_moran.append([nome, "Padronizada ('r')", moran_r.I, moran_r.p_sim])
            resultados_moran.append([nome, "Binária ('b')", moran_b.I, moran_b.p_sim])
        df_moran = pd.DataFrame(resultados_moran, columns=["Tipo de Vizinhança", "Tipo de Matriz", "I de Moran", "P-valor"])
        st.dataframe(df_moran.style.format({'I de Moran': '{:.4f}', 'P-valor': '{:.4f}'}))

        moran_escolhido = Moran(y, pesos_dict["Rainha"])
        
        if moran_escolhido.I > 0 and moran_escolhido.p_sim < 0.05:
            st.success(f"O Índice de Moran Global ({moran_escolhido.I:.4f}) é positivo e estatisticamente significativo. Prosseguindo com a análise detalhada...")
            
            # Pega o objeto de pesos original (não transformado)
            w_rainha_original = pesos_dict["Rainha"]
            
            # --- 3. CORRELOGRAMA ESPACIAL ---
            st.subheader("3. Correlograma Espacial (Vizinhança Rainha)")
            st.markdown("O correlograma mostra como a autocorrelação (I de Moran) diminui à medida que consideramos vizinhos mais distantes (lags).")
            
            with st.spinner("Calculando correlogramas..."):
                # A função é chamada com a matriz de pesos original e não transformada
                moran_W, p_W = calculate_spatial_correlogram(w_rainha_original, y, lags_selecionados, binaria='r')
                moran_B, p_B = calculate_spatial_correlogram(w_rainha_original, y, lags_selecionados, binaria='b')
            
            lags = np.arange(1, lags_selecionados + 1)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

            axes[0].plot(lags, moran_W, 'o-')
            axes[0].axhline(y=0, color='gray', linestyle='--')
            axes[0].set_title("Proximidade Padronizada")
            axes[0].set_xlabel("Ordem de Vizinhança (Lag)")
            axes[0].set_ylabel("I de Moran")
            axes[0].grid(linestyle='--', alpha=0.6)

            axes[1].plot(lags, moran_B, 'o-')
            axes[1].axhline(y=0, color='gray', linestyle='--')
            axes[1].set_title("Proximidade Binária")
            axes[1].set_xlabel("Ordem de Vizinhança (Lag)")
            axes[1].grid(linestyle='--', alpha=0.6)
            
            st.pyplot(fig)

            # --- PREPARAÇÃO PARA ANÁLISES SEGUINTES ---
            # Criamos uma CÓPIA do objeto de pesos e a transformamos para 'r'.
            # Isso garante que o objeto original no cache permaneça intacto.
            w_lisa = copy.deepcopy(w_rainha_original)
            w_lisa.transform = 'r'
            
            # --- 4. DIAGRAMA DE ESPALHAMENTO DE MORAN ---
            st.subheader("4. Diagrama de Espalhamento de Moran")
            lag_ideb = libpysal.weights.lag_spatial(w_lisa, y)
            fig, ax = plt.subplots()
            ax.scatter(y, lag_ideb, alpha=0.6)
            m, b = np.polyfit(y, lag_ideb, 1)
            ax.plot(y, m*y + b, color='red')
            ax.axvline(y.mean(), color='k', linestyle='--')
            ax.axhline(lag_ideb.mean(), color='k', linestyle='--')
            ax.set_title("IDEB do Município vs. Média dos Vizinhos")
            st.pyplot(fig)

            # --- 5. ANÁLISE DAS MÉDIAS ESPACIAIS (LAG ESPACIAL) ---
            st.subheader("5. Análise das Médias Espaciais")
            st.markdown("Estes mapas mostram a média das notas dos vizinhos para cada município (lag espacial).")
            dados_completos['lag_mat'] = libpysal.weights.lag_spatial(w_lisa, dados_completos['media_mat'])
            dados_completos['lag_por'] = libpysal.weights.lag_spatial(w_lisa, dados_completos['media_por'])
            dados_completos['lag_ideb'] = lag_ideb
            
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            for i, col in enumerate(['lag_mat', 'lag_por', 'lag_ideb']):
                dados_completos.plot(column=col, ax=axes[i], legend=True, cmap='viridis', scheme='quantiles', k=5, edgecolor='k', linewidth=0.5)
                axes[i].set_title(f"Média Espacial - {col.split('_')[1].capitalize()}")
                axes[i].set_axis_off()
            st.pyplot(fig)
            
            # --- 6. ANÁLISE DE CLUSTERS LOCAIS (LISA) ---
            st.subheader("6. Análise de Clusters Locais (LISA)")
            lisa = Moran_Local(y, w_lisa)
            dados_completos['lisa_Is'] = lisa.Is
            dados_completos['quadrante'] = lisa.q
            dados_completos['valor_p'] = lisa.p_sim

            st.markdown("**Mapa de Valores LISA (I Local)**")
            fig, ax = plt.subplots(figsize=(15, 10))
            dados_completos.plot(column='lisa_Is', cmap='viridis', scheme='quantiles', k=5, legend=True, ax=ax)
            ax.set_axis_off()
            st.pyplot(fig)
            
            st.markdown("**Mapa de Clusters LISA Significativos**")
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
                st.info("Apesar do padrão global, não foram encontrados clusters locais estatisticamente significativos.")
            ax.set_axis_off()
            st.pyplot(fig)

        else:
            st.warning(f"O Índice de Moran Global ({moran_escolhido.I:.4f}) não indica a presença de agrupamentos espaciais significativos (p-valor={moran_escolhido.p_sim:.4f}). A distribuição do IDEB no estado pode ser aleatória. A análise de clusters não será exibida.")

        # --- 7. TABELA DE DADOS ---
        with st.expander("Ver Tabela de Dados Completa"):
            st.dataframe(dados_completos.drop(columns=['geometry', 'abbrev_state'], errors='ignore'))