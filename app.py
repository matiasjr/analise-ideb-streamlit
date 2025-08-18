# ==============================================================================
# APLICATIVO WEB DE ANÁLISE ESPACIAL DO IDEB (VERSÃO 3.0 - ANÁLISE COMPLETA)
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

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Análise Espacial do IDEB")

# ==============================================================================
# FUNÇÕES DE CACHE E ANÁLISE
# ==============================================================================

@st.cache_data
def carregar_dados_geograficos(uf_sigla):
    """Carrega os dados geográficos para a UF selecionada."""
    try:
        gdf = geobr.read_municipality(code_muni=uf_sigla, year=2019).to_crs("EPSG:4326")
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

# ==============================================================================
# LÓGICA PRINCIPAL DO APLICATIVO
# ==============================================================================

if uf_selecionada:
    with st.spinner(f"Carregando e processando dados para {estados_br[uf_selecionada]}..."):
        geodados_uf = carregar_dados_geograficos(uf_selecionada)
        ideb_nacional = carregar_dados_ideb()
        dados_completos = processar_e_juntar_dados(geodados_uf, ideb_nacional, uf_selecionada) if geodados_uf is not None and ideb_nacional is not None else None

    if dados_completos is not None:
        st.header(f"Análise para: {estados_br[uf_selecionada]}")

        # --- 1. ANÁLISE EXPLORATÓRIA DE DADOS (EDA) ---
        st.subheader("1. Análise Exploratória de Dados (EDA)")
        y = dados_completos['media_ideb']
        media_nacional = ideb_nacional['ideb'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Média do IDEB no Estado", f"{y.mean():.2f}")
            st.metric("Média do IDEB no Brasil", f"{media_nacional:.2f}", delta=f"{y.mean() - media_nacional:.2f}")
            st.metric("Desvio Padrão no Estado", f"{y.std():.2f}")
        with col2:
            municipio_maior_valor = dados_completos.loc[y.idxmax()]
            municipio_menor_valor = dados_completos.loc[y.idxmin()]
            st.metric("Mediana do IDEB no Estado", f"{y.median():.2f}")
            st.info(f"📍 **Maior IDEB:** {municipio_maior_valor['name_muni']} ({municipio_maior_valor['media_ideb']:.2f})")
            st.info(f"📍 **Menor IDEB:** {municipio_menor_valor['name_muni']} ({municipio_menor_valor['media_ideb']:.2f})")

        # --- 2. ANÁLISE DE AUTOCORRELAÇÃO GLOBAL (I DE MORAN) ---
        st.subheader("2. Análise de Autocorrelação Espacial Global (I de Moran)")
        st.markdown("Verificamos se existe um padrão de agrupamento geral no estado. Para isso, comparamos diferentes definições de 'vizinhança'.")
        pesos_dict = calcular_pesos(dados_completos, k_selecionado)
        resultados_moran = []
        for nome, w in pesos_dict.items():
            moran_r = Moran(y, w, permutations=999) # Padrão é transformação 'r'
            moran_b = Moran(y, w, transformation='b', permutations=999) # Especificamos transformação 'b'
            resultados_moran.append([nome, "Padronizada ('r')", moran_r.I, moran_r.p_sim])
            resultados_moran.append([nome, "Binária ('b')", moran_b.I, moran_b.p_sim])
        df_moran = pd.DataFrame(resultados_moran, columns=["Tipo de Vizinhança", "Tipo de Matriz", "I de Moran", "P-valor"])
        st.dataframe(df_moran.style.format({'I de Moran': '{:.4f}', 'P-valor': '{:.4f}'}))

        # --- CONDIÇÃO PARA CONTINUAR A ANÁLISE ---
        moran_escolhido = Moran(y, pesos_dict["Rainha"]) # Escolhemos Rainha Padronizada como método principal
        
        if moran_escolhido.I > 0 and moran_escolhido.p_sim < 0.05:
            st.success(f"O Índice de Moran Global ({moran_escolhido.I:.4f}) é positivo e estatisticamente significativo (p-valor={moran_escolhido.p_sim:.4f}). Isso confirma a existência de agrupamentos espaciais. Prosseguindo com a análise detalhada...")
            
            # --- 3. CORRELOGRAMA ESPACIAL ---
            st.subheader("3. Correlograma Espacial (Vizinhança Rainha)")
            try:
                moran_lags = [Moran(y, higher_order(pesos_dict["Rainha"], k=i)).I for i in range(1, 7)]
                fig, ax = plt.subplots()
                ax.plot(range(1, 7), moran_lags, 'o-')
                ax.axhline(0, color='grey', linestyle='--')
                ax.set_xlabel("Ordem de Vizinhança (Lag)")
                ax.set_ylabel("I de Moran")
                st.pyplot(fig)
            except Exception:
                st.warning("Não foi possível gerar o correlograma. O estado pode ter poucos vizinhos para ordens superiores.")

            # --- 4. DIAGRAMA DE ESPALHAMENTO DE MORAN ---
            st.subheader("4. Diagrama de Espalhamento de Moran")
            w_escolhido = pesos_dict["Rainha"]
            w_escolhido.transform = 'r'
            lag_ideb = libpysal.weights.lag_spatial(w_escolhido, y)
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
            st.markdown("Estes mapas mostram a média das notas dos vizinhos para cada município (lag espacial). Isso ajuda a visualizar as 'ilhas' de alto e baixo desempenho e comparar se os padrões do IDEB são mais parecidos com os de Matemática ou Português.")
            dados_completos['lag_mat'] = libpysal.weights.lag_spatial(w_escolhido, dados_completos['media_mat'])
            dados_completos['lag_por'] = libpysal.weights.lag_spatial(w_escolhido, dados_completos['media_por'])
            dados_completos['lag_ideb'] = lag_ideb # Já calculado
            
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            for i, col in enumerate(['lag_mat', 'lag_por', 'lag_ideb']):
                dados_completos.plot(column=col, ax=axes[i], legend=True, cmap='viridis', scheme='quantiles', k=5, edgecolor='k', linewidth=0.5)
                axes[i].set_title(f"Média Espacial - {col.split('_')[1].capitalize()}")
                axes[i].set_axis_off()
            st.pyplot(fig)
            
            # --- 6. ANÁLISE DE CLUSTERS LOCAIS (LISA) ---
            st.subheader("6. Análise de Clusters Locais (LISA)")
            lisa = Moran_Local(y, w_escolhido)
            dados_completos['lisa_Is'] = lisa.Is
            dados_completos['quadrante'] = lisa.q
            dados_completos['valor_p'] = lisa.p_sim

            # Mapa de Valores LISA
            st.markdown("**Mapa de Valores LISA (I Local)**")
            st.markdown("Este mapa mostra a intensidade do agrupamento para cada município. Valores altos (amarelo) indicam forte semelhança com os vizinhos, enquanto valores baixos (roxo) indicam dissimilaridade.")
            fig, ax = plt.subplots(figsize=(15, 10))
            dados_completos.plot(column='lisa_Is', cmap='viridis', scheme='quantiles', k=5, legend=True, ax=ax)
            ax.set_axis_off()
            st.pyplot(fig)
            
            # Mapa de Clusters Significativos
            st.markdown("**Mapa de Clusters LISA Significativos**")
            st.markdown("Este é o mapa principal, mostrando apenas os agrupamentos que são estatisticamente significativos (p-valor < 0.05).")
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
            st.dataframe(dados_completos.drop(columns=['geometry', 'abbrev_state']))