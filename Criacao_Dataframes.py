# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:20:39 2024

@author: Flavio Renan Sant Anna
"""

#%% 

#pip install nba_api

#%% 

# Importando as bibliotecas

from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#%% 

# Funções


# Função para substituir os valores de Sim e Nao para 1 e 0

def substituir_sim_nao(df, coluna, sim, nao):
    df[coluna] = df[coluna].replace({sim: 1, nao: 0})
    df[coluna] = df[coluna].astype('uint8')
    return df

# Função para calcular as sequências anteriores de vitórias/derrotas

def calcular_streaks(data):
    data['CHANGE'] = data.groupby('TEAM_ID')['WL'].diff().fillna(0) != 0
    data['STREAK_GROUP'] = data.groupby('TEAM_ID')['CHANGE'].cumsum()
    data['STREAK'] = data.groupby(['TEAM_ID', 'STREAK_GROUP']).cumcount() + 1
    data['STREAK'] *= data['WL'].map({1: 1, 0: -1})
    data['PREV_STREAK'] = data.groupby('TEAM_ID')['STREAK'].shift(1).fillna(0).astype(int)
    return data.drop(columns=['CHANGE', 'STREAK_GROUP', 'STREAK' ])

# Função para calcular a media das observações, agrupados por Time e temporada.

def encontrar_media_times(group):
    
    # Identifica colunas numéricas
    
    colunas_numericas = group.select_dtypes(include=[np.number])

    # Lista de colunas a serem excluídas, se estiverem presentes no DataFrame
    
    colunas_para_excluir = ['TEAM_ID', 'GAME_ID', 'WL', 'PLUS_MINUS', 'HOME', 'REST', 'PREV_STREAK']
    colunas_para_excluir = [col for col in colunas_para_excluir if col in colunas_numericas.columns]

    # Exclui as colunas selecionadas
    
    colunas_numericas = colunas_numericas.drop(columns = colunas_para_excluir)    
    
    # Calcula a média móvel das últimas 10 observações, excluindo a atual    
    
    rolling_numeric = colunas_numericas.shift(1).rolling(window=10, min_periods=1).mean()
   
    # Mantém as colunas não numéricas
    
    colunas_nao_numericas = group[['WL', 'SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'PLUS_MINUS', 'HOME', 'REST', 'PREV_STREAK']].copy()

    # Combina as colunas numéricas calculadas com as colunas não numéricas
    
    result = pd.concat([colunas_nao_numericas, rolling_numeric], axis=1)
    return result

# Função para combinar as ocorrências

# Lista de estatísticas que queremos manipular

stats_columns = ['MIN', 'PTS', 'REB', 'OREB','DREB', 'AST', 'STL', 'BLK', 'TOV','PF', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'REST', 'PREV_STREAK']

def combinar_ocorrencias(group):
    if len(group) == 2:
        team_stats = group.iloc[0]
        opponent_stats = group.iloc[1]
        
        combinar_stats = team_stats.copy()
        for col in stats_columns:
            combinar_stats[f'OPP_{col}'] = opponent_stats[col]
        
        return pd.DataFrame([combinar_stats])  # Retornar como DataFrame
    else:
        return pd.DataFrame([group.iloc[0]])  # Retornar como DataFrame

#%% 

# Get **Todos** os jogos ( Limitado a 30.000 registros )

resultado = leaguegamefinder.LeagueGameFinder(league_id_nullable='00', season_type_nullable='Regular Season')
total_jogos = resultado.get_data_frames()[0]

# Filtrando a temporada Regular 2012 a 2024

total_jogos = total_jogos[(total_jogos.SEASON_ID.str[-4:].astype(int) >= 2012) & (total_jogos.SEASON_ID.str[-4:].astype(int) <= 2024)]


#%% 

# Incluindo a coluna de jogos em casa ('Home')

total_jogos['HOME'] = ~total_jogos['MATCHUP'].str.contains('@')


#%% 

# Substituindo os valores das variáveis categóricas

substituir_sim_nao(total_jogos, 'HOME', True, False)
substituir_sim_nao(total_jogos, 'WL', 'W', 'L')


#%% 

# Calculando os dias de folga

# Converter 'GAME_DATE' para o tipo datetime

total_jogos['GAME_DATE'] = pd.to_datetime(total_jogos['GAME_DATE'])

# Ordenar o DataFrame por TEAM_ID e GAME_DATE

total_jogos_ordenado = total_jogos.sort_values(by=['TEAM_ID', 'GAME_DATE'])

# Calcular a diferença em dias entre os jogos para cada time

total_jogos_ordenado['REST'] = (total_jogos_ordenado.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days) - 1

# Mostrar as colunas relevantes (TEAM_NAME, GAME_DATE, REST_DAY)

print(total_jogos_ordenado[['TEAM_NAME', 'GAME_DATE', 'REST']])

# Substituir os valores maiores que 5 ou NaN por 5

total_jogos_ordenado.loc[total_jogos_ordenado['REST'] > 5, 'REST'] = 5
total_jogos_ordenado['REST'].fillna(5, inplace=True)


#%% 

# Aplicando a função sequencia anterior ao DataFrame

total_jogos_ordenado = calcular_streaks(total_jogos_ordenado)



#%% 

# Verificando valores nulos de cada coluna

# Contantando os Valores nulos
nan_counts = total_jogos_ordenado.isna().sum()
print(nan_counts)

# Substituindo os valores NaN por 0
total_jogos_ordenado.fillna(0, inplace=True)

# Verificando as Substituições
nan_counts = total_jogos_ordenado.isna().sum()
print(nan_counts)


#%% 

# Separando a base de dados para a criação dos Arquivos CSV.

df = total_jogos_ordenado.copy()
base_dados = df
df_rolling_treino = df

#%% 

# Criando o DataFrame das médias dos Times

df_rolling_treino = df_rolling_treino.groupby(["TEAM_ABBREVIATION", "SEASON_ID"], group_keys=False).apply(encontrar_media_times)

# Verificação após a aplicação de encontrar_media_times

print("Tipo após encontrar_media_times:", type(df_rolling_treino))
print(df_rolling_treino.head())

#%% 

# Exclui linhas com qualquer valor NaN no df_rolling

df_rolling_treino = df_rolling_treino.dropna()

# Verificação após a exclusão de NaN

print("Tipo após dropna:", type(df_rolling_treino))
print(df_rolling_treino.head())


#%% 

# Carregando os dados

team_data = df_rolling_treino.copy()

# Agrupando por GAME_ID e combinando os registros

combined_data = team_data.groupby('GAME_ID', group_keys=False).apply(combinar_ocorrencias).reset_index(drop=True)

# Verificação após combinar os registros

print("Tipo de combined_data:", type(combined_data))
print(combined_data.head())


#%% 

# Excluindo linhas com qualquer valor NaN no df_rolling

combined_data = combined_data.dropna()


#%% 

# Criando os arquivos "CSV" dos DataFrames

combined_data.to_csv('Registros_Combinados.csv', index=False)
base_dados.to_csv('Base_Dados.csv', index=False)

