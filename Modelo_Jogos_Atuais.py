# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:20:39 2024

@author: Flavio Renan Sant Anna
"""

#%% 

# Importando as bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
import graphviz
import torch
import torch.nn as nn
from torchviz import make_dot
import warnings
warnings.filterwarnings('ignore')

#%% 

# Caminho do arquivo CSV
file_path = 'Base_Dados.csv'

# Leitura do arquivo CSV
data_treino = pd.read_csv(file_path)

#%% 

# Excluindo as colunas sem relevância ou que podem prejudicar a análise

df = data_treino
colunas_removidas = ['SEASON_ID','TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME','GAME_ID','GAME_DATE','MATCHUP', 'PLUS_MINUS']
colunas_selecionadas = df.columns[~df.columns.isin(colunas_removidas)]

#%% 

# Calculando a porcentagem de vitória de times que jogam em casa e fora
porcentagem_vitorias = df.groupby('HOME')['WL'].mean() * 100

# Imprimindo as porcentagens formatadas
print('Porcentagem de vitória dos times que jogam em casa: {:.2f}%'.format(porcentagem_vitorias.get(1, 0)))
print('Porcentagem de vitória dos times que jogam fora: {:.2f}%'.format(porcentagem_vitorias.get(0, 0)))

# Criando o gráfico
plt.figure(figsize=(8, 5))  # Define o tamanho da figura
barras = plt.bar(['Fora', 'Casa'], porcentagem_vitorias, color=['red', 'blue'])  # Cria um gráfico de barras

# Adicionando a porcentagem sobre as barras
for barra in barras:
    yval = barra.get_height()  # Pega a altura da barra
    plt.text(barra.get_x() + barra.get_width()/2, yval + 1,  # Define a posição do texto
             f'{yval:.2f}%',  # Formata o texto com a porcentagem
             ha='center', va='bottom')  # Centraliza o texto acima da barra

plt.title('Porcentagem de Vitórias Jogando em Casa vs. Fora')  # Adiciona um título ao gráfico
plt.xlabel('Localização do Jogo')  # Nomeia o eixo X
plt.ylabel('Porcentagem de Vitórias (%)')  # Nomeia o eixo Y
plt.ylim(0, 100)  # Define os limites do eixo Y
plt.axhline(y=50, color='gray', linestyle='--')  # Linha horizontal de referência
plt.show()  # Exibe o gráfico

#%% 

# Agrupar por 'DIAS_DESCANSO' e calcular a porcentagem de vitórias
porcentagem_vitorias_descanso = df.groupby('REST')['WL'].mean().reset_index()
porcentagem_vitorias_descanso.columns = ['REST', 'WIN_PERCENTAGE']

# Plotar os dados
plt.figure(figsize=(12, 6))
plt.plot(porcentagem_vitorias_descanso['REST'], porcentagem_vitorias_descanso['WIN_PERCENTAGE'], marker='o')
plt.title('Porcentagem de Vitorias por Dia de Descanso')
plt.xlabel('Dias de Descanso')
plt.ylabel('Porcentagem de Vitorias')
plt.grid(True)
plt.xticks(porcentagem_vitorias_descanso['REST'])
plt.tight_layout()

# Mostrar o gráfico
plt.show()


#%% 

# Gráfico de Sequência Anterior de Vitórias/ Derrotas por % de vitória

# Calculando a média de vitórias para cada valor de PREV_STREAK
streak_win_pct = df.groupby('PREV_STREAK').agg({'WL': 'mean'}).reset_index()
streak_win_pct['WL'] *= 100  # Convertendo para porcentagem

# Criando o gráfico de dispersão com linha de regressão
plt.figure(figsize=(12, 7))
sns.regplot(data=streak_win_pct, x='PREV_STREAK', y='WL', scatter_kws={'s': 50}, line_kws={"color": "red"})

# Adicionando título e labels
plt.title('Influência da Sequência Anterior de Vitórias/Derrotas no Resultado do Jogo')
plt.xlabel('Sequência de Vitórias/Derrotas (PREV_STREAK)')
plt.ylabel('Porcentagem de Vitória (%)')
plt.grid(True)
plt.show()


#%% 

# Dividindo o Dataframe

dfy = df['WL']
dfX = df[colunas_selecionadas].drop('WL', axis = 1)

#%% 

# Calculo da correlação das variáveis

corr = dfX.corr()

# Identificar pares de variáveis com correlação alta
threshold = 0.8  # Definir um limite para a correlação alta
high_corr_pairs = [(corr.index[i], corr.columns[j], corr.iloc[i, j]) 
                   for i in range(corr.shape[0]) 
                   for j in range(i+1, corr.shape[1]) 
                   if abs(corr.iloc[i, j]) > threshold]

print("Pares de variáveis com correlação alta:")
for pair in high_corr_pairs:
    print(f"{pair[0]} e {pair[1]} com correlação {pair[2]:.2f}")

# Remover uma variável de cada par com alta correlação
variables_to_remove = set()
for pair in high_corr_pairs:
    variables_to_remove.add(pair[1])  # Adicionar uma variável do par para remoção

# Atualizar o dataframe removendo as variáveis selecionadas
dfX_reduzido = dfX.drop(columns=variables_to_remove)
# data_teste_reduzido_X = data_teste_X.drop(columns=variables_to_remove)
print("Variáveis removidas devido à alta correlação:")
print(variables_to_remove)    


# Padronizar os dados
scaler = StandardScaler()
dfX_padronizado = scaler.fit_transform(dfX_reduzido)
#data_teste_padronizado_X = scaler.fit_transform(data_teste_reduzido_X)

#%% 

# Gráfico para visualizar as corrrelações

sns.set(style="white")

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-(1.0), center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#%% 

# Separando as observações para treino e teste

bX_train, bX_test, by_train, by_test = train_test_split(dfX, dfy, test_size=0.2, random_state=9)

# versão para a regressão logistica

bX_padronizado_train, bX_padronizado_test, by_train, by_test = train_test_split(dfX_padronizado, dfy, test_size=0.2, random_state=9)


# Separando as observações para treino e teste mna rede neural

previsores_treinamento = bX_train
previsores_teste = bX_test
classe_treinamento = by_train
classe_teste = by_test

#%% 

# Convertendo os dados para tensores

previsores_treinamento = torch.tensor(np.array(previsores_treinamento), dtype=torch.float)
classe_treinamento = torch.tensor(np.array(classe_treinamento), dtype=torch.float)


#%% 

# Criando o dataset e o dataloader

dataset = torch.utils.data.TensorDataset(previsores_treinamento, classe_treinamento)
train_loader = torch.utils.data.DataLoader(dataset, batch_size= 128, shuffle=True)


#%% 
# Criando a Rede Neural

classificador = nn.Sequential(
    nn.Linear(in_features=22, out_features=32), nn.ReLU(),
   # nn.Dropout(0.2),
    nn.Linear(in_features=32, out_features=32), nn.ReLU(),
   # nn.Dropout(0.2),    
    nn.Linear(in_features=32, out_features=1), nn.Sigmoid()
)


#%% 


# Criando uma entrada de amostra compatível com o tamanho de entrada do modelo
x = torch.randn(1, 22)  

# Passando a entrada de amostra pelo modelo para obter a saída
y = classificador(x)

# Gerando o gráfico
dot = make_dot(y, params=dict(classificador.named_parameters()))

# Salvar o gráfico como um arquivo PNG
dot.render('network_graph', format='png')


                                                      
#%% 

# Definindo o critério de perda

criterion = nn.BCELoss()
                                                                                           
#%% 

# Definindo o otimizador

optimizer = torch.optim.Adam(classificador.parameters(), lr=0.001, weight_decay=0.001)

                                                                                           
#%% 

## Treinamento do modelo

# Loop de treinamento com early stopping

# Early Stopping: Monitorando a perda de validação e pare o treinamento
# quando a perda de validação começar a aumentar, o que pode indicar overfitting.


best_loss = float('inf')
patience = 10
trigger_times = 0

for epoch in range(100):
    running_loss = 0.0
    classificador.train()  # Definindo o modelo em modo de treinamento
    for inputs, labels in train_loader:
        labels = labels.unsqueeze(1)  # Ajuste para garantir que as dimensões correspondam
        optimizer.zero_grad()
        outputs = classificador(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Época %3d: perda %.5f' % (epoch+1, running_loss/len(train_loader)))
    
    # Early Stopping
    if running_loss < best_loss:
        best_loss = running_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    

#%% 

# obtendo uma lista de todos os parâmetros
params = list(classificador.parameters())

#%% 
## Avaliação do modelo

classificador.eval()


#%% 

# Convertendo os dados para um tensor PyTorch
previsores_teste = torch.tensor(np.array(previsores_teste), dtype=torch.float)


#%% 
# aplicando o modelo classificador aos dados de teste
previsoes = classificador(previsores_teste)

#%% 

# Move as previsões de volta para a CPU
previsoes = previsoes.cpu().detach()

# Converte para um array NumPy
previsoes = previsoes.numpy()

#%% 

previsoes = np.array(previsoes > 0.5)
previsoes

#%% 

taxa_acerto = accuracy_score(classe_teste, previsoes)
print(taxa_acerto)


#%% 

# Modelo de Regressão Logistica

logreg = LogisticRegression(max_iter=10000)
logreg.fit(bX_padronizado_train, by_train)


# Modelo de Floresta Randômica

rf = RandomForestClassifier()
rf.fit(bX_train, by_train)


#%% 

# Calculando as previsões Regressão Logistica

y_logreg = logreg.predict(bX_padronizado_test)
y_prob_logreg = logreg.predict_proba(bX_padronizado_test)[:, 1]


# Calculando as previsões Floresta Randômica

y_rf = rf.predict(bX_test)
y_prob_rf = rf.predict_proba(bX_test)[:, 1]

# Calculando a probabilidades preditiva da Rede Neural
y_prob_rn = classificador(previsores_teste).cpu().detach().numpy()



#%% 


# Calculo da Precisão para os modelos

# Regressão Logistica
precisao_logreg = round(accuracy_score(by_test, y_logreg) * 100, 2)


# Floresta Randômica
precisao_rf = round(accuracy_score(by_test, y_rf) * 100, 2)


# Redes Neurais
precisao_rn = round(accuracy_score(previsoes, classe_teste) * 100, 2)


#%% 

# Imprimindo os Resultados

print('\n')
print('Dataset de Treino', end='\n')
print('Precisão para o modelo Regressão Logistica: {}'.format(precisao_logreg), end='\n')
print('Precisão para o modelo Floresta Randômica: {}'.format(precisao_rf), end='\n')
print('Precisão para o modelo Redes Neurais: {}'.format(precisao_rn), end='\n\n\n')


#%% 

# Função para grafico de calor da matriz de confusão

def grafico_matrix_confusao(y_real, y_previsto, titulo='Matriz de Confusão'):

    # Supondo que by_test são os rótulos verdadeiros e y_logreg são as previsões do seu modelo
    matriz_confusao = confusion_matrix(y_real, y_previsto, labels=[1, 0])   

    # Criando um heatmap com a matriz de confusão
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=['1', '0'], yticklabels=['1', '0'])

    # Adicionando rótulos e título
    plt.title(titulo)
    plt.xlabel('Previsto')
    plt.ylabel('Real')

    # Rotacionando os rótulos do eixo y para ficarem na vertical
    plt.yticks(rotation=0)  # Isso define a rotação dos rótulos do eixo y para 0 graus (vertical)

    # Exibir o gráfico
    plt.show()


#%% 

# Matriz de Confusão Regressão Logística

grafico_matrix_confusao(by_test, y_logreg, 'Regressão Logística: Caso 1')

# Matriz de Confusão Regressão Logística

grafico_matrix_confusao(by_test, y_rf, 'Floresta Randômica: Caso 1')

# Matriz de Confusão Regressão Logística

grafico_matrix_confusao(classe_teste, previsoes, 'Redes Neurais: Caso 1')


#%% 

def calcular_especificidade(y_real, y_previsto):

    # Calculando a matriz de confusão
    matriz_confusao = confusion_matrix(y_real, y_previsto)

    # Extraindo TN e FP da matriz de confusão
    TN = matriz_confusao[0, 0]
    FP = matriz_confusao[0, 1]

    # Calculando a especificidade
    especificidade = TN / (TN + FP)

    return especificidade

def calcular_taxa_de_erro(y_real, y_previsto):

    # Calculando a acurácia
    acurácia = accuracy_score(y_real, y_previsto)

    # Calculando a taxa de erro
    taxa_de_erro = 1 - acurácia

    return taxa_de_erro


#%% 

# Metricas Regressão Logistica

# Calculando as métricas
acuracia = accuracy_score(by_test, y_logreg)
precisao = precision_score(by_test, y_logreg)
revocacao = recall_score(by_test, y_logreg)
f1 = f1_score(by_test, y_logreg)
especificidade = calcular_especificidade(by_test, y_logreg)
taxa_erro = calcular_taxa_de_erro(by_test, y_logreg)


# Exibindo as métricas
print('Metricas Regressão Logística', end='\n')
print(f'Acurácia: {acuracia:.3f}', end='\n')
print(f'Precisão: {precisao:.3f}', end='\n')
print(f'Revocação: {revocacao:.3f}', end='\n')
print(f'F1-Score: {f1:.3f}', end='\n')
print(f'Especificidade: {especificidade:.3f}', end='\n')
print(f'Taxa de Erro: {taxa_erro:.3f}', end='\n\n\n')


#%% 

# Metricas Floresta Randômica

# Calculando as métricas
acuracia = accuracy_score(by_test, y_rf)
precisao = precision_score(by_test, y_rf)
revocacao = recall_score(by_test, y_rf)
f1 = f1_score(by_test, y_rf)
especificidade = calcular_especificidade(by_test, y_rf)
taxa_erro = calcular_taxa_de_erro(by_test, y_rf)

# Exibindo as métricas
print('Metricas Floresta Randômica', end='\n')
print(f'Acurácia: {acuracia:.3f}', end='\n')
print(f'Precisão: {precisao:.3f}', end='\n')
print(f'Revocação: {revocacao:.3f}', end='\n')
print(f'F1-Score: {f1:.3f}', end='\n')
print(f'Especificidade: {especificidade:.3f}', end='\n')
print(f'Taxa de Erro: {taxa_erro:.3f}', end='\n\n\n')


#%% 

# Metricas Redes Neurais

# Calculando as métricas
acuracia = accuracy_score(classe_teste, previsoes)
precisao = precision_score(classe_teste, previsoes)
revocacao = recall_score(classe_teste, previsoes)
f1 = f1_score(classe_teste, previsoes)
especificidade = calcular_especificidade(classe_teste, previsoes)
taxa_erro = calcular_taxa_de_erro(classe_teste, previsoes)

# Exibindo as métricas

# Exibindo as métricas
print('Metricas Redes Neurais', end='\n')
print(f'Acurácia: {acuracia:.3f}', end='\n')
print(f'Precisão: {precisao:.3f}', end='\n')
print(f'Revocação: {revocacao:.3f}', end='\n')
print(f'F1-Score: {f1:.3f}', end='\n')
print(f'Especificidade: {especificidade:.3f}', end='\n')
print(f'Taxa de Erro: {taxa_erro:.3f}', end='\n\n\n')

#%% 

# Curva Roc

# Supondo que by_test são os rótulos verdadeiros, y_prob_logreg, y_prob_rf e y_prob_rn são as probabilidades de predição dos seus modelos

# Calculando a taxa de verdadeiros positivos (TPR) e a taxa de falsos positivos (FPR) para o modelo de regressão logística
fpr_logreg, tpr_logreg, _ = roc_curve(by_test, y_prob_logreg)
roc_auc_logreg = roc_auc_score(by_test, y_logreg)

# Calculando a taxa de verdadeiros positivos (TPR) e a taxa de falsos positivos (FPR) para o modelo de floresta aleatória
fpr_rf, tpr_rf, _ = roc_curve(by_test, y_prob_rf)
roc_auc_rf = roc_auc_score(by_test, y_rf)

# Calculando a taxa de verdadeiros positivos (TPR) e a taxa de falsos positivos (FPR) para o modelo de redes neurais
fpr_rn, tpr_rn, _ = roc_curve(classe_teste, y_prob_rn)
roc_auc_rn = roc_auc_score(classe_teste, previsoes)

# Plotando as curvas ROC
plt.figure(figsize=(10, 7))
plt.plot(fpr_logreg, tpr_logreg, color='blue', lw=2, label=f'Regressão Logística (AUC = {roc_auc_logreg:.3f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Floresta Aleatória (AUC = {roc_auc_rf:.3f})')
plt.plot(fpr_rn, tpr_rn, color='red', lw=2, label=f'Redes Neurais (AUC = {roc_auc_rn:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Classificação Aleatória')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (False Positive Rate)')
plt.ylabel('Taxa de Verdadeiros Positivos (True Positive Rate)')
plt.title('Curvas ROC Comparativas')
plt.legend(loc='lower right')
plt.show()


#%% 

# Verificando a importância das variáveis
importancia = logreg.coef_[0]
feature_names = dfX_reduzido.columns

for i, (v, name) in enumerate(zip(importancia, feature_names)):
    print(f'Variável: {name}, Score: {v:.5f}')

# Plotando o gráfico de importância das variáveis
plt.figure(figsize=(12, 6))
plt.bar(feature_names, importancia)
plt.xticks(rotation=90)  # Rotacionar os nomes das variáveis para melhor legibilidade
plt.xlabel('Variável')
plt.ylabel('Importância')
plt.title('Importância das Variáveis na Regressão Logística')
plt.show()

#%% 

# Modelo de Árvore de Decisão
arvore = DecisionTreeClassifier(max_depth=3)
arvore.fit(bX_train, by_train)

# Exportando a árvore de decisão para um arquivo .dot
dot_data = export_graphviz(arvore, out_file=None,
                           feature_names=bX_train.columns,
                           filled=True, rounded=True,
                           class_names=["não", "sim"],
                           special_characters=True)

# Visualizando a árvore de decisão usando a biblioteca Graphviz
graph = graphviz.Source(dot_data)
graph.render("Vitórias Jogos")  # Isso criará um arquivo .pdf no diretório atual

# Exibindo a árvore
graph.view()

