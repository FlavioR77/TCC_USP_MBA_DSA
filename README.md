# 🏀 Utilizando Aprendizado de Máquina para Prever Resultados de Jogos da NBA

> **TCC — MBA em Data Science e Analytics | USP/ESALQ**  
> **Autor:** Flávio Renan Sant' Anna  
> **Orientadora:** Miriam Martin  
> **Nota:** 10,0 ⭐

---

## 📋 Descrição

Este repositório contém o código-fonte desenvolvido para o Trabalho de Conclusão de Curso do MBA em Data Science e Analytics da **USP/ESALQ**, com foco na previsão do resultado de jogos da **NBA (National Basketball Association)** utilizando algoritmos de aprendizado de máquina.

O problema foi modelado como uma **classificação binária**: Vitória (1) ou Derrota (0), com duas abordagens distintas — análise pós-jogo e análise pré-jogo.

---

## 🎯 Objetivo

Desenvolver e comparar modelos preditivos capazes de classificar o resultado de partidas da NBA, avaliando o impacto da qualidade e natureza dos dados de entrada na performance dos algoritmos.

**Grupos de interesse:** atletas, treinadores, franquias, patrocinadores, apostadores, fãs e entusiastas do esporte.

---

## 📊 Dados

- **Fonte:** Site oficial da NBA via biblioteca Python [`nba_api`](https://github.com/swar/nba_api)
- **Período:** 12 temporadas regulares (2011/2012 a 2023/2024)
- **Volume:** 28.876 registros
- **Escopo:** Apenas temporada regular (play-in e playoffs foram excluídos por apresentarem dinâmicas distintas)

### Variáveis principais

| Variável | Descrição |
|---|---|
| `WL` | Resultado: Vitória ou Derrota (target) |
| `PTS` | Total de pontos |
| `FGM / FGA / FG_PCT` | Arremessos convertidos / tentados / % |
| `FG3M / FG3A / FG3_PCT` | Arremessos de 3 pontos |
| `FTM / FTA / FT_PCT` | Lances livres |
| `OREB / DREB / REB` | Rebotes ofensivos, defensivos e totais |
| `AST / STL / BLK / TOV / PF` | Assistências, roubos, bloqueios, erros, faltas |
| `HOME` | Jogo em casa (criada) |
| `REST` | Dias de descanso entre jogos (criada) |
| `PREV_STREAK` | Sequência anterior de vitórias/derrotas (criada) |

---

## 🔬 Abordagens

### 1ª Abordagem — Análise Pós-Jogo (Caso 1)
Analisa as estatísticas **reais** de um time após a partida (adversário desconhecido) para determinar se ele venceu ou perdeu.  
📁 Script: `Modelo_Jogos_Atuais.py`

### 2ª Abordagem — Análise Pré-Jogo (Caso 2)
Analisa a **média dos últimos 10 jogos** de dois times para prever o vencedor antes da partida.  
📁 Script: `Modelo_Jogos_Futuros.py`

---

## 🤖 Modelos Utilizados

| Modelo | Biblioteca |
|---|---|
| Regressão Logística | `scikit-learn` |
| Floresta Randômica | `scikit-learn` |
| Redes Neurais Artificiais (RNA) | `PyTorch` |

### Arquitetura das Redes Neurais

**Caso 1** — 22 variáveis de entrada  
`Entrada(22) → Linear+ReLU(32) → Linear+ReLU(32) → Sigmoid(1)`

**Caso 2** — 43 variáveis de entrada  
`Entrada(43) → Linear+ReLU(64) → Linear+ReLU(64) → Sigmoid(1)`

Treinamento com **Early Stopping** (patience=10) e otimizador **Adam** (lr=0.001, weight_decay=0.001).

---

## 📈 Resultados

### Métricas de Desempenho

| Métrica | Reg. Logística | Floresta Randômica | Redes Neurais |
|---|---|---|---|
| | Caso 1 / Caso 2 | Caso 1 / Caso 2 | Caso 1 / Caso 2 |
| **Acurácia** | **0,843 / 0,654** | 0,813 / 0,617 | 0,834 / 0,628 |
| **Precisão** | **0,847 / 0,660** | 0,816 / 0,625 | 0,891 / 0,651 |
| **Sensibilidade** | **0,835 / 0,694** | 0,805 / 0,663 | 0,758 / 0,622 |
| **Especificidade** | 0,851 / 0,610 | 0,820 / 0,565 | **0,908 / 0,635** |
| **F1-Score** | **0,841 / 0,677** | 0,810 / 0,644 | 0,819 / 0,636 |

> ✅ **Modelo recomendado:** Regressão Logística — melhor desempenho geral em ambas as abordagens.

### Variáveis mais importantes (Caso 1 — Regressão Logística)

| Variável | Coeficiente | Interpretação |
|---|---|---|
| `DREB` | +2,00 | Rebotes defensivos → boa defesa → vitória |
| `FGA` | -1,88 | Muitos arremessos ineficientes → derrota |
| `TOV` | -1,29 | Turnovers geram pontos fáceis ao adversário |
| `PTS` | +1,09 | Mais pontos → maior chance de vitória |

### Comparação com a Literatura

| Modelo | Acurácia | Autor | Caso |
|---|---|---|---|
| Regressão Logística | 69,67% | Cao, 2012 | 1 |
| Regressão Logística | 69,10% | Lunelli, 2019 | 1 |
| RNA | 83,00% | Thabtah, 2019 | 1 |
| Floresta Randômica | 57,00% | Saijo, 2023 | 2 |
| **Regressão Logística** | **84,30%** | **Este trabalho** | **1** |
| **Regressão Logística** | **65,40%** | **Este trabalho** | **2** |

---

## 📁 Estrutura do Repositório

```
├── Criacao_Dataframes.py       # Coleta dos dados via nba_api e geração dos CSVs
├── Modelo_Jogos_Atuais.py      # Modelos para a 1ª Abordagem (pós-jogo)
├── Modelo_Jogos_Futuros.py     # Modelos para a 2ª Abordagem (pré-jogo)
├── README.md                   # Este arquivo
│
├── Base_Dados.csv              # Gerado por Criacao_Dataframes.py (1ª abordagem)
└── Registros_Combinados.csv    # Gerado por Criacao_Dataframes.py (2ª abordagem)
```

> ⚠️ Os arquivos `.csv` não estão incluídos no repositório por serem gerados localmente. Execute `Criacao_Dataframes.py` para gerá-los.

---

## ⚙️ Como Executar

### 1. Instale as dependências

```bash
pip install nba_api pandas numpy scikit-learn torch torchviz matplotlib seaborn
```

### 2. Gere as bases de dados

```bash
python Criacao_Dataframes.py
```

Isso irá criar `Base_Dados.csv` e `Registros_Combinados.csv`.

### 3. Execute os modelos

```bash
# 1ª Abordagem (pós-jogo)
python Modelo_Jogos_Atuais.py

# 2ª Abordagem (pré-jogo)
python Modelo_Jogos_Futuros.py
```

---

## 💡 Insights Relevantes

- **Vantagem de jogar em casa:** times mandantes venceram **57,33%** das partidas.
- **Dias de descanso:** o período ideal de preparação é de **3 dias** — jogos consecutivos impactam negativamente o desempenho.
- **Sequência de resultados (PREV_STREAK):** times em sequência de vitórias tendem a manter o desempenho positivo.
- **Qualidade dos dados:** o Caso 1 (dados reais pós-jogo) supera significativamente o Caso 2 (médias pré-jogo), pois a média suaviza os dados e perde informações relevantes para o modelo.

---

## 🔭 Trabalhos Futuros

- Utilizar o **somatório das médias individuais dos jogadores escalados** em vez da média do time como um todo, aproximando os dados estimados dos dados reais de cada partida.
- Replicar a metodologia para outros esportes com grande volume de dados estatísticos: **NFL** e **MLB**.
- Explorar modelos de **gradient boosting** (XGBoost, LightGBM) como alternativa à floresta randômica.

---

## 📚 Referências

- CAO, C. (2012). *Sports Data Mining Technology Used in Basketball Outcome Prediction.*
- LUNELLI, R. (2019). *Previsão de Resultados de Jogos de Futebol com Aprendizado de Máquina.*
- THABTAH, F. et al. (2019). *NBA Game Result Prediction Using Feature Analysis and Neural Networks.*
- SAIJO, R. (2023). *Predicting NBA Match Outcomes Using Random Forest.*

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do MBA em Data Science e Analytics da **USP/ESALQ**.

---

<div align="center">
  <strong>🏀 Flávio Renan Sant' Anna — USP/ESALQ — 2024</strong>
</div>
