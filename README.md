# Tech Challenge — Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

Projeto integrador da Fase 3 da Pós-Tech (Ciência de Dados). Constrói, a partir da camada Gold
desenvolvida na Fase 2 (pipeline de engenharia de dados do Indicador Criança Alfabetizada),
um modelo supervisionado e uma camada de inteligência analítica para apoiar decisões de
política pública educacional.

## Contexto do problema

A alfabetização infantil é um dos principais indicadores do desenvolvimento educacional e
social do Brasil. Gestores públicos precisam antecipar riscos, identificar regiões vulneráveis
e entender quais fatores mais impactam os resultados — não basta observar o indicador atual,
é preciso prever e agir preventivamente.

**Adaptação de escopo:** o enunciado original pede um modelo para prever se um *aluno* será
alfabetizado. A camada Gold da Fase 2, no entanto, está agregada em **município × ano**
(2021–2024, ~512 municípios) — não existe dado no nível de aluno individual disponível.
O problema foi reformulado, preservando a essência (antecipar risco de não-alfabetização para
apoiar decisão pública), para a granularidade real dos dados disponíveis: **prever se um
município atingirá ou não a meta de alfabetização em um determinado ano**. Essa reformulação
está diretamente alinhada às próprias perguntas de negócio do desafio ("quais municípios
apresentam maior risco educacional", "como prever municípios que podem não atingir metas
futuras").

## Objetivo analítico

Desenvolver uma pipeline de Machine Learning (scikit-learn) que classifique município-ano em
**"atingiu a meta de alfabetização"** vs. **"não atingiu"**, com:

- Tratamento adequado de valores faltantes e variáveis categóricas/numéricas;
- Prevenção explícita de data leakage;
- Validação temporal (o modelo é testado prevendo o "futuro" a partir do "passado");
- Interpretabilidade (Feature Importance + SHAP) para identificar os fatores mais associados
  ao (in)sucesso da alfabetização;
- Tradução dos resultados em inteligência aplicável a políticas públicas.

## Descrição da base utilizada

Fonte: BigQuery `tech-challenge-2-fiap-joao.gold_alfabetizacao` (camada Gold da Fase 2).

| Tabela | Granularidade | Uso no projeto |
|---|---|---|
| `indicador_alfabetizacao_municipio` | município × ano | Base principal (features + alvo) |
| `metas_vs_resultados_municipio/uf/brasil` | município/UF/Brasil × ano | Cruzamento e validação de metas |
| `evolucao_temporal_uf/regiao/brasil` | UF/região/Brasil × ano | EDA — tendência temporal agregada |

Colunas principais da tabela base: `ano`, `id_municipio`, `nome_municipio`, `sigla_uf`,
`nome_uf`, `regiao`, `indicador_alfabetizacao_percentual`, `proficiencia_media_saeb`,
`total_estudantes_avaliados`, `total_escolas_avaliadas`, `meta_percentual`,
`gap_meta_percentual`.

Os dados brutos ficam versionados em `data/raw/*.parquet` (extraídos via
`src/preprocessing/extract_gold_data.py`) e a base analítica final (com features de engenharia)
em `data/processed/base_analitica.parquet`.

## Etapas de modelagem

1. **Extração** (`src/preprocessing/extract_gold_data.py`): lê as 7 tabelas da camada Gold via
   BigQuery e salva em Parquet local.
2. **Engenharia de atributos e alvo** (`src/preprocessing/build_features.py`):
   - Define o alvo `alfabetizacao_adequada` = 1 se `gap_meta_percentual >= 0`.
   - **Remove data leakage**: exclui `indicador_alfabetizacao_percentual`,
     `gap_meta_percentual` e `proficiencia_media_saeb` do **ano corrente** (definem/vazam o
     próprio alvo).
   - Cria **features defasadas (lag 1 ano)** por município para essas mesmas variáveis, mais
     uma feature de tendência (`indicador_tendencia_lag1`) — captura o histórico do município
     sem revelar o resultado do ano avaliado.
   - Descarta 2021 (sem histórico para lag) e 2025 (ano incompleto na Gold).
3. **Pipeline de pré-processamento + modelo** (`src/modeling/train.py`), tudo integrado em um
   único `sklearn.Pipeline`:
   - Imputação de numéricas (mediana) e categóricas (moda);
   - `StandardScaler` nas numéricas, `OneHotEncoder` em `sigla_uf`;
   - Comparação de 3 algoritmos com `RandomizedSearchCV` (Logistic Regression, Random Forest,
     Gradient Boosting), otimizando ROC-AUC;
   - **Validação**: split temporal (treino 2022–2023, teste 2024) + `GroupKFold` por município
     na validação cruzada de hiperparâmetros (evita vazamento entre anos do mesmo município).
4. **Avaliação e interpretabilidade** (`src/evaluation/interpret.py`): matriz de confusão,
   curva ROC, Feature Importance e SHAP values.
5. **Inteligência de negócio** (`src/evaluation/business_insights.py`): ranking de municípios
   por risco, agregações por UF/região.

Reproduzir tudo:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.preprocessing.extract_gold_data   # requer gcloud auth application-default login
python -m src.preprocessing.build_features
python -m src.modeling.train
python -m src.evaluation.interpret
python -m src.evaluation.business_insights
```

Ou explore os notebooks narrados em `notebooks/01_eda.ipynb` (análise exploratória) e
`notebooks/02_modelagem.ipynb` (pipeline completa + resultados).

## Escolha do algoritmo

Três algoritmos foram comparados via validação cruzada (GroupKFold, ROC-AUC) e avaliados no
holdout temporal de 2024:

| Modelo | CV ROC-AUC (treino) | ROC-AUC (holdout 2024) | F1 (holdout 2024) |
|---|---|---|---|
| **Regressão Logística** (selecionado) | 0.972 | **0.964** | **0.893** |
| Random Forest | 0.965 | 0.955 | 0.863 |
| Gradient Boosting | 0.965 | 0.960 | 0.879 |

A **Regressão Logística** foi escolhida: desempenho estatisticamente equivalente aos modelos
mais complexos, com a vantagem de ser mais simples, rápida e interpretável (coeficientes
diretamente relacionáveis ao SHAP) — preferência por parcimônia quando o desempenho empata.

## Métricas de avaliação

Usamos ROC-AUC e F1 (em vez de acurácia pura) por causa do desbalanceamento do alvo (~69% dos
município-ano não atingem a meta, ~31% atingem). No holdout de 2024, o modelo final atinge:

- **ROC-AUC: 0.964**
- **F1: 0.893** (classe "atingiu meta": precisão 0.91 / recall 0.88)
- Acurácia: 0.94

Ver `reports/model_comparison.csv` e as figuras em `images/` (`confusion_matrix.png`,
`roc_curve.png`).

## Interpretação dos resultados

- **UF (estado)** é, de longe, o grupo de variáveis mais influente (Feature Importance e SHAP):
  PR, RS, RJ, MG, SC, SP, ES empurram a predição para "atingiu a meta"; CE, RN, PE, SE, AC, AM,
  PA, TO empurram para "não atingiu".
- **`alfabetizacao_adequada_lag1`** (o município já havia atingido a meta no ano anterior) é a
  segunda variável mais relevante — há forte autocorrelação/inércia temporal: quem vinha bem
  tende a continuar bem.
- Note-se que `regiao` foi **removida** das features do modelo (é uma agregação determinística
  de `sigla_uf`; incluir ambas geraria colinearidade perfeita e instabilizaria os coeficientes).
  Leituras regionais são obtidas agregando as UFs (ver `reports/resumo_taxa_atingimento_*.csv`).

## Insights encontrados

- **Disparidade regional extrema**: no período 2022–2024, Sul (97,8%) e Sudeste (93,8%) dos
  município-ano atingiram a meta de alfabetização; Nordeste (2,0%) e Norte (1,0%) praticamente
  não atingiram. Centro-Oeste fica no meio (36,1%).
- Isso não é uma diferença marginal — é uma clivagem estrutural, o que explica por que a
  variável geográfica domina o modelo: ela funciona como *proxy* de um conjunto mais amplo de
  fatores socioeconômicos e de infraestrutura educacional historicamente desiguais entre
  regiões.
- O histórico recente do próprio município (lag do indicador) é o segundo fator mais
  importante — risco educacional tem inércia; municípios que melhoram tendem a sustentar a
  melhora, e vice-versa. Isso sugere que **intervenções sustentadas ano a ano** têm efeito
  cumulativo mensurável.
- O porte do sistema escolar (nº de estudantes/escolas avaliadas), isoladamente, tem baixa
  correlação com o resultado — tamanho não é destino; a geografia e a trajetória histórica
  pesam mais.

Veja `reports/ranking_risco_municipios_2024.csv` para a lista completa de municípios
classificados por risco (probabilidade prevista de não atingir a meta em 2024), útil para
priorização de políticas focalizadas.

## Perguntas de negócio respondidas

- **Quais fatores mais impactam a alfabetização?** UF/localização geográfica e histórico
  recente do próprio município (ver seção "Interpretação dos resultados").
- **Quais municípios apresentam maior risco educacional?** Ver
  `reports/ranking_risco_municipios_2024.csv` — concentrados em CE, SE, PA, PE, AM, RN, TO.
- **Quais regiões possuem padrões semelhantes?** Sul e Sudeste formam um cluster de alto
  desempenho; Norte e Nordeste formam um cluster de baixo desempenho; Centro-Oeste é
  intermediário/transição (`reports/resumo_taxa_atingimento_regiao.csv`).
- **Como prever municípios que podem não atingir metas futuras?** O modelo treinado
  (`models/modelo_alfabetizacao.joblib`) gera, para qualquer município-ano com dados do ano
  anterior, uma probabilidade de atingir a meta — permite ranquear municípios para ação
  preventiva antes do resultado ser conhecido.
- **Quais variáveis possuem maior influência nos modelos?** Ver SHAP summary
  (`images/shap_summary.png`) e Feature Importance (`reports/feature_importance.csv`).

## Limitações do projeto

- Amostra de ~512 municípios (não o universo de ~5.570 municípios brasileiros) e apenas 4 anos
  de dados (2021–2024); o holdout temporal de 1 ano (2024) é pequeno para uma conclusão
  definitiva sobre generalização de longo prazo.
- O alvo é uma reformulação em nível de município do problema original (nível aluno); não
  substitui um modelo de predição individual, que exigiria dados de matrícula/aluno não
  presentes na camada Gold.
- Não foram incorporadas variáveis socioeconômicas externas (renda, IDHM, Cadastro Único) — o
  forte efeito de UF no modelo provavelmente atua como *proxy* dessas variáveis omitidas.
- `meta_percentual` é definida por política pública (mesmo valor nacional em cada ano), então
  carrega pouco poder discriminante *entre* municípios no mesmo ano — é mais relevante para
  comparações entre anos.

## Aplicação prática para políticas públicas

- **Priorização de recursos**: o ranking de risco (`reports/ranking_risco_municipios_2024.csv`)
  permite direcionar programas de reforço/formação de professores e material didático aos
  municípios com maior probabilidade prevista de não atingir a meta, antes do resultado do ano
  ser conhecido.
- **Diferenciação de política por perfil regional**: dado o padrão claro Sul/Sudeste vs.
  Norte/Nordeste, políticas nacionais uniformes tendem a ser menos eficazes que estratégias
  regionalizadas, com metas e apoio técnico escalonados por contexto local.
- **Monitoramento de trajetória**: como o histórico recente pesa fortemente na predição,
  acompanhar a tendência ano a ano de cada município (não só o valor absoluto) é um sinal de
  alerta precoce útil para gestores.

## Possíveis evoluções futuras

- Enriquecer a base com fontes externas (IBGE, Censo Escolar, FUNDEB, PNAD, Atlas do
  Desenvolvimento Humano, Cadastro Único) para captar diretamente os fatores socioeconômicos
  que hoje ficam implícitos no efeito de UF.
- Expandir a cobertura para o universo completo de municípios brasileiros e mais anos
  históricos, permitindo holdouts temporais maiores e mais robustos.
- Modelo de regressão para prever o valor contínuo do indicador (não só a classificação
  binária), útil para estimar a magnitude do gap, não apenas sua direção.
- Clusterização não supervisionada (K-Means/hierárquico) de municípios por padrão
  socioeducacional, complementando a resposta à pergunta "quais regiões possuem padrões
  semelhantes" com uma abordagem orientada a dados (não só agregação geográfica pré-definida).
- Dashboard interativo (Streamlit/Looker Studio) para gestores explorarem o ranking de risco e
  os drivers do modelo sem depender de notebooks.

## Estrutura do repositório

```
tech-challenge-fase3/
├── data/
│   ├── raw/            # extração direta da camada Gold (BigQuery -> Parquet)
│   └── processed/       # base analítica com features de engenharia + alvo
├── notebooks/
│   ├── 01_eda.ipynb            # análise exploratória e hipóteses
│   └── 02_modelagem.ipynb      # pipeline completa, resultados e interpretabilidade
├── src/
│   ├── preprocessing/   # extração de dados, feature engineering, tratamento de leakage
│   ├── modeling/        # pipeline sklearn, treino, validação, seleção de modelo
│   ├── evaluation/      # métricas, SHAP/feature importance, insights de negócio
│   └── visualization/   # (reservado para utilitários de plot reutilizáveis)
├── reports/             # CSVs de resultados (comparação de modelos, rankings, insights)
├── images/              # gráficos gerados pela EDA e pela avaliação do modelo
├── models/              # pipeline treinada serializada (.joblib)
├── requirements.txt
└── README.md
```

## Vídeo executivo

Pendente de gravação (roteiro disponível em `reports/roteiro_video_executivo.md`).
