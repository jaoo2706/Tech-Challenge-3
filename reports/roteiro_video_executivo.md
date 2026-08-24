# Roteiro — Vídeo Executivo (até 5 minutos)

Formato sugerido: simular uma reunião executiva com gestores públicos/stakeholders, apresentando
o problema, os insights e o valor estratégico da solução.

## 1. Abertura — o problema educacional (≈45s)

- A alfabetização infantil é um dos principais indicadores do desenvolvimento educacional e
  social do Brasil.
- Hoje, gestores só enxergam o resultado *depois* que ele acontece — não há antecipação de
  risco.
- Pergunta que guia o projeto: **dado o histórico de um município, dá para prever, com
  antecedência, se ele vai atingir a meta de alfabetização?**

## 2. A base de dados (≈30s)

- Camada Gold da Fase 2: indicador de alfabetização, metas, dados territoriais e educacionais
  de ~512 municípios brasileiros, 2021–2024.
- Importante: dado agregado por município, não por aluno — o problema foi adaptado para prever
  **risco municipal**, que é justamente o nível de decisão dos gestores públicos.

## 3. Principais insights (≈90s)

- **Mostrar o gráfico de evolução por região** (`images/evolucao_regiao.png`): disparidade
  Sul/Sudeste vs. Norte/Nordeste é estrutural, não conjuntural.
- **Números que chocam**: no período, Sul atingiu a meta em 97,8% dos casos; Nordeste, em
  apenas 2%.
- **Mostrar o SHAP summary** (`images/shap_summary.png`): o modelo confirma estatisticamente
  que localização geográfica e histórico recente do município são os fatores mais decisivos —
  não é o tamanho da rede escolar.
- Inércia histórica: quem vinha bem tende a continuar bem — e vice-versa. Isso é uma boa
  notícia: **intervenção sustentada funciona e é mensurável**.

## 4. O modelo e como ele apoia decisão (≈90s)

- Pipeline de Machine Learning (scikit-learn) treinada com validação temporal (prevê 2024
  usando dados até 2023) — ROC-AUC de 0,96 no teste, ou seja, alta capacidade de distinguir
  município em risco de município no caminho certo.
- **Mostrar o ranking de risco** (`reports/ranking_risco_municipios_2024.csv`): lista priorizada
  de municípios com maior probabilidade de não atingir a meta — pronta para orientar alocação de
  recursos (formação de professores, material didático, apoio técnico) **antes** do resultado
  ruim se concretizar.

## 5. Valor estratégico e próximos passos (≈45s)

- Substitui reação por antecipação: shift de "medir o passado" para "agir no presente com base
  no que sabemos sobre o futuro provável".
- Próximos passos: enriquecer com dados socioeconômicos (IBGE, Cadastro Único) para entender
  *por que* a geografia pesa tanto, e expandir para o universo completo de municípios.

## Checklist de slides/telas para gravar

1. Slide título + pergunta-guia.
2. `images/evolucao_regiao.png` (disparidade regional ao longo do tempo).
3. `reports/resumo_taxa_atingimento_regiao.csv` (números por região).
4. `images/shap_summary.png` (drivers do modelo).
5. `images/roc_curve.png` ou `images/confusion_matrix.png` (qualidade do modelo).
6. `reports/ranking_risco_municipios_2024.csv` (aplicação prática — ranking de risco).
7. Slide de encerramento com próximos passos.
