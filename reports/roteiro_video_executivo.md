# Roteiro — Vídeo Executivo (até 5 minutos)

Este roteiro acompanha o PowerPoint [`apresentacao_executiva.pptx`](apresentacao_executiva.pptx) — 7 slides, cada um com as falas já cravadas como notas do apresentador (Modo de Exibição do Apresentador no PowerPoint mostra o texto abaixo sem aparecer na tela compartilhada). Formato: simular uma reunião executiva com gestores públicos/stakeholders.

Tempo total alvo: até **5min**, distribuído por slide conforme abaixo (soma ~4min20s, com folga de propósito).

## Slide 1 — Título + pergunta-guia (≈20s)

> Boa tarde. Hoje eu queria trazer uma pergunta simples: dá pra saber, com antecedência, se um município vai atingir a meta de alfabetização — antes do resultado sair? Foi isso que fui investigar, usando os dados da camada Gold que construí na fase anterior do projeto.

## Slide 2 — O problema educacional (≈40s)

> Hoje o acompanhamento é reativo: só descobrimos se um município atingiu a meta depois que o ano fechou. Isso significa que reforço, formação de professor, apoio técnico — tudo chega tarde demais pra mudar aquele resultado. Uma nota rápida sobre o enunciado: ele fala em prever se "um aluno" será alfabetizado, mas exige dados da camada Gold, agregada em município. Então aqui, isso vira a probabilidade de um aluno daquele município ser alfabetizado — é o nível que o dado público sustenta, respeitando a LGPD, já que resultado individual de alfabetização não é dado público no Brasil. A pergunta que guiou o projeto foi: dá pra antecipar esse risco?

*Sobre a nota no meio da fala:* o enunciado pede previsão "por aluno", mas exige dados da camada Gold (agregada em município × ano) e lista como entregável perguntas de negócio explicitamente municipais — o que ele pede como saída (aluno alfabetizado ou não) e o que ele exige como fonte (Gold, município) só se conciliam se "aluno alfabetizado" for lido como "a probabilidade de um aluno daquele contexto ser alfabetizado", que é exatamente o que o indicador percentual já representa. Isso também é a única leitura possível na prática: resultado individual de alfabetização é protegido por LGPD/sigilo educacional, não existe base pública nesse nível no Brasil para nenhuma equipe. Por isso essa nota fica curta e dentro do slide 2 — é contexto, não o argumento central do vídeo.

## Slide 3 — Principal insight: disparidade regional (≈55s)

**Mostrar o gráfico de barras por região na tela** — Sul/Sudeste vs. Norte/Nordeste.

> E aqui está o achado mais forte do projeto. Não é uma diferença sutil entre regiões, é uma clivagem estrutural. No período analisado, praticamente todo o Sul e o Sudeste atingiram a meta — 98% e 94% dos casos. Nordeste e Norte, o oposto: só 2% e 1%. Isso não é ruído estatístico, é um padrão consistente ano após ano, e foi esse padrão que guiou toda a modelagem.

## Slide 4 — Pipeline técnica (≈40s)

**Slide técnico** — diagrama da pipeline (extração → features/lag → sklearn Pipeline → validação temporal → 3 modelos comparados) e as métricas finais.

> Por trás desse número tem um processo técnico rigoroso. Extraí os dados direto do BigQuery, construí variáveis de histórico do próprio município sem vazar informação do resultado que eu queria prever, integrei tudo numa pipeline única do scikit-learn, e validei de um jeito realista: treinei com 2022 e 2023, testei em 2024, como se estivesse prevendo o futuro de verdade. Comparei três algoritmos, e o modelo final atinge 96% de ROC-AUC.

## Slide 5 — Interpretabilidade (≈40s)

**Mostrar SHAP summary e curva ROC lado a lado.**

> Mas não bastava ter uma métrica alta, eu precisava entender por quê. Usando SHAP, confirmei estatisticamente o que o gráfico anterior sugeria: a localização do município é o fator mais decisivo, seguido de perto pelo histórico recente dele. E a curva ROC mostra que o modelo separa muito bem quem está em risco de quem está no caminho certo.

## Slide 6 — Aplicação prática: ranking de risco (≈45s)

**Mostrar a tabela com os 5 municípios de maior risco.**

> Na prática, isso vira uma ferramenta de priorização: uma lista de municípios ordenada por probabilidade de não atingir a meta. Esses cinco aqui, por exemplo, têm menos de 1% de chance prevista de atingir a meta em 2024 — candidatos naturais para reforço escolar e formação de professores antes do resultado ruim se confirmar. E como o modelo entende o padrão regional, dá pra pensar política diferenciada por perfil de região, não uma receita única pro país inteiro.

## Slide 7 — Valor estratégico e encerramento (≈20s)

> No fim, a proposta é simples: sair de medir o passado pra agir sobre o futuro provável. Os próximos passos são enriquecer a base com dados socioeconômicos, expandir pra mais municípios, e colocar isso num dashboard que o gestor usa direto, sem precisar abrir notebook. Obrigado.

## Checklist de gravação

1. Abrir o PowerPoint em **Modo de Exibição do Apresentador** — as falas acima já estão nas notas de cada slide, não precisa decorar.
2. Gravar com câmera + tela (Teams/Zoom/OBS) simulando uma reunião real — olhar pra câmera nas transições, não só ler a tela.
3. Cronometrar o ensaio: ~260s de fala corrida (sem pausas longas) cabe folgado nos 5 minutos.
4. Exportar em MP4 e subir junto com a entrega final.
