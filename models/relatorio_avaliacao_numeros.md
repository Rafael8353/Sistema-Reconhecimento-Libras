# Relatorio de avaliacao - Sistema de Reconhecimento de Libras

Gerado em 12/11/2026 02:07:10

## 1. Resumo do dataset

- Total de amostras: 1973
- Letras/classes distintas: 10
- Pessoas distintas no dataset: 1 (Rafael)
- Atributos usados: 87 (42 coordenadas normalizadas + atributos geometricos + profundidade z)

### Amostras por letra

| Letra | Amostras |
|---|---|
| 0 | 152 |
| 1 | 121 |
| 2 | 200 |
| 3 | 195 |
| 4 | 207 |
| 5 | 210 |
| 6 | 220 |
| 7 | 335 |
| 8 | 144 |
| 9 | 189 |

## 2. Comparacao entre modelos (validacao cruzada 5-fold)

Mesmas metricas calculadas para todos os modelos, sobre o dataset inteiro:

| Modelo | Acuracia | F1 (macro) | Log Loss | AUC-ROC (OvR) |
|---|---|---|---|---|
| MLP (rede neural) | 99.59% (+/- 0.12) | 0.9956 | 0.0265 | 0.9992 |
| RandomForest | 99.34% (+/- 0.30) | 0.9931 | 0.0643 | 0.9995 |
| KNN (k=5) | 98.99% (+/- 0.70) | 0.9887 | 0.0893 | 0.9985 |

### 2.1 Validacao em blocos de tempo (sem vazamento de quadros vizinhos)

As amostras sao gravadas em sequencia e quadros vizinhos ficam quase identicos; na validacao aleatoria acima eles aparecem em treino e teste ao mesmo tempo, o que torna o resultado otimista. Aqui, cada dobra deixa de fora blocos inteiros de tempo (GroupKFold, 5 dobras), estimativa mais proxima do uso real (Random Forest):

- Acuracia (blocos de tempo): 98.99%
- F1 macro (blocos de tempo): 0.9893

| Classe | Precisao | Recall | F1-score | Suporte |
|---|---|---|---|---|
| 0 | 0.986 | 0.954 | 0.970 | 152 |
| 1 | 1.000 | 1.000 | 1.000 | 121 |
| 2 | 1.000 | 1.000 | 1.000 | 200 |
| 3 | 1.000 | 0.995 | 0.997 | 195 |
| 4 | 0.995 | 0.995 | 0.995 | 207 |
| 5 | 0.986 | 1.000 | 0.993 | 210 |
| 6 | 0.995 | 0.995 | 0.995 | 220 |
| 7 | 0.988 | 0.991 | 0.990 | 335 |
| 8 | 0.993 | 0.965 | 0.979 | 144 |
| 9 | 0.959 | 0.989 | 0.974 | 189 |

Confusoes mais frequentes na validacao em blocos de tempo:

| Classe real | Confundida com | Quantidade |
|---|---|---|
| 0 | 9 | 7 |
| 8 | 7 | 3 |
| 9 | 0 | 2 |
| 3 | 4 | 1 |
| 4 | 5 | 1 |
| 6 | 7 | 1 |
| 7 | 5 | 1 |
| 7 | 6 | 1 |

Veja `matriz_confusao_blocos` (imagem) para o mapa completo.

## 3. Metricas por letra (modelo Random Forest, conjunto de teste holdout 20%)

Log loss no holdout: 0.0465

| Letra | Precisao | Recall | F1-score | AUC-ROC | Suporte |
|---|---|---|---|---|---|
| 0 | 0.968 | 1.000 | 0.984 | 1.000 | 30 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 24 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 40 |
| 3 | 1.000 | 0.974 | 0.987 | 1.000 | 39 |
| 4 | 0.976 | 0.976 | 0.976 | 1.000 | 42 |
| 5 | 0.955 | 1.000 | 0.977 | 1.000 | 42 |
| 6 | 1.000 | 1.000 | 1.000 | 1.000 | 44 |
| 7 | 1.000 | 1.000 | 1.000 | 1.000 | 67 |
| 8 | 1.000 | 0.966 | 0.982 | 1.000 | 29 |
| 9 | 1.000 | 0.974 | 0.987 | 1.000 | 38 |

**Piores letras (menor F1-score):**

- 4: F1 = 0.976
- 5: F1 = 0.977
- 8: F1 = 0.982
- 0: F1 = 0.984
- 9: F1 = 0.987

**Melhores letras (maior F1-score):**

- 1: F1 = 1.000
- 2: F1 = 1.000
- 7: F1 = 1.000
- 6: F1 = 1.000
- 3: F1 = 0.987

## 4. Letras mais confundidas entre si

| Letra real | Confundida com | Quantidade de vezes (no teste) |
|---|---|---|
| 3 | 4 | 1 |
| 4 | 5 | 1 |
| 8 | 5 | 1 |
| 9 | 0 | 1 |

Veja a imagem `matriz_confusao.png` para o mapa completo de confusoes.

## 5. O desempenho varia de acordo com a pessoa?

**Ainda nao e possivel responder isso com confianca**: o dataset atual tem praticamente uma unica pessoa registrada (Rafael). O orientador pediu a coleta de sinais de 3 ou 4 pessoas diferentes justamente para permitir esta analise (testar se o modelo generaliza para maos/estilos diferentes, em vez de ter decorado as caracteristicas de uma unica pessoa). Assim que houver dados de mais pessoas, basta rodar este script novamente: essa secao passa a ser preenchida automaticamente com uma validacao 'deixe uma pessoa de fora' (treina com todas as pessoas menos uma, testa na pessoa deixada de fora).

## 6. Aumentar a quantidade/diversidade de dados melhora os resultados?

Esta pergunta so pode ser respondida comparando este relatorio com um novo relatorio gerado apos a coleta de mais amostras/pessoas: rode `python scripts/avaliacao.py` novamente depois de expandir o dataset e compare a secao 2 (comparacao entre modelos) e a secao 3 (metricas por letra) entre as duas versoes do relatorio.
