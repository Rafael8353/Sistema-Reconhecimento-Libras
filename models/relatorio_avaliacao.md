# Relatorio de avaliacao - Sistema de Reconhecimento de Libras

Gerado em 12/11/2026 02:47:27

## 1. Resumo do dataset

- Total de amostras: 7303
- Letras/classes distintas: 23
- Pessoas distintas no dataset: 1 (Rafael)
- Atributos usados: 62 (42 coordenadas normalizadas + atributos geometricos)

### Amostras por letra

| Letra | Amostras |
|---|---|
| A | 243 |
| B | 246 |
| C | 241 |
| D | 246 |
| E | 254 |
| F | 515 |
| G | 335 |
| H | 254 |
| I | 364 |
| K | 274 |
| L | 407 |
| M | 371 |
| N | 364 |
| O | 264 |
| P | 263 |
| Q | 279 |
| R | 412 |
| S | 283 |
| T | 437 |
| U | 431 |
| V | 272 |
| W | 267 |
| Y | 281 |

## 2. Comparacao entre modelos (validacao cruzada 5-fold)

Mesmas metricas calculadas para todos os modelos, sobre o dataset inteiro:

| Modelo | Acuracia | F1 (macro) | Log Loss | AUC-ROC (OvR) |
|---|---|---|---|---|
| RandomForest | 99.44% (+/- 0.12) | 0.9950 | 0.0558 | 1.0000 |
| MLP (rede neural) | 99.11% (+/- 0.21) | 0.9925 | 0.0338 | 0.9999 |
| KNN (k=5) | 98.85% (+/- 0.26) | 0.9906 | 0.0929 | 0.9991 |

### 2.1 Validacao em blocos de tempo (sem vazamento de quadros vizinhos)

As amostras sao gravadas em sequencia e quadros vizinhos ficam quase identicos; na validacao aleatoria acima eles aparecem em treino e teste ao mesmo tempo, o que torna o resultado otimista. Aqui, cada dobra deixa de fora blocos inteiros de tempo (GroupKFold, 5 dobras), estimativa mais proxima do uso real (Random Forest):

- Acuracia (blocos de tempo): 97.95%
- F1 macro (blocos de tempo): 0.9826

| Classe | Precisao | Recall | F1-score | Suporte |
|---|---|---|---|---|
| A | 0.996 | 1.000 | 0.998 | 243 |
| B | 0.988 | 1.000 | 0.994 | 246 |
| C | 0.996 | 0.992 | 0.994 | 241 |
| D | 1.000 | 0.992 | 0.996 | 246 |
| E | 0.981 | 1.000 | 0.990 | 254 |
| F | 0.947 | 0.938 | 0.942 | 515 |
| G | 0.979 | 0.997 | 0.988 | 335 |
| H | 0.984 | 0.988 | 0.986 | 254 |
| I | 0.997 | 0.992 | 0.994 | 364 |
| K | 0.993 | 0.978 | 0.985 | 274 |
| L | 1.000 | 0.978 | 0.989 | 407 |
| M | 0.983 | 0.960 | 0.971 | 371 |
| N | 0.957 | 0.986 | 0.972 | 364 |
| O | 0.989 | 1.000 | 0.994 | 264 |
| P | 0.989 | 0.996 | 0.992 | 263 |
| Q | 1.000 | 0.989 | 0.995 | 279 |
| R | 0.973 | 0.973 | 0.973 | 412 |
| S | 0.989 | 0.996 | 0.993 | 283 |
| T | 0.936 | 0.936 | 0.936 | 437 |
| U | 0.963 | 0.968 | 0.965 | 431 |
| V | 0.952 | 0.949 | 0.950 | 272 |
| W | 1.000 | 1.000 | 1.000 | 267 |
| Y | 1.000 | 1.000 | 1.000 | 281 |

Confusoes mais frequentes na validacao em blocos de tempo:

| Classe real | Confundida com | Quantidade |
|---|---|---|
| F | T | 28 |
| T | F | 27 |
| M | N | 13 |
| V | U | 13 |
| U | R | 9 |
| R | V | 8 |
| L | G | 7 |
| N | M | 5 |

Veja `matriz_confusao_blocos` (imagem) para o mapa completo.

## 3. Metricas por letra (modelo Random Forest, conjunto de teste holdout 20%)

Log loss no holdout: 0.0551

| Letra | Precisao | Recall | F1-score | AUC-ROC | Suporte |
|---|---|---|---|---|---|
| A | 1.000 | 1.000 | 1.000 | 1.000 | 49 |
| B | 1.000 | 1.000 | 1.000 | 1.000 | 49 |
| C | 1.000 | 1.000 | 1.000 | 1.000 | 48 |
| D | 1.000 | 0.959 | 0.979 | 1.000 | 49 |
| E | 0.981 | 1.000 | 0.990 | 1.000 | 51 |
| F | 0.990 | 0.990 | 0.990 | 1.000 | 103 |
| G | 1.000 | 1.000 | 1.000 | 1.000 | 67 |
| H | 1.000 | 1.000 | 1.000 | 1.000 | 51 |
| I | 0.973 | 0.973 | 0.973 | 1.000 | 73 |
| K | 1.000 | 1.000 | 1.000 | 1.000 | 55 |
| L | 1.000 | 1.000 | 1.000 | 1.000 | 81 |
| M | 1.000 | 1.000 | 1.000 | 1.000 | 74 |
| N | 1.000 | 1.000 | 1.000 | 1.000 | 73 |
| O | 1.000 | 1.000 | 1.000 | 1.000 | 53 |
| P | 1.000 | 1.000 | 1.000 | 1.000 | 53 |
| Q | 1.000 | 1.000 | 1.000 | 1.000 | 56 |
| R | 0.976 | 0.988 | 0.982 | 1.000 | 82 |
| S | 0.966 | 1.000 | 0.983 | 1.000 | 57 |
| T | 1.000 | 0.989 | 0.994 | 1.000 | 88 |
| U | 0.988 | 0.965 | 0.976 | 1.000 | 86 |
| V | 0.982 | 1.000 | 0.991 | 1.000 | 54 |
| W | 1.000 | 1.000 | 1.000 | 1.000 | 53 |
| Y | 1.000 | 1.000 | 1.000 | 1.000 | 56 |

**Piores letras (menor F1-score):**

- I: F1 = 0.973
- U: F1 = 0.976
- D: F1 = 0.979
- R: F1 = 0.982
- S: F1 = 0.983

**Melhores letras (maior F1-score):**

- A: F1 = 1.000
- B: F1 = 1.000
- C: F1 = 1.000
- G: F1 = 1.000
- L: F1 = 1.000

## 4. Letras mais confundidas entre si

| Letra real | Confundida com | Quantidade de vezes (no teste) |
|---|---|---|
| D | I | 2 |
| I | S | 2 |
| U | R | 2 |
| F | E | 1 |
| R | U | 1 |
| T | F | 1 |
| U | V | 1 |

Veja a imagem `matriz_confusao.png` para o mapa completo de confusoes.

## 5. O desempenho varia de acordo com a pessoa?

**Ainda nao e possivel responder isso com confianca**: o dataset atual tem praticamente uma unica pessoa registrada (Rafael). O orientador pediu a coleta de sinais de 3 ou 4 pessoas diferentes justamente para permitir esta analise (testar se o modelo generaliza para maos/estilos diferentes, em vez de ter decorado as caracteristicas de uma unica pessoa). Assim que houver dados de mais pessoas, basta rodar este script novamente: essa secao passa a ser preenchida automaticamente com uma validacao 'deixe uma pessoa de fora' (treina com todas as pessoas menos uma, testa na pessoa deixada de fora).

## 6. Aumentar a quantidade/diversidade de dados melhora os resultados?

Esta pergunta so pode ser respondida comparando este relatorio com um novo relatorio gerado apos a coleta de mais amostras/pessoas: rode `python scripts/avaliacao.py` novamente depois de expandir o dataset e compare a secao 2 (comparacao entre modelos) e a secao 3 (metricas por letra) entre as duas versoes do relatorio.
