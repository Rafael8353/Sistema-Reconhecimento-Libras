"""
dataset_utils.py

Funcoes compartilhadas para ler os datasets estaticos (dados.csv,
dados_numeros.csv), montar a matriz de atributos e gerar os grupos usados
na validacao em BLOCOS DE TEMPO.

Por que validacao em blocos de tempo?
-------------------------------------
As amostras sao gravadas em sequencia, enquanto a pessoa segura o sinal
diante da camera; quadros vizinhos ficam quase identicos. Em uma divisao
aleatoria (StratifiedKFold com shuffle), quadros vizinhos caem em treino
E em teste ao mesmo tempo, e o modelo "acerta" por ja ter visto uma copia
quase igual. Em um teste feito sobre o dataset deste projeto, a acuracia
caiu de ~99,4% (divisao aleatoria) para ~97,0% quando blocos inteiros de
tempo ficaram de fora do treino. A validacao em blocos e mais proxima do
que acontece na pratica (sinal feito em outro momento, em outra postura).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from normalizacao import COLUNAS_XY, COLUNAS_Z, construir_features

BLOCOS_POR_SESSAO = 10


def carregar_csv(caminho):
    """Le o CSV garantindo que 'label' e 'pessoa' sejam texto (assim o
    rotulo '5' nao vira o numero 5)."""
    return pd.read_csv(caminho, dtype={"label": str, "pessoa": str})


def linhas_com_z(df):
    """Mascara booleana: linhas que tem os 21 valores de z preenchidos."""
    if not all(c in df.columns for c in COLUNAS_Z):
        return pd.Series(False, index=df.index)
    return df[COLUNAS_Z].notna().all(axis=1)


def matriz_features(df, usar_geom=True, usar_z=False):
    """Matriz (n_amostras, n_atributos) calculada com construir_features()
    - a MESMA funcao usada na predicao em tempo real."""
    xy = df[COLUNAS_XY].to_numpy(dtype=np.float64)
    z = df[COLUNAS_Z].to_numpy(dtype=np.float64) if usar_z else None
    linhas = [
        construir_features(xy[i], z[i] if usar_z else None, usar_geom=usar_geom)
        for i in range(len(df))
    ]
    return np.asarray(linhas, dtype=np.float64)


def grupos_temporais(df, blocos_por_sessao=BLOCOS_POR_SESSAO):
    """Atribui a cada linha um grupo = (sessao de coleta, bloco de tempo).

    Uma 'sessao' e uma sequencia CONTIGUA de linhas com o mesmo rotulo
    (o coletor grava as amostras na ordem em que sao capturadas). Cada
    sessao e dividida em `blocos_por_sessao` blocos contiguos; o
    GroupKFold deixa blocos inteiros de fora do treino em cada dobra.
    """
    rotulos = df["label"].reset_index(drop=True)
    sessao = (rotulos != rotulos.shift()).cumsum().to_numpy()
    posicao = pd.Series(sessao).groupby(sessao).cumcount().to_numpy()
    tamanho = pd.Series(sessao).groupby(sessao).transform("size").to_numpy()
    bloco = np.minimum(posicao * blocos_por_sessao // tamanho, blocos_por_sessao - 1)
    return sessao * 1000 + bloco
