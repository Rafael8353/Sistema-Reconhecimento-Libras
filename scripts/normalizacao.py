"""
normalizacao.py

Funcao compartilhada de pre-processamento das coordenadas da mao extraidas
pelo MediaPipe Hands.

Por que isso existe
--------------------
O coletor.py salva as coordenadas (x, y) de cada um dos 21 pontos da mao
exatamente como o MediaPipe devolve: normalizadas em relacao ao TAMANHO DA
IMAGEM (0.0 a 1.0), nao em relacao a propria mao. Isso significa que o
vetor de 42 numeros muda dependendo de:
  - em que parte do quadro a mao esta (esquerda/direita/centro);
  - a que distancia a mao esta da camera (mao "grande" perto vs "pequena"
    longe);
enquanto o que realmente importa para reconhecer a letra e o FORMATO da
mao (a posicao relativa dos dedos entre si).

Sem essa normalizacao, o Random Forest aprende, sem querer, a decorar
"a mao estava naquela regiao da tela com aquele tamanho" em vez de
aprender o formato do sinal. Isso explica o comportamento errático
relatado (o sistema falando letras diferentes quando a mao muda de
posicao/distancia da camera): pequenas variacoes de enquadramento jogam
o vetor de entrada para fora da regiao que o modelo aprendeu.

A solucao classica (e a mesma usada por boa parte dos projetos que usam
MediaPipe Hands + classificador tabular) e:
  1) Tornar as coordenadas invariantes a TRANSLACAO: subtrair a posicao do
     pulso (landmark 0) de todos os pontos, para que a mao "ancore" sempre
     na origem, nao importa onde esteja na tela.
  2) Tornar as coordenadas invariantes a ESCALA: dividir todos os pontos
     pela maior distancia entre o pulso e qualquer outro ponto da mao, para
     que uma mao grande (perto da camera) e uma mao pequena (longe da
     camera) gerem o mesmo vetor, desde que o FORMATO do gesto seja igual.

Essa funcao e usada tanto no treinamento.py/avaliacao.py (para treinar com
dados normalizados) quanto no main.py (para prever com o mesmo
pre-processamento). É fundamental que os dois lados usem exatamente a
mesma transformacao, senao o modelo recebe dados num "espaco" diferente
do que foi treinado.

Importante: essa normalizacao e aplicada A PARTIR das coordenadas brutas
que ja estao salvas no data/dados.csv. Ou seja, nao e necessario recoletar
nada: os 21 pontos brutos continuam sendo a fonte de verdade, e a
normalizacao vira uma etapa de pre-processamento (feature engineering)
tanto no treino quanto na inferencia.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
NUM_COORDS = NUM_LANDMARKS * 2  # 42 valores (x0, y0, x1, y1, ..., x20, y20)


def normalizar_landmarks(coords):
    """
    Recebe uma lista/array de 42 valores brutos (x0, y0, x1, y1, ..., x20,
    y20) exatamente como extraidos do MediaPipe (hand_lms.landmark) e
    devolve uma lista de 42 valores normalizados (invariantes a posicao
    da mao no quadro e a distancia da camera).

    Parametros
    ----------
    coords : list[float] | np.ndarray
        Vetor com 42 valores (21 pontos x 2 eixos).

    Retorno
    -------
    list[float]
        Vetor normalizado com 42 valores, mesma ordem de entrada.
    """
    pts = np.asarray(coords, dtype=np.float64).reshape(NUM_LANDMARKS, 2)

    # 1) Invariancia a translacao: ancora tudo no pulso (landmark 0)
    pulso = pts[0].copy()
    pts = pts - pulso

    # 2) Invariancia a escala: normaliza pela maior distancia entre o
    #    pulso (agora na origem) e qualquer outro ponto da mao.
    distancias = np.linalg.norm(pts, axis=1)
    escala = distancias.max()
    if escala > 1e-6:
        pts = pts / escala

    return pts.flatten().tolist()


def normalizar_dataframe_colunas(df, colunas_coords):
    """
    Aplica normalizar_landmarks() em cada linha de um DataFrame do pandas,
    retornando um novo DataFrame apenas com as colunas normalizadas (mesmos
    nomes de coluna de entrada, na mesma ordem).

    Usado pelo treinamento.py/avaliacao.py para normalizar o dataset
    inteiro de uma vez.
    """
    import pandas as pd  # import local para nao forcar pandas em quem so usa normalizar_landmarks (ex: main.py em tempo real)

    valores = df[colunas_coords].to_numpy(dtype=np.float64)
    normalizados = np.array([normalizar_landmarks(linha) for linha in valores])
    return pd.DataFrame(normalizados, columns=colunas_coords, index=df.index)


# ======================================================================
# ATRIBUTOS ADICIONAIS (geometria + profundidade z)  -  TCC 2
# ======================================================================
#
# Motivacao (diagnostico feito sobre o data/dados.csv): letras que so
# diferem pela posicao do polegar em relacao ao indicador (F/T, M/N,
# G/L) ficam quase iguais quando olhamos apenas as coordenadas (x, y),
# porque no plano da imagem as pontas dos dedos ficam proximas nos dois
# sinais; a diferenca real esta na PROFUNDIDADE (o polegar passa por
# cima ou por tras do indicador). O MediaPipe devolve, alem de x e y,
# uma coordenada z por ponto (profundidade relativa ao pulso), que antes
# nao era salva. Agora:
#
#   1) atributos GEOMETRICOS: distancias e angulos entre pontos
#      importantes, calculados a partir das coordenadas (x, y) ja
#      normalizadas. Servem para qualquer dataset, inclusive o antigo
#      (sem z), e ajudam o Random Forest a enxergar relacoes entre
#      poucos pontos que ficariam "diluidas" entre os 42 valores;
#   2) atributos de PROFUNDIDADE (z): os 21 valores de z divididos pela
#      mesma escala usada em (x, y) + algumas distancias 3D. So existem
#      em amostras coletadas depois desta atualizacao.
#
# O modelo guarda no proprio arquivo .pkl quais atributos usa (usa_geom,
# usa_z), e o motor de reconhecimento monta o mesmo vetor na predicao.

# pares (ponto A, ponto B) cujas distancias viram atributos
PARES_DISTANCIA = [
    (4, 8), (4, 6), (4, 7), (4, 5), (4, 12), (8, 12), (12, 16), (16, 20), (4, 20),
    (4, 0), (8, 0), (12, 0), (16, 0), (20, 0),
]
# trios (A, vertice B, C) cujos angulos (graus/180) viram atributos
TRIOS_ANGULO = [
    (2, 3, 4), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19), (6, 7, 8),
]
# pares cujas distancias 3D (x, y, z) viram atributos quando ha z
PARES_DISTANCIA_3D = [(4, 8), (4, 6), (4, 7), (4, 5)]

NUM_FEATURES_GEOM = len(PARES_DISTANCIA) + len(TRIOS_ANGULO)
NUM_FEATURES_Z = NUM_LANDMARKS + len(PARES_DISTANCIA_3D)

COLUNAS_XY = [f"p{i}_{eixo}" for i in range(NUM_LANDMARKS) for eixo in ("x", "y")]
COLUNAS_Z = [f"p{i}_z" for i in range(NUM_LANDMARKS)]


def _pts_normalizados(coords_xy):
    """Mesma normalizacao de normalizar_landmarks(), mas devolve tambem a
    escala usada (para normalizar o z com a mesma escala)."""
    pts = np.asarray(coords_xy, dtype=np.float64).reshape(NUM_LANDMARKS, 2)
    pts = pts - pts[0]
    escala = np.linalg.norm(pts, axis=1).max()
    if escala > 1e-6:
        pts = pts / escala
    else:
        escala = 1.0
    return pts, escala


def _angulo(pts, a, b, c):
    v1 = pts[a] - pts[b]
    v2 = pts[c] - pts[b]
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    cos = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos))) / 180.0


def features_geometricas(pts):
    """pts: array (21, 2) ja normalizado. Devolve NUM_FEATURES_GEOM valores."""
    dist = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in PARES_DISTANCIA]
    ang = [_angulo(pts, a, b, c) for a, b, c in TRIOS_ANGULO]
    return dist + ang


def construir_features(coords_xy, z=None, usar_geom=True):
    """Monta o vetor de atributos de UMA amostra.

    coords_xy : 42 valores brutos (x0, y0, ..., x20, y20).
    z         : 21 valores brutos de profundidade (ou None para nao usar z).
    usar_geom : inclui os atributos geometricos (distancias/angulos).

    Layout do vetor: [42 xy normalizados] + [NUM_FEATURES_GEOM geometricos,
    se usar_geom] + [NUM_FEATURES_Z de profundidade, se z foi informado].
    Treino e predicao DEVEM chamar esta mesma funcao com as mesmas opcoes.
    """
    pts, escala = _pts_normalizados(coords_xy)
    vetor = list(pts.flatten())
    if usar_geom:
        vetor.extend(features_geometricas(pts))
    if z is not None:
        zn = np.asarray(z, dtype=np.float64).reshape(NUM_LANDMARKS) / escala
        vetor.extend(zn.tolist())
        p3 = np.column_stack([pts, zn])
        vetor.extend(float(np.linalg.norm(p3[a] - p3[b])) for a, b in PARES_DISTANCIA_3D)
    return vetor
