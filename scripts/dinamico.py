"""
dinamico.py

Suporte compartilhado para as letras com MOVIMENTO do alfabeto de Libras:
J, X, Z e Ç. Diferente das letras estaticas (uma unica configuracao de
mao), essas letras sao reconhecidas pela TRAJETORIA da mao ao longo de
um pequeno intervalo de tempo, entao uma unica foto nao basta.

Por isso elas precisam de:
  - uma coleta diferente: em vez de UM quadro, uma JANELA de varios
    quadros seguidos (uma pequena sequencia de video);
  - um dataset proprio (data/dados_dinamicos.csv), com uma coluna por
    coordenada POR QUADRO da janela, em vez de uma coluna por
    coordenada so;
  - um modelo proprio (models/modelo_libras_dinamico.pkl), treinado
    sobre essas sequencias achatadas (flatten) em vez de sobre fotos
    unicas.

Sobre o gatilho de gravacao
----------------------------
A deteccao de QUANDO comecar/parar de gravar e feita por um gatilho
MANUAL (o usuario clica em um botao, o sistema conta um pequeno
"prepare-se" e depois grava os proximos NUM_QUADROS_JANELA quadros) em
vez de deteccao automatica de inicio/fim do movimento. Essa e uma
simplificacao deliberada: detectar sozinho onde um movimento comeca e
termina e um problema bem mais dificil (e sujeito a falso-positivo /
falso-negativo) do que vale a pena resolver neste projeto. Isso e
implementado na GUI (gui_app.py); este modulo so cuida dos dados.

A normalizacao espacial (translacao + escala) continua sendo aplicada
QUADRO A QUADRO, reaproveitando normalizar_landmarks() de
normalizacao.py - a diferenca e que agora concatenamos varios quadros
normalizados em vez de usar um so.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from normalizacao import NUM_COORDS, normalizar_landmarks

# Ç entra no alfabeto datilologico porque e usado na soletracao de
# palavras em portugues que tem cedilha (ex: "ação"). Nao faz parte do
# alfabeto A-Z "normal" por isso fica fora de LETRAS_ALFABETO
# (gui_app.py), e precisa ser adicionado a parte nos seletores da GUI.
LETRAS_DINAMICAS = ["J", "X", "Z", "Ç"]  # Ç

NUM_QUADROS_PREPARO = 25   # ~quadros de "prepare-se" antes de comecar a gravar
NUM_QUADROS_JANELA = 30    # quantos quadros sao gravados por amostra dinamica

COLUNAS_COORDS_DINAMICAS = [
    f"f{quadro}_p{ponto}_{eixo}"
    for quadro in range(NUM_QUADROS_JANELA)
    for ponto in range(21)
    for eixo in ("x", "y")
]
COLUNAS_FINAIS_DINAMICAS = COLUNAS_COORDS_DINAMICAS + ["pessoa", "label"]


def achatar_sequencia_bruta(sequencia_quadros):
    """Recebe uma lista de NUM_QUADROS_JANELA listas de 42 coordenadas
    brutas (uma por quadro, exatamente como vem de
    info['landmarks_brutos'] do motor_reconhecimento.py) e devolve uma
    unica lista achatada (flatten) com NUM_QUADROS_JANELA * 42 valores,
    na ordem quadro-a-quadro."""
    if len(sequencia_quadros) != NUM_QUADROS_JANELA:
        raise ValueError(
            f"Esperava {NUM_QUADROS_JANELA} quadros, recebi {len(sequencia_quadros)}."
        )
    achatado = []
    for quadro in sequencia_quadros:
        if len(quadro) != NUM_COORDS:
            raise ValueError(f"Cada quadro precisa ter {NUM_COORDS} coordenadas, recebi {len(quadro)}.")
        achatado.extend(quadro)
    return achatado


def normalizar_sequencia(sequencia_achatada):
    """Recebe uma sequencia JA achatada (NUM_QUADROS_JANELA * 42 valores
    brutos) e devolve a mesma sequencia com CADA quadro normalizado
    individualmente por normalizar_landmarks() (translacao + escala),
    mantendo a mesma ordem/tamanho. Usada tanto no treinamento quanto no
    reconhecimento em tempo real, para garantir que os dois usem
    exatamente o mesmo pre-processamento (o mesmo cuidado que
    normalizacao.py ja tomava para as letras estaticas)."""
    arr = np.asarray(sequencia_achatada, dtype=np.float64).reshape(NUM_QUADROS_JANELA, NUM_COORDS)
    normalizado = [normalizar_landmarks(quadro) for quadro in arr]
    return [valor for quadro in normalizado for valor in quadro]


def normalizar_dataframe_dinamico(df):
    """Equivalente a normalizar_dataframe_colunas() do normalizacao.py,
    so que para o dataset dinamico: normaliza cada LINHA (amostra)
    chamando normalizar_sequencia(), que por sua vez normaliza quadro a
    quadro. Usado pelo treinamento_dinamico.py para normalizar o dataset
    inteiro de uma vez."""
    valores = df[COLUNAS_COORDS_DINAMICAS].to_numpy(dtype=np.float64)
    normalizados = np.array([normalizar_sequencia(linha) for linha in valores])
    return pd.DataFrame(normalizados, columns=COLUNAS_COORDS_DINAMICAS, index=df.index)


def salvar_amostra_dinamica(sequencia_quadros, pessoa, letra, caminho="data/dados_dinamicos.csv"):
    """Salva UMA amostra dinamica (uma sequencia de NUM_QUADROS_JANELA
    quadros ja capturados pela GUI) no CSV de sinais dinamicos. Espelha
    coletor.salvar_amostra(), so que para sequencias em vez de quadros
    unicos - os dados continuam sendo salvos BRUTOS (sem normalizar),
    pelo mesmo motivo do dados.csv original: a normalizacao fica como
    etapa de pre-processamento no treino/inferencia, nao na coleta."""
    linha_achatada = achatar_sequencia_bruta(sequencia_quadros)
    linha = pd.DataFrame([linha_achatada + [pessoa, letra]], columns=COLUNAS_FINAIS_DINAMICAS)
    arquivo_existe = os.path.exists(caminho)
    linha.to_csv(caminho, mode='a', index=False, header=not arquivo_existe)


def remover_ultima_amostra_dinamica(caminho="data/dados_dinamicos.csv"):
    """Remove a ULTIMA linha do CSV de sinais dinamicos. Espelha
    coletor.remover_ultima_amostra(). Retorna True se removeu algo,
    False se o arquivo nao existe ou esta vazio."""
    if not os.path.exists(caminho):
        return False
    df = pd.read_csv(caminho)
    if df.empty:
        return False
    df.iloc[:-1].to_csv(caminho, index=False)
    return True
