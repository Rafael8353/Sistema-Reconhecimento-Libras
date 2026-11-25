"""
modos.py

Configuracao dos "modos" de reconhecimento ESTATICO do sistema. Cada modo
tem o seu proprio dataset, o seu proprio modelo e o seu proprio conjunto
de rotulos:

  - "letras"  : alfabeto datilologico estatico (A-Z, sem os sinais com
                movimento J, X, Z, Ç - esses usam o pipeline dinamico).
  - "numeros" : numerais 0 a 9.

Por que um modelo separado para os numeros? Varios numerais usam a
mesma configuracao de mao de uma letra (ex.: 2 e V, 0 e O, 1 e D). Com
modelos separados, quando o usuario escolhe "Numeros" o sistema so
considera os numeros (e vice-versa), o que elimina essa ambiguidade -
mesma ideia usada para os sinais dinamicos.

Os sinais DINAMICOS (J, X, Z, Ç) continuam com o pipeline proprio em
dinamico.py/treinamento_dinamico.py.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

NUMEROS = [str(n) for n in range(10)]

MODOS = {
    "letras": {
        "nome": "Letras",
        "rotulo_singular": "Letra",
        "csv": RAIZ / "data" / "dados.csv",
        "modelo": RAIZ / "models" / "modelo_libras.pkl",
        "sufixo_arquivos": "",  # relatorios: relatorio_avaliacao.md, ...
        # pares/grupos de sinais que so se distinguem bem com a profundidade
        # (z): quando ha amostras com z suficientes, treina-se um
        # classificador "especialista" extra para cada grupo
        "especialistas": [("F", "T"), ("M", "N"), ("G", "L")],
    },
    "numeros": {
        "nome": "Numeros",
        "rotulo_singular": "Numero",
        "csv": RAIZ / "data" / "dados_numeros.csv",
        "modelo": RAIZ / "models" / "modelo_libras_numeros.pkl",
        "sufixo_arquivos": "_numeros",
        "especialistas": [],
    },
}


def modo_do_rotulo(rotulo):
    """'5' -> 'numeros'; qualquer outro rotulo estatico -> 'letras'."""
    return "numeros" if str(rotulo).strip() in NUMEROS else "letras"
