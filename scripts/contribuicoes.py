"""
contribuicoes.py

Fluxo para ampliar a base de dados com amostras de outras pessoas.

Quem contribui coleta os sinais normalmente (tela Coletar, com o proprio
nome no campo "Pessoa") e depois EXPORTA as suas amostras para uma pasta
propria, contribuicoes/<nome>/, que e o que vai no Pull Request. Assim o
PR nunca mexe nos arquivos principais de data/ (que mudam toda hora e
dariam conflito), e o mantenedor decide quando incorporar cada contribuicao.

Uso:
    # quem contribui (tambem ha um botao na tela Coletar)
    python scripts/contribuicoes.py exportar --pessoa "Maria"

    # o mantenedor, depois de aceitar o PR
    python scripts/contribuicoes.py importar
    python scripts/contribuicoes.py importar --forcar   # reimporta mesmo se a pessoa ja estiver na base

Arquivos de cada contribuicao:
    contribuicoes/<nome>/letras.csv     -> mesclado em data/dados.csv
    contribuicoes/<nome>/numeros.csv    -> mesclado em data/dados_numeros.csv
    contribuicoes/<nome>/movimento.csv  -> mesclado em data/dados_dinamicos.csv
"""

from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

from coletor import COLUNAS_FINAIS, garantir_schema_z
from dinamico import COLUNAS_FINAIS_DINAMICAS

RAIZ = Path(__file__).resolve().parent.parent
PASTA_CONTRIBUICOES = RAIZ / "contribuicoes"

# nome do arquivo da contribuicao -> (dataset principal, colunas esperadas)
CONJUNTOS = {
    "letras.csv": (RAIZ / "data" / "dados.csv", COLUNAS_FINAIS),
    "numeros.csv": (RAIZ / "data" / "dados_numeros.csv", COLUNAS_FINAIS),
    "movimento.csv": (RAIZ / "data" / "dados_dinamicos.csv", COLUNAS_FINAIS_DINAMICAS),
}


def nome_da_pasta(pessoa):
    """'Maria Souza' -> 'maria-souza' (sem acentos, espacos ou simbolos)."""
    sem_acento = unicodedata.normalize("NFKD", pessoa).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", sem_acento.lower()).strip("-")
    return slug or "pessoa"


def _ler(caminho):
    return pd.read_csv(caminho, dtype={"label": str, "pessoa": str})


def exportar_contribuicao(pessoa, raiz=RAIZ):
    """Copia as amostras da pessoa (letras, numeros e movimento) para
    contribuicoes/<nome>/. Devolve (pasta, {arquivo: quantidade}).
    Arquivos ja existentes da mesma pessoa sao substituidos, porque a
    exportacao sempre leva TODAS as amostras dela."""
    pessoa = pessoa.strip()
    if not pessoa:
        raise ValueError("Informe o nome da pessoa.")
    pasta = Path(raiz) / "contribuicoes" / nome_da_pasta(pessoa)
    contagens = {}
    for nome_arquivo, (dataset, colunas) in CONJUNTOS.items():
        dataset = Path(raiz) / dataset.relative_to(RAIZ)
        if not dataset.exists():
            continue
        df = _ler(dataset)
        if "pessoa" not in df.columns:
            continue
        minhas = df[df["pessoa"].fillna("").str.strip().str.lower() == pessoa.lower()]
        if minhas.empty:
            continue
        pasta.mkdir(parents=True, exist_ok=True)
        for c in colunas:
            if c not in minhas.columns:
                minhas = minhas.assign(**{c: pd.NA})
        minhas[colunas].to_csv(pasta / nome_arquivo, index=False)
        contagens[nome_arquivo] = len(minhas)
    return pasta, contagens


def importar_contribuicoes(forcar=False, raiz=RAIZ):
    """Mescla todas as pastas de contribuicoes/ nos datasets de data/.
    Pula uma pessoa que ja esteja no dataset (para nao duplicar ao rodar
    duas vezes), a nao ser com forcar=True. Faz backup de cada dataset
    alterado. Devolve a lista de mensagens do que foi feito."""
    pasta_base = Path(raiz) / "contribuicoes"
    mensagens = []
    if not pasta_base.exists():
        return ["Nenhuma pasta contribuicoes/ encontrada."]

    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    for nome_arquivo, (dataset, colunas) in CONJUNTOS.items():
        dataset = Path(raiz) / dataset.relative_to(RAIZ)
        novas = []
        for arquivo in sorted(pasta_base.glob(f"*/{nome_arquivo}")):
            df = _ler(arquivo)
            faltando = [c for c in colunas if c not in df.columns]
            if faltando:
                mensagens.append(f"IGNORADO {arquivo.relative_to(raiz)}: formato inesperado "
                                 f"(faltam {len(faltando)} colunas).")
                continue
            novas.append((arquivo, df[colunas]))
        if not novas:
            continue

        if dataset.exists():
            if colunas is COLUNAS_FINAIS:
                garantir_schema_z(str(dataset))
            atual = _ler(dataset)
        else:
            atual = pd.DataFrame(columns=colunas)
        ja_presentes = set(atual["pessoa"].fillna("").str.strip().str.lower())

        adicionar = []
        for arquivo, df in novas:
            pessoas = set(df["pessoa"].fillna("").str.strip().str.lower())
            repetidas = pessoas & ja_presentes
            if repetidas and not forcar:
                mensagens.append(f"PULADO {arquivo.relative_to(raiz)}: {', '.join(sorted(repetidas))} "
                                 f"ja esta em {dataset.name} (use --forcar para importar mesmo assim).")
                continue
            adicionar.append(df)
            mensagens.append(f"OK {arquivo.relative_to(raiz)}: {len(df)} amostras -> {dataset.name}")

        if adicionar:
            if dataset.exists():
                backup = dataset.with_name(f"{dataset.stem}_backup_antes_importacao_{carimbo}.csv")
                shutil.copy2(dataset, backup)
            pd.concat([atual[colunas]] + adicionar, ignore_index=True).to_csv(dataset, index=False)
    if not mensagens:
        mensagens.append("Nenhuma contribuicao encontrada em contribuicoes/.")
    return mensagens


def main():
    parser = argparse.ArgumentParser(description="Exporta ou importa contribuicoes de dados.")
    sub = parser.add_subparsers(dest="acao", required=True)
    p_exp = sub.add_parser("exportar", help="exporta as amostras de uma pessoa para contribuicoes/<nome>/")
    p_exp.add_argument("--pessoa", required=True, help="nome usado na coleta (campo Pessoa)")
    p_imp = sub.add_parser("importar", help="mescla contribuicoes/ nos datasets de data/")
    p_imp.add_argument("--forcar", action="store_true", help="importa mesmo se a pessoa ja estiver na base")
    args = parser.parse_args()

    if args.acao == "exportar":
        pasta, contagens = exportar_contribuicao(args.pessoa)
        if not contagens:
            print(f"Nenhuma amostra de '{args.pessoa}' encontrada em data/.")
            return
        for nome, qtd in contagens.items():
            print(f"{nome}: {qtd} amostras")
        print(f"Pronto! Envie a pasta {pasta.relative_to(RAIZ)} no seu Pull Request.")
    else:
        for msg in importar_contribuicoes(forcar=args.forcar):
            print(msg)
        print("\nDepois de importar, treine de novo na tela Treinar (ou rode scripts/treinamento.py).")


if __name__ == "__main__":
    main()
