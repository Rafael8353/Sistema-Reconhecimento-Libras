"""
migrar_dataset.py

Script de migracao (rodar UMA VEZ) para atualizar o data/dados.csv antigo
(colunas numericas "0".."41" + rotulo em "42", sem informacao de quem
coletou) para o novo schema usado pelo restante do projeto a partir de
agora:

    p0_x, p0_y, p1_x, p1_y, ..., p20_x, p20_y, pessoa, label

Mudancas:
  - Renomeia as 42 colunas de coordenadas para nomes descritivos
    (p{indice}_x / p{indice}_y), mantendo a MESMA ordem e os MESMOS
    valores brutos (nada e recalculado aqui: a normalizacao acontece
    depois, em tempo de treino/predicao, via normalizacao.py).
  - Adiciona a coluna 'pessoa', preenchida com um valor padrao para as
    amostras ja coletadas (nao tinhamos essa informacao antes).
  - Renomeia a coluna de rotulo "42" para 'label'.

Faz backup do arquivo original antes de sobrescrever.

Uso:
    python scripts/migrar_dataset.py --pessoa "Rafael"
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CSV_PATH = RAIZ / "data" / "dados.csv"


def nomes_colunas_coords():
    nomes = []
    for i in range(21):
        nomes.append(f"p{i}_x")
        nomes.append(f"p{i}_y")
    return nomes


def main():
    parser = argparse.ArgumentParser(description="Migra data/dados.csv para o novo schema com coluna 'pessoa'.")
    parser.add_argument("--pessoa", default="Rafael", help="Nome a atribuir as amostras ja existentes (padrao: Rafael)")
    parser.add_argument("--csv", default=str(CSV_PATH), help="Caminho do CSV a migrar")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Arquivo nao encontrado: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "pessoa" in df.columns and "label" in df.columns:
        print("O CSV ja parece estar no novo schema (colunas 'pessoa' e 'label' presentes). Nada a fazer.")
        return

    if df.shape[1] != 43:
        print(f"Aviso: esperava 43 colunas (42 coords + 1 rotulo), encontrei {df.shape[1]}. Abortando por seguranca.")
        return

    # Backup do arquivo original
    backup_path = csv_path.with_name(csv_path.stem + "_backup_pre_migracao.csv")
    shutil.copy2(csv_path, backup_path)
    print(f"Backup salvo em: {backup_path}")

    colunas_coords_antigas = list(df.columns[:-1])
    coluna_label_antiga = df.columns[-1]

    novas_colunas_coords = nomes_colunas_coords()
    mapa_renomeio = dict(zip(colunas_coords_antigas, novas_colunas_coords))
    df = df.rename(columns=mapa_renomeio)
    df = df.rename(columns={coluna_label_antiga: "label"})

    df["pessoa"] = args.pessoa

    # Reordena: coords..., pessoa, label
    colunas_finais = novas_colunas_coords + ["pessoa", "label"]
    df = df[colunas_finais]

    df.to_csv(csv_path, index=False)
    print(f"Migracao concluida: {len(df)} amostras, pessoa='{args.pessoa}'.")
    print(f"Novas colunas: {list(df.columns)[:4]} ... {list(df.columns)[-4:]}")


if __name__ == "__main__":
    main()
