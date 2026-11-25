"""
remover_amostras.py

Remove do data/dados.csv (ou de outro dataset, via --csv, ex.: numeros em
data/dados_numeros.csv) as amostras de uma letra/numero especifico (e,
opcionalmente, de uma pessoa especifica), para o caso de voce querer
descartar uma leva de capturas ruins/malfeitas e coletar de novo do
zero para aquela letra.

Sempre faz backup do CSV antes de remover qualquer coisa.

Uso:
    # Remove so as amostras da letra Z feitas pelo Rafael
    python scripts/remover_amostras.py --letra Z --pessoa Rafael

    # Remove as amostras da letra Z de TODAS as pessoas (pede confirmacao)
    python scripts/remover_amostras.py --letra Z
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CSV_PATH = RAIZ / "data" / "dados.csv"


def remover_amostras(letra, pessoa=None, csv_path=CSV_PATH):
    """Remove do CSV as amostras de uma letra (e, opcionalmente, de uma
    pessoa especifica), com backup automatico antes de alterar. NAO pede
    confirmacao - quem chama decide isso (o main() abaixo pergunta no
    terminal com input(); a tela 'Treinar' da GUI, em gui_app.py,
    pergunta com uma caixa de dialogo antes de chamar esta funcao).

    Retorna (quantidade_removida, caminho_do_backup). Quando nenhuma
    amostra bate com o filtro, retorna (0, None) e nao mexe no arquivo.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"label": str, "pessoa": str})
    if "label" not in df.columns:
        raise ValueError(
            "O CSV ainda esta no formato antigo (sem coluna 'label'/'pessoa'). "
            "Rode primeiro scripts/migrar_dataset.py."
        )

    letra = letra.strip().upper()
    if pessoa:
        mascara_remover = (df["label"] == letra) & (df["pessoa"].str.lower() == pessoa.strip().lower())
    else:
        mascara_remover = (df["label"] == letra)

    qtd_remover = int(mascara_remover.sum())
    if qtd_remover == 0:
        return 0, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = csv_path.with_name(f"{csv_path.stem}_backup_antes_remocao_{timestamp}.csv")
    shutil.copy2(csv_path, backup_path)

    df_restante = df[~mascara_remover].reset_index(drop=True)
    df_restante.to_csv(csv_path, index=False)

    return qtd_remover, backup_path


def main():
    parser = argparse.ArgumentParser(description="Remove amostras de uma letra (e opcionalmente pessoa) do dataset.")
    parser.add_argument("--letra", required=True, help="Letra/sinal cujas amostras serao removidas (ex: Z)")
    parser.add_argument("--pessoa", default=None, help="Se informado, remove so as amostras dessa pessoa para a letra. Se omitido, remove de TODAS as pessoas.")
    parser.add_argument("--csv", default=str(CSV_PATH), help="Caminho do CSV")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    letra = args.letra.strip().upper()

    if not args.pessoa:
        if not csv_path.exists():
            print(f"Arquivo nao encontrado: {csv_path}")
            return
        df_preview = pd.read_csv(csv_path, dtype={"label": str, "pessoa": str})
        qtd_preview = int((df_preview["label"] == letra).sum()) if "label" in df_preview.columns else 0
        resposta = input(f"Voce nao informou --pessoa: isso vai remover {qtd_preview} amostras "
                          f"da letra '{letra}' de TODAS as pessoas do dataset. Confirma? [s/N] ")
        if resposta.strip().lower() != "s":
            print("Cancelado.")
            return

    try:
        qtd_remover, backup_path = remover_amostras(letra, args.pessoa, csv_path=csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        return

    descricao = (f"letra '{letra}' da pessoa '{args.pessoa}'" if args.pessoa
                 else f"letra '{letra}' de TODAS as pessoas")

    if qtd_remover == 0:
        print(f"Nenhuma amostra encontrada para {descricao}. Nada foi alterado.")
        return

    print(f"Backup salvo em: {backup_path}")
    print(f"Removidas {qtd_remover} amostras ({descricao}).")
    print("\nProximos passos: colete amostras novas se quiser (scripts/coletor.py) e depois rode:")
    print("  python scripts/treinamento.py")
    print("  python scripts/avaliacao.py")


if __name__ == "__main__":
    main()
