"""
avaliacao.py

Script de AVALIACAO cientifica do sistema, criado para atender as
orientacoes recebidas do orientador:

  1) Comparar mais de um modelo/algoritmo (Random Forest, KNN, MLP -
     os mesmos tipos usados nos trabalhos relacionados citados no TCC1:
     Loreto (2023) usou KNN, Silva (2022) usou MLP), registrando as
     MESMAS metricas para todos, permitindo comparacao justa.
  2) Usar validacao cruzada (cross-validation) em vez de depender de um
     unico split treino/teste.
  3) Ir alem da acuracia: calcular tambem loss (log loss), F1-score e
     AUC-ROC (multiclasse, one-vs-rest).
  4) Calcular metricas POR LETRA (nao so uma media global), para
     identificar quais letras o modelo confunde mais.
  5) Gerar uma matriz de confusao para visualizar quais letras sao
     confundidas entre si.
  6) Deixar pronta a analise "o desempenho varia por pessoa?" para
     quando o dataset tiver mais de uma pessoa (hoje o dataset e
     essencialmente de uma unica pessoa - o relatorio avisa isso
     explicitamente em vez de fingir uma analise que nao e possivel
     ainda).

Este script NAO substitui o treinamento.py (que gera o modelo usado em
producao pelo main.py). Ele existe para gerar o material de analise que
vai para o texto do TCC2 (tabelas, matriz de confusao, relatorio).

Atualizacao (TCC 2): alem da validacao cruzada aleatoria (a de antes), o
relatorio agora traz a validacao em BLOCOS DE TEMPO (quadros vizinhos
nunca ficam em treino e teste ao mesmo tempo), que e uma estimativa mais
honesta - ver dataset_utils.py. Tambem usa os atributos geometricos (e z,
quando o dataset tem) e aceita o modo "numeros".

Saidas geradas em models/ (modo numeros: mesmo nome + "_numeros"):
  - relatorio_avaliacao.md      -> relatorio em markdown com todas as tabelas
  - comparacao_modelos.csv      -> tabela comparativa entre os modelos (cross-validation)
  - metricas_por_letra.csv      -> precisao/recall/F1/suporte de cada letra (holdout)
  - matriz_confusao.png         -> imagem da matriz de confusao (holdout)
  - matriz_confusao_blocos.png  -> matriz de confusao da validacao em blocos de tempo

Uso:
    python scripts/avaliacao.py                  # letras
    python scripts/avaliacao.py --modo numeros   # numeros 0-9
"""

import argparse

import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # nao precisa de tela para gerar a imagem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize

from dataset_utils import carregar_csv, grupos_temporais, linhas_com_z, matriz_features
from modos import MODOS

RAIZ = Path(__file__).resolve().parent.parent
MODELS_DIR = RAIZ / "models"
DB_PATH = RAIZ / "sistema_libras.db"

MODELOS = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "MLP (rede neural)": MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=42),
}


def carregar_dataset(csv_file=None, modo="letras"):
    csv_file = csv_file or MODOS[modo]["csv"]
    try:
        df = carregar_csv(csv_file)
    except FileNotFoundError:
        raise SystemExit(f"Arquivo nao encontrado: {csv_file}. Colete dados primeiro.")
    if "label" not in df.columns:
        raise SystemExit(
            "O CSV ainda esta no formato antigo (sem coluna 'label'/'pessoa'). "
            "Rode primeiro: python scripts/migrar_dataset.py --pessoa \"SeuNome\""
        )
    # mesmos atributos do treinamento.py: coordenadas normalizadas +
    # geometricos (+ z, se quase todas as amostras tiverem z)
    usar_z = bool(linhas_com_z(df).mean() >= 0.95)
    X = matriz_features(df, usar_geom=True, usar_z=usar_z)
    y = df["label"]
    pessoas = df["pessoa"] if "pessoa" in df.columns else None
    return df, X, y, pessoas


def comparar_modelos_cv(X, y, n_splits=5):
    """Roda validacao cruzada estratificada para cada modelo e devolve um
    DataFrame comparativo com as MESMAS metricas para todos: acuracia,
    F1 macro, log loss e AUC-ROC (one-vs-rest, macro)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "acuracia": "accuracy",
        "f1_macro": "f1_macro",
        "log_loss": "neg_log_loss",  # sklearn so tem a versao negativa pronta; invertemos o sinal depois
        "auc_roc_ovr": "roc_auc_ovr",
    }

    linhas = []
    for nome_modelo, modelo in MODELOS.items():
        print(f"Rodando validacao cruzada ({n_splits}-fold) para: {nome_modelo} ...")
        resultado = cross_validate(modelo, X, y, cv=cv, scoring=scoring, n_jobs=1)
        linhas.append({
            "modelo": nome_modelo,
            "acuracia_media": resultado["test_acuracia"].mean(),
            "acuracia_desvio": resultado["test_acuracia"].std(),
            "f1_macro_media": resultado["test_f1_macro"].mean(),
            "f1_macro_desvio": resultado["test_f1_macro"].std(),
            "log_loss_media": -resultado["test_log_loss"].mean(),
            "auc_roc_ovr_media": resultado["test_auc_roc_ovr"].mean(),
        })

    return pd.DataFrame(linhas).sort_values("acuracia_media", ascending=False).reset_index(drop=True)


def avaliar_modelo_final_holdout(X, y):
    """Treina o modelo principal do projeto (Random Forest) em um split
    treino/teste estratificado e calcula metricas detalhadas por letra +
    matriz de confusao, para a analise qualitativa (quais letras
    confundem mais, etc)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    relatorio_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_relatorio = pd.DataFrame(relatorio_dict).transpose()

    classes = modelo.classes_
    y_test_bin = label_binarize(y_test, classes=classes)
    auc_por_classe = roc_auc_score(y_test_bin, y_proba, average=None, multi_class="ovr")
    df_relatorio.loc[[c for c in classes], "auc_roc"] = auc_por_classe

    loss_holdout = log_loss(y_test, y_proba, labels=classes)

    matriz = confusion_matrix(y_test, y_pred, labels=classes)

    return modelo, df_relatorio, matriz, classes, loss_holdout, y_test, y_pred


def avaliar_em_blocos(X, y, grupos, n_splits=5):
    """Validacao cruzada em BLOCOS DE TEMPO (GroupKFold): cada dobra deixa
    de fora blocos contiguos de quadros, entao nao ha quadros quase
    identicos em treino e teste ao mesmo tempo. Devolve um dict com
    acuracia, F1 macro, relatorio por classe e matriz de confusao, ou
    None se nao houver grupos suficientes."""
    n_splits = min(n_splits, len(np.unique(grupos)))
    if n_splits < 2:
        return None
    y_pred = cross_val_predict(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        X, y, cv=GroupKFold(n_splits=n_splits), groups=grupos,
    )
    classes = np.array(sorted(y.unique()))
    relatorio = pd.DataFrame(classification_report(y, y_pred, output_dict=True, zero_division=0)).transpose()
    return {
        "acuracia": float((y.to_numpy() == y_pred).mean()),
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "relatorio": relatorio,
        "matriz": confusion_matrix(y, y_pred, labels=classes),
        "classes": classes,
        "n_splits": n_splits,
    }


def pares_mais_confundidos(matriz, classes, top_n=8):
    """Retorna os pares de letras (real, prevista) mais confundidos entre
    si (fora da diagonal principal), para a analise 'existem letras que
    sao frequentemente confundidas?'."""
    pares = []
    for i, real in enumerate(classes):
        for j, prevista in enumerate(classes):
            if i != j and matriz[i, j] > 0:
                pares.append((real, prevista, int(matriz[i, j])))
    pares.sort(key=lambda x: x[2], reverse=True)
    return pares[:top_n]


def salvar_matriz_confusao_png(matriz, classes, caminho, titulo="Matriz de confusao - Random Forest (conjunto de teste)"):
    fig, ax = plt.subplots(figsize=(11, 11))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(titulo)
    plt.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def analisar_por_pessoa(df, X, y, pessoas):
    """Se o dataset tiver mais de uma pessoa, treina em N-1 pessoas e
    testa na pessoa restante (validacao 'leave-one-person-out'), o jeito
    correto de responder 'o modelo generaliza para outras pessoas, ou so
    decorou uma pessoa especifica?'. Se so houver uma pessoa no dataset,
    explica isso no relatorio em vez de fingir uma analise."""
    pessoas_unicas = sorted(pessoas.dropna().unique()) if pessoas is not None else []

    if len(pessoas_unicas) < 2:
        return None, pessoas_unicas

    linhas = []
    for pessoa_teste in pessoas_unicas:
        idx_teste = pessoas == pessoa_teste
        idx_treino = ~idx_teste

        if idx_teste.sum() == 0 or idx_treino.sum() == 0:
            continue

        modelo = RandomForestClassifier(n_estimators=200, random_state=42)
        modelo.fit(X[idx_treino.to_numpy()], y[idx_treino])
        acc = modelo.score(X[idx_teste.to_numpy()], y[idx_teste])
        linhas.append({"pessoa_teste": pessoa_teste, "amostras_teste": int(idx_teste.sum()), "acuracia": acc})

    return pd.DataFrame(linhas), pessoas_unicas


def registrar_comparacao_no_db(df_comparacao):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs_avaliacao (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_avaliacao TEXT,
                modelo TEXT,
                acuracia_media REAL,
                f1_macro_media REAL,
                log_loss_media REAL,
                auc_roc_ovr_media REAL
            )
        ''')
        data_atual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        for _, linha in df_comparacao.iterrows():
            cursor.execute(
                """INSERT INTO logs_avaliacao
                   (data_avaliacao, modelo, acuracia_media, f1_macro_media, log_loss_media, auc_roc_ovr_media)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (data_atual, linha["modelo"], linha["acuracia_media"], linha["f1_macro_media"],
                 linha["log_loss_media"], linha["auc_roc_ovr_media"]),
            )
        conn.commit()
        conn.close()
        print("Comparacao de modelos registrada no SQLite (tabela logs_avaliacao).")
    except sqlite3.OperationalError as e:
        print(f"Aviso: nao foi possivel gravar a comparacao no SQLite ({e}). "
              f"Os arquivos de relatorio (CSV/MD) foram gerados normalmente.")


def escrever_relatorio_md(caminho, df, df_comparacao, df_relatorio_letras, matriz, classes,
                           loss_holdout, pares_confundidos, df_pessoa, pessoas_unicas, blocos=None,
                           descricao_atributos=""):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("# Relatorio de avaliacao - Sistema de Reconhecimento de Libras\n\n")
        f.write(f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")

        f.write("## 1. Resumo do dataset\n\n")
        f.write(f"- Total de amostras: {len(df)}\n")
        f.write(f"- Letras/classes distintas: {df['label'].nunique()}\n")
        if "pessoa" in df.columns:
            f.write(f"- Pessoas distintas no dataset: {df['pessoa'].nunique()} ({', '.join(pessoas_unicas) if pessoas_unicas else df['pessoa'].unique()[0]})\n")
        if descricao_atributos:
            f.write(f"- Atributos usados: {descricao_atributos}\n")
        f.write("\n### Amostras por letra\n\n")
        contagem = df["label"].value_counts().sort_index()
        f.write("| Letra | Amostras |\n|---|---|\n")
        for letra, qtd in contagem.items():
            f.write(f"| {letra} | {qtd} |\n")

        f.write("\n## 2. Comparacao entre modelos (validacao cruzada 5-fold)\n\n")
        f.write("Mesmas metricas calculadas para todos os modelos, sobre o dataset inteiro:\n\n")
        f.write("| Modelo | Acuracia | F1 (macro) | Log Loss | AUC-ROC (OvR) |\n")
        f.write("|---|---|---|---|---|\n")
        for _, linha in df_comparacao.iterrows():
            f.write(f"| {linha['modelo']} | {linha['acuracia_media']*100:.2f}% (+/- {linha['acuracia_desvio']*100:.2f}) "
                     f"| {linha['f1_macro_media']:.4f} | {linha['log_loss_media']:.4f} | {linha['auc_roc_ovr_media']:.4f} |\n")

        f.write("\n### 2.1 Validacao em blocos de tempo (sem vazamento de quadros vizinhos)\n\n")
        if blocos is None:
            f.write("Nao foi possivel calcular (poucos grupos de amostras).\n")
        else:
            f.write(
                "As amostras sao gravadas em sequencia e quadros vizinhos ficam quase identicos; "
                "na validacao aleatoria acima eles aparecem em treino e teste ao mesmo tempo, o que "
                "torna o resultado otimista. Aqui, cada dobra deixa de fora blocos inteiros de tempo "
                f"(GroupKFold, {blocos['n_splits']} dobras), estimativa mais proxima do uso real "
                "(Random Forest):\n\n"
            )
            f.write(f"- Acuracia (blocos de tempo): {blocos['acuracia']*100:.2f}%\n")
            f.write(f"- F1 macro (blocos de tempo): {blocos['f1_macro']:.4f}\n\n")
            rel = blocos["relatorio"]
            f.write("| Classe | Precisao | Recall | F1-score | Suporte |\n|---|---|---|---|---|\n")
            for letra in blocos["classes"]:
                linha = rel.loc[letra]
                f.write(f"| {letra} | {linha['precision']:.3f} | {linha['recall']:.3f} | "
                         f"{linha['f1-score']:.3f} | {int(linha['support'])} |\n")
            pares_b = pares_mais_confundidos(blocos["matriz"], blocos["classes"])
            if pares_b:
                f.write("\nConfusoes mais frequentes na validacao em blocos de tempo:\n\n")
                f.write("| Classe real | Confundida com | Quantidade |\n|---|---|---|\n")
                for real, prevista, qtd in pares_b:
                    f.write(f"| {real} | {prevista} | {qtd} |\n")
            f.write("\nVeja `matriz_confusao_blocos` (imagem) para o mapa completo.\n")

        f.write("\n## 3. Metricas por letra (modelo Random Forest, conjunto de teste holdout 20%)\n\n")
        f.write(f"Log loss no holdout: {loss_holdout:.4f}\n\n")
        f.write("| Letra | Precisao | Recall | F1-score | AUC-ROC | Suporte |\n|---|---|---|---|---|---|\n")
        for letra in classes:
            linha = df_relatorio_letras.loc[letra]
            f.write(f"| {letra} | {linha['precision']:.3f} | {linha['recall']:.3f} | {linha['f1-score']:.3f} "
                     f"| {linha.get('auc_roc', float('nan')):.3f} | {int(linha['support'])} |\n")

        acc_geral = df_relatorio_letras.loc["accuracy", "precision"] if "accuracy" in df_relatorio_letras.index else None
        f.write(f"\n**Piores letras (menor F1-score):**\n\n")
        piores = df_relatorio_letras.loc[list(classes)].sort_values("f1-score").head(5)
        for letra, linha in piores.iterrows():
            f.write(f"- {letra}: F1 = {linha['f1-score']:.3f}\n")

        f.write("\n**Melhores letras (maior F1-score):**\n\n")
        melhores = df_relatorio_letras.loc[list(classes)].sort_values("f1-score", ascending=False).head(5)
        for letra, linha in melhores.iterrows():
            f.write(f"- {letra}: F1 = {linha['f1-score']:.3f}\n")

        f.write("\n## 4. Letras mais confundidas entre si\n\n")
        if pares_confundidos:
            f.write("| Letra real | Confundida com | Quantidade de vezes (no teste) |\n|---|---|---|\n")
            for real, prevista, qtd in pares_confundidos:
                f.write(f"| {real} | {prevista} | {qtd} |\n")
        else:
            f.write("Nenhuma confusao relevante encontrada no conjunto de teste.\n")
        f.write("\nVeja a imagem `matriz_confusao.png` para o mapa completo de confusoes.\n")

        f.write("\n## 5. O desempenho varia de acordo com a pessoa?\n\n")
        if df_pessoa is None:
            f.write(
                "**Ainda nao e possivel responder isso com confianca**: o dataset atual tem "
                f"praticamente uma unica pessoa registrada ({', '.join(pessoas_unicas) if pessoas_unicas else 'desconhecida'}). "
                "O orientador pediu a coleta de sinais de 3 ou 4 pessoas diferentes justamente para "
                "permitir esta analise (testar se o modelo generaliza para maos/estilos diferentes, "
                "em vez de ter decorado as caracteristicas de uma unica pessoa). Assim que houver dados "
                "de mais pessoas, basta rodar este script novamente: essa secao passa a ser preenchida "
                "automaticamente com uma validacao 'deixe uma pessoa de fora' (treina com todas as "
                "pessoas menos uma, testa na pessoa deixada de fora).\n"
            )
        else:
            f.write("Validacao 'deixe uma pessoa de fora' (treina com as demais pessoas, testa na pessoa indicada):\n\n")
            f.write("| Pessoa (teste) | Amostras de teste | Acuracia |\n|---|---|---|\n")
            for _, linha in df_pessoa.iterrows():
                f.write(f"| {linha['pessoa_teste']} | {linha['amostras_teste']} | {linha['acuracia']*100:.2f}% |\n")

        f.write("\n## 6. Aumentar a quantidade/diversidade de dados melhora os resultados?\n\n")
        f.write(
            "Esta pergunta so pode ser respondida comparando este relatorio com um novo relatorio "
            "gerado apos a coleta de mais amostras/pessoas: rode `python scripts/avaliacao.py` "
            "novamente depois de expandir o dataset e compare a secao 2 (comparacao entre modelos) "
            "e a secao 3 (metricas por letra) entre as duas versoes do relatorio.\n"
        )

    print(f"Relatorio salvo em: {caminho}")


def main(modo="letras"):
    cfg = MODOS[modo]
    sufixo = cfg["sufixo_arquivos"]
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"=== Avaliacao - modo: {cfg['nome']} ===")
    print("Carregando o dataset e calculando os atributos...")
    df, X, y, pessoas = carregar_dataset(modo=modo)
    print(f"{len(df)} amostras, {y.nunique()} classes distintas, {X.shape[1]} atributos por amostra.")

    usa_z = bool(linhas_com_z(df).mean() >= 0.95)
    descricao_atributos = (
        f"{X.shape[1]} (42 coordenadas normalizadas + atributos geometricos"
        f"{' + profundidade z' if usa_z else ''})"
    )

    df_comparacao = comparar_modelos_cv(X, y)
    df_comparacao.to_csv(MODELS_DIR / f"comparacao_modelos{sufixo}.csv", index=False)
    print("\n=== Comparacao entre modelos (cross-validation aleatoria) ===")
    print(df_comparacao.to_string(index=False))

    print("\nValidacao em BLOCOS DE TEMPO (Random Forest)...")
    grupos = grupos_temporais(df)
    blocos = avaliar_em_blocos(X, y, grupos)
    if blocos is not None:
        print(f"Acuracia em blocos de tempo: {blocos['acuracia'] * 100:.2f}%  |  F1 macro: {blocos['f1_macro']:.4f}")
        salvar_matriz_confusao_png(
            blocos["matriz"], blocos["classes"], MODELS_DIR / f"matriz_confusao_blocos{sufixo}.png",
            titulo="Matriz de confusao - Random Forest (validacao em blocos de tempo)")
        print(f"Matriz de confusao (blocos) salva em: {MODELS_DIR / f'matriz_confusao_blocos{sufixo}.png'}")

    print("\nAvaliando o modelo Random Forest em detalhe (holdout 20%)...")
    modelo, df_relatorio_letras, matriz, classes, loss_holdout, y_test, y_pred = avaliar_modelo_final_holdout(X, y)
    df_relatorio_letras.to_csv(MODELS_DIR / f"metricas_por_letra{sufixo}.csv")

    salvar_matriz_confusao_png(matriz, classes, MODELS_DIR / f"matriz_confusao{sufixo}.png")
    print(f"Matriz de confusao salva em: {MODELS_DIR / f'matriz_confusao{sufixo}.png'}")

    pares_confundidos = pares_mais_confundidos(matriz, classes)

    print("\nAnalisando desempenho por pessoa (se houver mais de uma no dataset)...")
    df_pessoa, pessoas_unicas = analisar_por_pessoa(df, X, y, pessoas)

    escrever_relatorio_md(
        MODELS_DIR / f"relatorio_avaliacao{sufixo}.md",
        df, df_comparacao, df_relatorio_letras, matriz, classes,
        loss_holdout, pares_confundidos, df_pessoa, pessoas_unicas,
        blocos=blocos, descricao_atributos=descricao_atributos,
    )

    registrar_comparacao_no_db(df_comparacao)

    print("\nConcluido! Arquivos gerados em models/:")
    print(f"  - relatorio_avaliacao{sufixo}.md")
    print(f"  - comparacao_modelos{sufixo}.csv")
    print(f"  - metricas_por_letra{sufixo}.csv")
    print(f"  - matriz_confusao{sufixo}.png")
    print(f"  - matriz_confusao_blocos{sufixo}.png")


def _parse_args():
    parser = argparse.ArgumentParser(description="Avalia o classificador estatico (letras ou numeros).")
    parser.add_argument("--modo", choices=list(MODOS), default="letras", help="'letras' (padrao) ou 'numeros'")
    return parser.parse_args()


if __name__ == "__main__":
    main(modo=_parse_args().modo)
