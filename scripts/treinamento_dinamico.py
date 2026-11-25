"""
treinamento_dinamico.py

Treinamento do classificador para os sinais DINAMICOS do alfabeto de
Libras (J, X, Z e Ç), que dependem de movimento e por isso tem um
dataset e um modelo proprios - ver dinamico.py para a explicacao
completa de por que isso e separado do treinamento.py normal.

Mesma logica do treinamento.py (split estratificado 80/20 + Random
Forest + log de auditoria no SQLite), so que aplicada ao dataset
achatado (flatten) de sequencias em vez de quadros unicos. A validacao
cruzada (StratifiedKFold de 5 dobras) so roda se houver pelo menos 5
amostras da letra com menos amostras - como o dataset dinamico comeca
pequeno (poucas letras, coleta mais trabalhosa que a estatica), exigir
isso direto evita um erro confuso do scikit-learn ("n_splits=5 cannot be
greater than the number of members in each class").

Uso:
    python scripts/treinamento_dinamico.py
"""

import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from dinamico import normalizar_dataframe_dinamico

RAIZ = Path(__file__).resolve().parent.parent


class TreinadorDinamicoIA:
    def __init__(self, db_name=None):
        self.db_name = db_name or str(RAIZ / "sistema_libras.db")
        self.criar_tabela_logs()

    def criar_tabela_logs(self):
        """Mesma logica defensiva de treinamento.py: se o SQLite nao
        conseguir criar/alterar a tabela (ex: pasta acessada via mount de
        rede/ponte remota), avisamos e o treinamento continua sem o log
        de auditoria."""
        try:
            conn = sqlite3.connect(self.db_name, timeout=30)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs_treino_dinamico (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_treino TEXT,
                    qtd_amostras INTEGER,
                    precisao REAL,
                    cv_acuracia_media REAL,
                    cv_acuracia_desvio REAL,
                    num_classes INTEGER
                )
            ''')
            conn.commit()
            conn.close()
        except sqlite3.OperationalError as e:
            print(f"Aviso: nao foi possivel preparar a tabela de logs no SQLite ({e}). "
                  f"O treinamento vai continuar normalmente, so o log de auditoria "
                  f"pode nao ficar disponivel desta vez.")

    def treinar(self, csv_file=None):
        csv_file = csv_file or str(RAIZ / "data" / "dados_dinamicos.csv")

        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print("Erro: data/dados_dinamicos.csv nao encontrado. Colete amostras de "
                  "J, X, Z ou Ç primeiro, na tela 'Coletar'.")
            return

        if "label" not in df.columns:
            print("Erro: o CSV dinamico esta num formato inesperado (sem coluna 'label').")
            return

        classes_presentes = sorted(df["label"].unique())
        if len(classes_presentes) < 2:
            print(f"Erro: so ha amostras de {classes_presentes} no dataset dinamico. "
                  f"E preciso pelo menos 2 letras diferentes (ex: J e Z) para treinar "
                  f"um classificador.")
            return

        # 1. Normalizar (quadro a quadro) e separar entrada/saida
        X = normalizar_dataframe_dinamico(df).to_numpy()
        y = df["label"]

        # 2. Split estratificado (80% treino, 20% teste)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 3. Treinar o classificador (Random Forest) no split de treino
        modelo = RandomForestClassifier(n_estimators=200, random_state=42)
        modelo.fit(X_train, y_train)
        precisao = modelo.score(X_test, y_test)
        print(f"Treino concluido! Acuracia no conjunto de teste (holdout 20%): {precisao * 100:.2f}%")

        # 4. Validacao cruzada (5-fold), so se houver amostras suficientes
        #    por classe - ver docstring do modulo.
        contagem_por_classe = y.value_counts()
        cv_media = cv_desvio = None
        if contagem_por_classe.min() >= 5:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            modelo_cv = RandomForestClassifier(n_estimators=200, random_state=42)
            scores_cv = cross_val_score(modelo_cv, X, y, cv=cv, scoring="accuracy")
            cv_media, cv_desvio = float(scores_cv.mean()), float(scores_cv.std())
            print(f"Validacao cruzada (5-fold): acuracia media = {cv_media * 100:.2f}% "
                  f"(desvio padrao = {cv_desvio * 100:.2f} pontos percentuais)")
            print(f"Acuracias por fold: {[round(s * 100, 2) for s in scores_cv]}")
        else:
            print(f"Aviso: pulando validacao cruzada - a letra com menos amostras tem "
                  f"so {contagem_por_classe.min()} amostra(s) (precisa de pelo menos 5 "
                  f"por letra). Colete mais amostras para uma estimativa mais robusta.")

        # 5. Treinar o modelo FINAL com todos os dados disponiveis
        modelo_final = RandomForestClassifier(n_estimators=200, random_state=42)
        modelo_final.fit(X, y)

        modelos_dir = RAIZ / "models"
        modelos_dir.mkdir(exist_ok=True)
        caminho_modelo = modelos_dir / "modelo_libras_dinamico.pkl"
        with open(caminho_modelo, "wb") as f:
            pickle.dump(modelo_final, f)
        print(f"Modelo dinamico salvo em {caminho_modelo} (treinado com {len(df)} amostras, "
              f"letras: {', '.join(classes_presentes)})")

        # 6. Salvar log no SQLite
        self.registrar_no_db(
            qtd=len(df), acc=precisao, cv_media=cv_media, cv_desvio=cv_desvio,
            num_classes=int(y.nunique()),
        )

    def registrar_no_db(self, qtd, acc, cv_media, cv_desvio, num_classes):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            data_atual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            cursor.execute(
                """INSERT INTO logs_treino_dinamico
                   (data_treino, qtd_amostras, precisao, cv_acuracia_media, cv_acuracia_desvio, num_classes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (data_atual, qtd, acc, cv_media, cv_desvio, num_classes),
            )
            conn.commit()
            conn.close()
            print("Log de treinamento dinamico salvo no SQLite!")
        except sqlite3.OperationalError as e:
            print(f"Aviso: nao foi possivel gravar o log no SQLite ({e}). "
                  f"O modelo foi treinado e salvo normalmente; apenas o registro "
                  f"de auditoria nao foi gravado desta vez.")


if __name__ == "__main__":
    treinador = TreinadorDinamicoIA()
    treinador.treinar()
