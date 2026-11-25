"""
treinamento.py

Rotina de treinamento do classificador (Random Forest) usado pelo
main.py / gui_app.py, com registro de log de auditoria no SQLite.

Correcoes/melhorias em relacao a versao anterior
--------------------------------------------------
1) Normalizacao: as coordenadas brutas do CSV sao transformadas pela
   mesma funcao usada no motor de reconhecimento (normalizacao.py),
   tornando o modelo invariante a posicao da mao no quadro e a distancia
   da camera.
2) Split estratificado (stratify=y) no holdout 80/20.
3) Validacao cruzada aleatoria (StratifiedKFold, 5 folds) - pedido do
   orientador - E TAMBEM validacao em BLOCOS DE TEMPO (GroupKFold): quadros
   vizinhos ficam sempre do mesmo lado (treino ou teste). A validacao
   aleatoria e otimista porque quadros quase identicos caem em treino e
   teste; a de blocos e mais proxima do uso real (ver dataset_utils.py).
4) Atributos geometricos (distancias/angulos) e, quando o dataset tem a
   coordenada z (profundidade), atributos de profundidade (ver
   normalizacao.py). O modelo e salvo junto com a informacao de quais
   atributos usa.
5) Classificadores ESPECIALISTAS para pares confusos (F/T, M/N, G/L): se
   houver amostras COM z suficientes para as duas classes do par, treina
   um modelo binario com profundidade, usado como segunda etapa pelo
   motor de reconhecimento. Assim nao e preciso recoletar o alfabeto
   inteiro: basta recoletar (com z) os pares problematicos.
6) Modos: "letras" (dados.csv -> modelo_libras.pkl) e "numeros"
   (dados_numeros.csv -> modelo_libras_numeros.pkl). Ver modos.py.

Uso:
    python scripts/treinamento.py                  # letras
    python scripts/treinamento.py --modo numeros   # numeros 0-9

Para a avaliacao completa (F1, AUC-ROC, matriz de confusao, comparacao de
modelos), ver scripts/avaliacao.py.
"""

import argparse
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from dataset_utils import carregar_csv, grupos_temporais, linhas_com_z, matriz_features
from modos import MODOS

RAIZ = Path(__file__).resolve().parent.parent

# fracao minima de amostras com z para o modelo PRINCIPAL passar a usar z
FRACAO_MINIMA_Z_MODELO_PRINCIPAL = 0.95
# amostras com z necessarias, POR CLASSE, para treinar um especialista
MIN_AMOSTRAS_Z_ESPECIALISTA = 80


def _rf():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


class TreinadorIA:
    def __init__(self, db_name=None):
        self.db_name = db_name or str(RAIZ / "sistema_libras.db")
        self.criar_tabela_logs()

    def criar_tabela_logs(self):
        """Cria/atualiza o banco SQLite para gerenciar os historicos de
        treino. Todo o metodo e protegido contra erros de I/O: em alguns
        ambientes (ex: pasta do projeto acessada via um mount de
        rede/ponte remota) o SQLite pode nao conseguir criar o arquivo de
        journal que usa para escritas seguras. Se acontecer, avisamos e o
        treinamento continua sem o log de auditoria."""
        try:
            conn = sqlite3.connect(self.db_name, timeout=30)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs_treino (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_treino TEXT,
                    qtd_amostras INTEGER,
                    precisao REAL
                )
            ''')

            cursor.execute("PRAGMA table_info(logs_treino)")
            colunas_existentes = {linha[1] for linha in cursor.fetchall()}

            colunas_novas = {
                "cv_acuracia_media": "REAL",
                "cv_acuracia_desvio": "REAL",
                "num_classes": "INTEGER",
                "cv_blocos_acuracia_media": "REAL",
                "modo": "TEXT",
            }
            for nome, tipo in colunas_novas.items():
                if nome not in colunas_existentes:
                    cursor.execute(f"ALTER TABLE logs_treino ADD COLUMN {nome} {tipo}")

            conn.commit()
            conn.close()
        except sqlite3.OperationalError as e:
            print(f"Aviso: nao foi possivel preparar a tabela de logs no SQLite ({e}). "
                  f"O treinamento vai continuar normalmente, so o log de auditoria "
                  f"pode nao ficar disponivel desta vez.")

    # ------------------------------------------------------------------

    def _cv_blocos(self, X, y, grupos, n_splits=5):
        """Validacao cruzada em blocos de tempo. Devolve (media, desvio)
        das acuracias por dobra, ou (None, None) se nao houver grupos
        suficientes."""
        n_grupos = len(np.unique(grupos))
        n_splits = min(n_splits, n_grupos)
        if n_splits < 2:
            return None, None
        scores = cross_val_score(_rf(), X, y, cv=GroupKFold(n_splits=n_splits), groups=grupos,
                                 scoring="accuracy")
        return scores.mean(), scores.std()

    def _treinar_especialistas(self, df, tem_z, pares):
        """Treina um classificador binario COM profundidade (z) para cada
        par/grupo confuso que tenha amostras com z suficientes."""
        especialistas = {}
        for grupo in pares:
            grupo = tuple(grupo)
            mascara = df["label"].isin(grupo) & tem_z
            contagens = df.loc[mascara, "label"].value_counts()
            faltando = [c for c in grupo if contagens.get(c, 0) < MIN_AMOSTRAS_Z_ESPECIALISTA]
            if faltando:
                print(f"[Especialista {'/'.join(grupo)}] ignorado: faltam amostras COM z "
                      f"(minimo {MIN_AMOSTRAS_Z_ESPECIALISTA} por classe; "
                      f"tenho {dict(contagens)}). Recolete essas letras na tela 'Coletar'.")
                continue

            sub = df[mascara].reset_index(drop=True)
            X = matriz_features(sub, usar_geom=True, usar_z=True)
            y = sub["label"]
            grupos = grupos_temporais(sub)
            media, desvio = self._cv_blocos(X, y, grupos)

            # para comparar: mesmo par, sem z, mesmas amostras
            X_sem_z = matriz_features(sub, usar_geom=True, usar_z=False)
            media_sem_z, _ = self._cv_blocos(X_sem_z, y, grupos)

            txt_sem0 = f"{media_sem_z * 100:.1f}%" if media_sem_z is not None else "n/d"
            if media is not None and media_sem_z is not None and media < media_sem_z:
                print(f"[Especialista {'/'.join(grupo)}] descartado: com z ({media * 100:.1f}%) ficou pior "
                      f"do que sem z ({txt_sem0}) na validacao em blocos de tempo. Colete mais variacao.")
                continue

            modelo = _rf().fit(X, y)
            especialistas[grupo] = modelo
            txt_sem = f"{media_sem_z * 100:.1f}%" if media_sem_z is not None else "n/d"
            txt_com = f"{media * 100:.1f}% (+/- {desvio * 100:.1f})" if media is not None else "n/d"
            print(f"[Especialista {'/'.join(grupo)}] treinado com {len(sub)} amostras. "
                  f"Acuracia em blocos de tempo: {txt_com} COM z  vs  {txt_sem} SEM z.")
        return especialistas

    # ------------------------------------------------------------------

    def treinar(self, csv_file=None, modo="letras"):
        cfg = MODOS[modo]
        csv_file = csv_file or str(cfg["csv"])
        caminho_modelo = Path(cfg["modelo"])

        print(f"=== Treinamento - modo: {cfg['nome']} ===")

        # 1. Carregar os dados do CSV
        try:
            df = carregar_csv(csv_file)
        except FileNotFoundError:
            print(f"Erro: o arquivo {Path(csv_file).name} nao foi encontrado. "
                  f"Colete dados primeiro (tela 'Coletar')!")
            return

        if "label" not in df.columns:
            print("Erro: o CSV ainda esta no formato antigo (sem coluna 'label'/'pessoa').")
            print("Rode primeiro: python scripts/migrar_dataset.py --pessoa \"SeuNome\"")
            return

        if len(df) == 0 or df["label"].nunique() < 2:
            print("Erro: preciso de amostras de pelo menos 2 classes diferentes para treinar.")
            return

        tem_z = linhas_com_z(df)
        usar_z = bool(tem_z.mean() >= FRACAO_MINIMA_Z_MODELO_PRINCIPAL)
        print(f"{len(df)} amostras, {df['label'].nunique()} classes. "
              f"Amostras com z: {int(tem_z.sum())} ({tem_z.mean() * 100:.0f}%). "
              f"Modelo principal {'USA' if usar_z else 'nao usa'} z.")

        # 2. Atributos (coordenadas normalizadas + geometricos [+ z]).
        #    Array numpy (sem nomes de coluna) porque em tempo real a
        #    predicao e feita com uma lista simples de numeros.
        X = matriz_features(df, usar_geom=True, usar_z=usar_z)
        y = df["label"]
        grupos = grupos_temporais(df)

        menor_classe = int(y.value_counts().min())

        # 3. Holdout estratificado (80/20), so se todas as classes tiverem
        #    amostras suficientes
        precisao = float("nan")
        if menor_classe >= 5:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            modelo = _rf().fit(X_train, y_train)
            precisao = modelo.score(X_test, y_test)
            print(f"Acuracia no conjunto de teste (holdout 20% aleatorio): {precisao * 100:.2f}%")
        else:
            print("Holdout ignorado: alguma classe tem menos de 5 amostras.")

        # 4. Validacao cruzada aleatoria (a de antes) ...
        cv_media = cv_desvio = float("nan")
        if menor_classe >= 5:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores_cv = cross_val_score(_rf(), X, y, cv=cv, scoring="accuracy")
            cv_media, cv_desvio = scores_cv.mean(), scores_cv.std()
            print(f"Validacao cruzada ALEATORIA (5-fold): acuracia media = {cv_media * 100:.2f}% "
                  f"(desvio padrao = {cv_desvio * 100:.2f} pontos percentuais)")

        # ... e a validacao em BLOCOS DE TEMPO (mais realista)
        blocos_media, blocos_desvio = self._cv_blocos(X, y, grupos)
        if blocos_media is not None:
            print(f"Validacao cruzada em BLOCOS DE TEMPO (5-fold): acuracia media = "
                  f"{blocos_media * 100:.2f}% (desvio padrao = {blocos_desvio * 100:.2f} pontos percentuais)")
            print("  -> use este valor como estimativa mais honesta do desempenho em uso real; "
                  "a validacao aleatoria tende a ser otimista (quadros vizinhos quase iguais).")

        # 5. Modelo FINAL com todos os dados
        modelo_final = _rf().fit(X, y)

        # 6. Especialistas para pares confusos (so no modo letras e so se
        #    o modelo principal ainda nao usa z)
        especialistas = {}
        if not usar_z and cfg["especialistas"]:
            especialistas = self._treinar_especialistas(df, tem_z, cfg["especialistas"])

        pacote = {
            "versao": 2,
            "modo": modo,
            "model": modelo_final,
            "usa_geom": True,
            "usa_z": usar_z,
            "especialistas": especialistas,
        }

        caminho_modelo.parent.mkdir(exist_ok=True)
        with open(caminho_modelo, "wb") as f:
            pickle.dump(pacote, f)
        print(f"Modelo final salvo em {caminho_modelo} (treinado com {len(df)} amostras)")

        # 7. Salvar log no SQLite
        self.registrar_no_db(
            qtd=len(df), acc=precisao, cv_media=cv_media, cv_desvio=cv_desvio,
            num_classes=int(y.nunique()), cv_blocos=blocos_media, modo=modo,
        )

    def registrar_no_db(self, qtd, acc, cv_media, cv_desvio, num_classes, cv_blocos=None, modo="letras"):
        def _num(v):
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            data_atual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            cursor.execute(
                """INSERT INTO logs_treino
                   (data_treino, qtd_amostras, precisao, cv_acuracia_media, cv_acuracia_desvio,
                    num_classes, cv_blocos_acuracia_media, modo)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (data_atual, qtd, _num(acc), _num(cv_media), _num(cv_desvio), num_classes,
                 _num(cv_blocos), modo),
            )
            conn.commit()
            conn.close()
            print("Log de treinamento salvo no SQLite!")
        except sqlite3.OperationalError as e:
            print(f"Aviso: nao foi possivel gravar o log no SQLite ({e}). "
                  f"O modelo foi treinado e salvo normalmente; apenas o registro "
                  f"de auditoria nao foi gravado desta vez.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Treina o classificador estatico (letras ou numeros).")
    parser.add_argument("--modo", choices=list(MODOS), default="letras",
                        help="'letras' (padrao) ou 'numeros'")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    TreinadorIA().treinar(modo=args.modo)
