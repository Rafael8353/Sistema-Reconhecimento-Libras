import pandas as pd
import sqlite3
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

class TreinadorIA:
    def __init__(self, db_name="sistema_libras.db"):
        self.db_name = db_name
        self.criar_tabela_logs()

    def criar_tabela_logs(self):
        """Cria o banco SQLite para gerenciar os históricos de treino"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs_treino (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_treino TEXT,
                qtd_amostras INTEGER,
                precisao REAL
            )
        ''')
        conn.commit()
        conn.close()

    def treinar(self, csv_file="data/dados.csv"):
        # 1. Carregar os dados do CSV
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print("Erro: O arquivo dados.csv não foi encontrado. Colete dados primeiro!")
            return

        # 2. Separar X (pontos da mão) e y (a letra/label)
        X = df.iloc[:, :-1] # Todas as colunas exceto a última
        y = df.iloc[:, -1]  # Apenas a última coluna (a letra)

        # 3. Dividir em treino e teste (80% treina, 20% testa se aprendeu)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # 4. Criar o classificador (Random Forest)
        modelo = RandomForestClassifier(n_estimators=100)
        modelo.fit(X_train, y_train)

        # 5. Avaliar precisão
        precisao = modelo.score(X_test, y_test)
        print(f"Treino concluído! Precisão: {precisao * 100:.2f}%")

        # 6. Salvar o arquivo do "Cérebro" (Modelo Serializado)
        with open('models/modelo_libras.pkl', 'wb') as f:
            pickle.dump(modelo, f)

        # 7. Salvar log no SQLite
        self.registrar_no_db(len(df), precisao)

    def registrar_no_db(self, qtd, acc):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        data_atual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        cursor.execute("INSERT INTO logs_treino (data_treino, qtd_amostras, precisao) VALUES (?, ?, ?)", 
                       (data_atual, qtd, acc))
        conn.commit()
        conn.close()
        print("Log de treinamento salvo no SQLite!")

if __name__ == "__main__":
    treinador = TreinadorIA()
    treinador.treinar()