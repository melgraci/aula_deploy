import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Inicializa a aplicação Flask
app = Flask(__name__)
CORS(app) # Permite requisições de origens DIFERENTES

# PREENCHA A LACUNA ABAIXO (Carregar os arquivos salvos):
model = pickle.load(open("model.pkl","rb"))
names = pickle.load(open("names.pkl","rb"))

# Rota principal para testar se a API está online
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API Online",
        "message": "API de Machine Learning (Iris Classifier) em execução!",
        "versao": "1.0"
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    dados = request.get_json(force=True)   
    # Exemplo de dados recebidos: {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    valores_entrada = list(dados.values())
    array_features = np.array([valores_entrada])
    predicao_cod = model.pred(array_features)
    nome_especie = names[predicao_cod[0]]
    return jsonify({
        "codigo": int(predicao_cod[0]),
        "dados_recebidos": dados,
        "predicao": nome_especie
    })


if __name__ == "__main__":
    app.run(port=5000)
