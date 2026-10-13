"""F4 - Fabrica dos modelos recorrentes: SimpleRNN, GRU e LSTM.

As tres familias sao construidas com a MESMA assinatura de entrada/saida e o mesmo
numero de camadas/unidades, para que a comparacao da frente 3.2 seja justa
(mesmos dados, mesma janela, mesma seed).

  entrada: (window, n_features)  ->  saida: Dense(1) = Total do dia seguinte
"""

import tensorflow as tf
from tensorflow import keras

CAMADAS = {
    "SimpleRNN": keras.layers.SimpleRNN,
    "GRU": keras.layers.GRU,
    "LSTM": keras.layers.LSTM,
}


def construir(params: dict, window: int, n_features: int) -> keras.Model:
    """Monta e compila um modelo recorrente a partir de um dicionario de hiperparametros."""
    tf.keras.utils.set_random_seed(int(params.get("seed", 42)))

    tipo = params["model_type"]
    if tipo not in CAMADAS:
        raise ValueError(f"model_type invalido: {tipo}")

    units = int(params["units"])
    n_layers = int(params.get("n_layers", 1))
    dropout = float(params.get("dropout", 0.0))
    rec_drop = float(params.get("recurrent_dropout", 0.0))

    entrada = keras.Input(shape=(window, n_features), name="janela")
    x = entrada
    for i in range(n_layers):
        x = CAMADAS[tipo](
            units,
            return_sequences=(i < n_layers - 1),   # so a ultima camada entrega o vetor final
            dropout=dropout,
            recurrent_dropout=rec_drop,
            name=f"{tipo}_{i + 1}",
        )(x)
    saida = keras.layers.Dense(1, name="total_t1")(x)

    modelo = keras.Model(entrada, saida, name=f"{tipo}_{n_layers}x{units}")

    lr = float(params.get("learning_rate", 1e-3))
    clip = params.get("clipnorm")                  # gradiente explosivo e comum em RNN
    if params.get("optimizer", "adam").lower() == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=lr, clipnorm=clip)
    else:
        opt = keras.optimizers.Adam(learning_rate=lr, clipnorm=clip)

    modelo.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return modelo
