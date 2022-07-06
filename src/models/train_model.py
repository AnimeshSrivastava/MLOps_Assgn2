import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Flatten
from keras import Input
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self,img_size=(299,299),vocab_size=10000,seq_len=20,
                embed_dim=512,ff_dim=512,batch_size=64,epochs=30,) -> None:
        self.data_path = "/home/archana/projects/MLOps/src/data/raw/"

    def buildModel(self):
        model_2 = keras.Sequential(name="model_2")
        model_2.add(keras.Input(shape=(13, )))
        model_2.add(keras.layers.Dense(64, activation=tf.nn.relu, name="layer-1"))
        model_2.add(keras.layers.Dense(64, activation=tf.nn.relu, name="layer-2"))
        model_2.add(keras.layers.Dense(1, name="layer-3"))

        model_2.compile(optimizer =keras.optimizers.RMSprop(),
              loss = keras.losses.mean_squared_error,
              metrics = ["mae"])

        k = 4       # 4- fold validation
        num_val_samples = len(train_feaS)//k  
        num_epochs = 300
        all_accuracy_histories = [] 
        all_loss_histories = []
        all_train_loss_histories = []
        all_train_accuracy_histories = []

        for i in range(k):
            print(f"processing split {i}")
            x_val = train_feaS[i * num_val_samples: (i + 1) * num_val_samples] 
            y_val = train_price[i * num_val_samples: (i + 1) * num_val_samples]

            partial_x_train = np.concatenate(
                                [train_feaS[:i * num_val_samples],
                                train_feaS[(i + 1) * num_val_samples:]],
                                axis=0)
            partial_y_train = np.concatenate(
                                [train_price[:i * num_val_samples],
                                train_price[(i + 1) * num_val_samples:]],
                                axis=0)
            model = model_2
            history = model.fit(train_feaS_train, train_price_train, validation_data=(train_feaS_val, train_price_val),epochs=num_epochs, batch_size=16, verbose=0)

            mae_history = history.history["val_mae"]
            loss_history = history.history["val_loss"]
            train_loss_history = history.history["loss"]
            train_mae_history = history.history["mae"]

            all_accuracy_histories.append(mae_history)
            all_loss_histories.append(loss_history)
            all_train_loss_histories.append(train_loss_history)
            all_train_accuracy_histories.append(train_mae_history)

            return model_2

        def predict():
            return self.buildModel()
