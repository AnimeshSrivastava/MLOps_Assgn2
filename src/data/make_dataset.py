# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from tensorflow.keras.datasets import boston_housing

class Dataset:
    def __init__(self,path="BostonHousing", img_size=(299,299),vocab_size=10000,seq_len=25,
                embed_dim=512,ff_dim=512,batch_size=64,epochs=30,) -> None:
        (train_fea, train_price),(test_fea, test_price) = boston_housing.load_data()
        print(f"train_fea.shape = {train_fea.shape}")
        print(f"train_price.shape = {train_price.shape}")
        print(f"train_fea [0] = {train_fea[0]}")

        # transform data
        scaler = StandardScaler()
        train_feaS = scaler.fit_transform(train_fea)
        test_feaS = scaler.fit_transform(test_fea)

        return train_feaS, test_feaS