import os 
import sys

sys.path.append('/home/archana/projects/MLOps/src/')
from tensorflow import keras
from helper import TransformerEncoder, TransformerDecoder, BuildModel
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from model import Model

class TestModel:
    def __init__(self,vocab_size=10000,seq_len=20,embed_dim=512,ff_dim=512):
        self.data_path = "/home/archana/projects/MLOps/src/data/raw/"

    # def get_tokens_data(self):
    #     return self.dataset.buildCaptionsMap(self.tokens_path)

    def predict(self):
        Model.predict()