import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.utils import resample
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score
from tensorflow import keras
import pickle
import json
from .fusion_model_helper_functions import *
# import mlflow 

class TextLSTMModel:
    def __init__(self, max_words=10000, max_sequence_length=10):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        self.tokenizer.fit_on_texts(X_train["description"])

        tokenizer_config = self.tokenizer.to_json()
        with open("models/tokenizer_config.json", "w", encoding="utf-8") as json_file:
            json_file.write(tokenizer_config)

        train_sequences = self.tokenizer.texts_to_sequences(X_train["description"])
        train_padded_sequences = pad_sequences(
            train_sequences,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )

        val_sequences = self.tokenizer.texts_to_sequences(X_val["description"])
        val_padded_sequences = pad_sequences(
            val_sequences,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )
        
        text_input = Input(shape=(self.max_sequence_length,))
        text_model = Embedding(input_dim=self.max_words, output_dim=128)(text_input)
        text_model = LSTM(128, return_sequences=True)(text_model)
        text_model = Dropout(0.5)(text_model)
        text_model = LSTM(64, return_sequences=False)(text_model)
        text_model = Dropout(0.5)(text_model)
        text_model = BatchNormalization()(text_model)
        text_model = Dense(128, activation='relu')(text_model)
        text_model = BatchNormalization()(text_model)
        text_model = Dense(64, activation = 'relu')(text_model)
        text_model = Dropout(.5)(text_model)
        output = Dense(27, activation="softmax")(text_model)

        self.model = Model(inputs=text_input, outputs=output)

        



        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        lstm_callbacks = [
            ModelCheckpoint(
                filepath="models/best_lstm_model.keras", save_best_only=True
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
        ]

        self.model.fit(
            [train_padded_sequences],
            tf.keras.utils.to_categorical(y_train, num_classes=27),
            epochs=1,
            batch_size=32,
            validation_data=(
                [val_padded_sequences],
                tf.keras.utils.to_categorical(y_val, num_classes=27),
            ),
            callbacks=lstm_callbacks,
        )


class ImageVGG16Model:
    def __init__(self):
        self.model = None

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        # Paramètres
        batch_size = 32
        num_classes = 27

        df_train = pd.concat([X_train, y_train.astype(str)], axis=1)
        df_val = pd.concat([X_val, y_val.astype(str)], axis=1)

        # Créer un générateur d'images pour le set d'entraînement
        train_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),  # Adapter à la taille d'entrée de VGG16
            batch_size=batch_size,
            class_mode="categorical",  # Utilisez 'categorical' pour les entiers encodés en one-hot
            shuffle=True,
        )

        # Créer un générateur d'images pour le set de validation
        val_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,  # Pas de mélange pour le set de validation
        )

        img_input = Input(shape=(224,224,3), name='img_input')
        img_model = Conv2D(64, (3,3),1, activation='relu',kernel_regularizer=regularizers.l1(0.001))(img_input)
        img_model = MaxPooling2D()(img_model)
        img_model = Conv2D(32, (3,3), 1, activation='relu',kernel_regularizer=regularizers.l1(0.001))(img_model)
        img_model = MaxPooling2D()(img_model)
        img_model = Conv2D(16, (3,3), 1, activation='relu',kernel_regularizer=regularizers.l1(0.001))(img_model)
        img_model = MaxPooling2D()(img_model)
        img_model = Dropout(rate = .4)(img_model)
        img_model = Flatten()(img_model)
        img_model = Dense(64, activation='relu')(img_model)
        img_model = BatchNormalization()(img_model)
        img_model = Dropout(rate=.4)(img_model)
        output = Dense(27, activation="softmax")(img_model)

        self.model = Model(inputs=img_input, outputs=output)

        

        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        vgg_callbacks = [
            ModelCheckpoint(
                filepath="models/best_vgg16_model.keras", save_best_only=True
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
        ]

        self.model.fit(
            train_generator,
            epochs=1,
            validation_data=val_generator,
            callbacks=vgg_callbacks,
        )


class concatenate:
    def __init__(self, tokenizer, lstm, vgg16):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(
        self, X_train, y_train, new_samples_per_class=50, max_sequence_length=10
    ):
        num_classes = 27

        new_X_train = pd.DataFrame(columns=X_train.columns)
        new_y_train = pd.DataFrame(
            columns=[0]
        )  # Créez la structure pour les étiquettes

        # Boucle à travers chaque classe
        for class_label in range(num_classes):
            # Indices des échantillons appartenant à la classe actuelle
            indices = np.where(y_train == class_label)[0]

            # Sous-échantillonnage aléatoire pour sélectionner 'new_samples_per_class' échantillons
            sampled_indices = resample(
                indices, n_samples=new_samples_per_class, replace=False, random_state=42
            )

            # Ajout des échantillons sous-échantillonnés et de leurs étiquettes aux DataFrames
            new_X_train = pd.concat([new_X_train, X_train.loc[sampled_indices]])
            new_y_train = pd.concat([new_y_train, y_train.loc[sampled_indices]])

        # Réinitialiser les index des DataFrames
        new_X_train = new_X_train.reset_index(drop=True)
        new_y_train = new_y_train.reset_index(drop=True)
        new_y_train = new_y_train['prdtypecode'].values.astype(int)
        # new_y_train = new_y_train.values.reshape(1350).astype("int")

        # Charger les modèles préalablement sauvegardés
        tokenizer = self.tokenizer
        lstm_model = self.lstm
        vgg16_model = self.vgg16

        train_sequences = tokenizer.texts_to_sequences(new_X_train["description"])
        train_padded_sequences = pad_sequences(
            train_sequences, maxlen=10, padding="post", truncating="post"
        )

        # Paramètres pour le prétraitement des images
        target_size = (
            224,
            224,
            3,
        )  # Taille cible pour le modèle VGG16, ajustez selon vos besoins

        images_train = new_X_train["image_path"].apply(
            lambda x: self.preprocess_image(x, target_size)
        )

        images_train = tf.convert_to_tensor(images_train.tolist(), dtype=tf.float32)

        lstm_proba = lstm_model.predict([train_padded_sequences])

        vgg16_proba = vgg16_model.predict([images_train])

        return lstm_proba, vgg16_proba, new_y_train

    def optimize(self, lstm_proba, vgg16_proba, y_train):
        # Recherche des poids optimaux en utilisant la validation croisée
        best_weights = None
        best_accuracy = 0.0

        for lstm_weight in np.linspace(0, 1, 101):  # Essayer différents poids pour LSTM
            vgg16_weight = 1.0 - lstm_weight  # Le poids total doit être égal à 1

            combined_predictions = (lstm_weight * lstm_proba) + (
                vgg16_weight * vgg16_proba
            )
            final_predictions = np.argmax(combined_predictions, axis=1)
            accuracy = accuracy_score(y_train, final_predictions)

            if accuracy > best_accuracy:# Training loop with tracking metrics
                best_accuracy = accuracy
                best_weights = (lstm_weight, vgg16_weight)

        return best_weights




class FusionModel:
    def __init__(self, tokenizer, experiment_name=None):
        self.tokenizer = tokenizer
        self.model = None
        self.experiment_name = experiment_name



    def preprocess_and_fit(self, X_train, y_train, X_val, y_val, epochs=1):
        # if self.experiment_name: 
        #     mlflow.set_experiment(experiment_name=self.experiment_name) 
        
        df_train = pd.concat([X_train, y_train], axis=1)
        df_val = pd.concat([X_val, y_val], axis=1)


        ds_train = df_to_dataset(df_train, target_col = "prdtypecode", features=['description', 'image_path'], batch_size = 32, tokenizer = self.tokenizer, max_sequence_length = 10, shuffle=True, seed=0, img_size=(250,250))
        ds_val = df_to_dataset(df_val, target_col = "prdtypecode", features=['description', 'image_path'], batch_size = 32, tokenizer = self.tokenizer, max_sequence_length = 10, shuffle=True, seed=0, img_size=(250,250))
    
        
        text_input = Input(shape=(10,), name='text_input')
        text_model = Embedding(input_dim=10000, output_dim=128)(text_input)
        text_model = LSTM(64, return_sequences=True)(text_model)
        text_model = Dropout(0.5)(text_model)
        text_model = LSTM(64, return_sequences=False)(text_model)
        text_model = Dropout(0.5)(text_model)
        text_model = Dense(128, activation='relu')(text_model)
        text_model = BatchNormalization()(text_model)
        

        img_input = Input(shape=(250,250,3), name='img_input')
        img_model = Conv2D(64, (3,3),1, activation='relu',kernel_regularizer=regularizers.l1(0.001))(img_input)
        img_model = MaxPooling2D()(img_model)
        img_model = Conv2D(32, (3,3), 1, activation='relu',kernel_regularizer=regularizers.l1(0.001))(img_model)
        img_model = MaxPooling2D()(img_model)
        img_model = Conv2D(16, (3,3), 1, activation='relu',kernel_regularizer=regularizers.l1(0.001))(img_model)
        img_model = MaxPooling2D()(img_model)
        img_model = Dropout(rate = .4)(img_model)
        img_model = Flatten()(img_model)
        img_model = Dense(64, activation='relu')(img_model)
        img_model = BatchNormalization()(img_model)

        concatenated = tf.keras.layers.Concatenate(axis=1)([text_model, img_model])

        dense = Dense(128, activation = 'relu')(concatenated)
        dense = Dropout(.5)(dense)
        dense = Dense(64, activation = 'relu')(dense)
        dense = Dropout(.5)(dense)
        out = tf.keras.layers.Dense(27, activation='softmax')(dense)

        self.model = Model([text_input, img_input], out)



        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        fusion_callbacks = [
            ModelCheckpoint(
                filepath="models/best_fusion_model.keras", save_best_only=True
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
        ]
        self.model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=1,
            callbacks=fusion_callbacks
        )
        # # Training loop with tracking metrics
        # for epoch in range(epochs):
        #     history = self.model.fit(ds_train, validation_data=ds_val, epochs=1, callbacks=fusion_callbacks)
        #     
        #     train_loss = history.history['loss'][0]
        #     train_accuracy = history.history['accuracy'][0]
        #     validation_loss = history.history['val_loss'][0]
        #     validation_accuracy = history.history['val_accuracy'][0]
        #     
        #     # Log metrics with MLflow
        #     mlflow.log_metric("train_loss", train_loss)
        #     mlflow.log_metric("train_accuracy", train_accuracy)
        #     mlflow.log_metric("validation_loss", validation_loss)
        #     mlflow.log_metric("validation_accuracy", validation_accuracy)
