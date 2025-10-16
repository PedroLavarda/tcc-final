import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# Baixar o dataset do Kaggle
path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")
print("Path to dataset files:", path)

# Configurações iniciais
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']

X_data = []
y_data = []
image_size = 150

# Carregar imagens de treino
print("Carregando imagens de treino...")
for i in labels:
    folderPath = os.path.join(path, 'Training', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size, image_size))
        X_data.append(img)
        y_data.append(i)

# Carregar imagens de teste
print("Carregando imagens de teste...")
for i in labels:
    folderPath = os.path.join(path, 'Testing', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_data.append(img)
        y_data.append(i)

# Converter para arrays numpy
X_data = np.array(X_data)
y_data = np.array(y_data)

# Embaralhar e dividir os dados
X_data, y_data = shuffle(X_data, y_data, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=101)

print(f"Formato dos dados de treino: {X_train.shape}")
print(f"Contagem de classes nos dados de treino:\n{pd.Series(y_train).value_counts()}")

# Preparar os rótulos (labels) para o modelo
y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = tf.keras.utils.to_categorical(y_train_new)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = tf.keras.utils.to_categorical(y_test_new)

# Carregar o modelo base ResNet101V2 pré-treinado
base_model = ResNet101V2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
base_model.trainable = False 

# Construir o modelo final
inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
x = tf.keras.applications.resnet_v2.preprocess_input(inputs) # Normalização específica do ResNetV2
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(rate=0.5)(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Compilar o modelo
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

# Definir callbacks
checkpoint = ModelCheckpoint("resnet_finetuning.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)
callbacks_list = [checkpoint, reduce_lr]

# --- FASE 1: AQUECIMENTO DA "CABEÇA" DE CLASSIFICAÇÃO ---
print("\n--- INICIANDO FASE 1: TREINAMENTO DA CABEÇA ---")
history = model.fit(X_train, y_train, validation_split=0.1, epochs=5, verbose=1, batch_size=8,
                    callbacks=callbacks_list)

# --- FASE 2: AJUSTE FINO (FINE-TUNING) ---
print("\n--- INICIANDO FASE 2: AJUSTE FINO DO MODELO COMPLETO ---")
base_model.trainable = True

model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              metrics=['accuracy'])

model.summary() 

history_fine_tune = model.fit(X_train, y_train, validation_split=0.1, epochs=10, verbose=1, batch_size=8,
                              initial_epoch=history.epoch[-1] + 1,
                              callbacks=callbacks_list)

# --- AVALIAÇÃO DO MODELO ---
print("\nCarregando o melhor modelo salvo para avaliação...")
# O nome do arquivo deve corresponder ao salvo pelo ModelCheckpoint
model.load_weights("resnet_finetuning.h5")

pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
y_test_new = np.argmax(y_test, axis=1)

print(classification_report(y_test_new, pred, target_names=labels))

# Plotar a matriz de confusão
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sns.heatmap(confusion_matrix(y_test_new, pred), ax=ax, xticklabels=labels, yticklabels=labels, annot=True,
            fmt='d', cmap='viridis', alpha=0.7, linewidths=2)
fig.text(s='Heatmap of the Confusion Matrix', size=18, fontweight='bold',
         fontname='monospace', color=colors_dark[1], y=0.92, x=0.28, alpha=0.8)
plt.close(fig)