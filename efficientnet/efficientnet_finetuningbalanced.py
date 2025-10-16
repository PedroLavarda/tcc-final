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
# --- MODIFICAÇÃO 1: Importar o EfficientNetB5 ---
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import balance_dataset

# --- CONFIGURAÇÃO INICIAL E DOWNLOAD DO DATASET ---
path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")
print("Caminho para os arquivos do dataset:", path)

colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
image_size = 150

# --- CARREGAMENTO DAS IMAGENS ---
X_data = []
y_data = []

print("Carregando imagens de treino...")
for i in labels:
    folderPath = os.path.join(path, 'Training', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        if img is not None:
            img = cv2.resize(img, (image_size, image_size))
            X_data.append(img)
            y_data.append(i)

print("Carregando imagens de teste...")
for i in labels:
    folderPath = os.path.join(path, 'Testing', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        if img is not None:
            img = cv2.resize(img, (image_size, image_size))
            X_data.append(img)
            y_data.append(i)

X_data = np.array(X_data)
y_data = np.array(y_data)

# --- DIVISÃO EM TREINO E TESTE ---
X_data, y_data = shuffle(X_data, y_data, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=101)

print("\n--- DISTRIBUIÇÃO INICIAL DAS CLASSES ---")
print("Contagem de classes de TREINO (antes do balanceamento):\n", pd.Series(y_train).value_counts())
print("\nContagem de classes de TESTE (antes do balanceamento):\n", pd.Series(y_test).value_counts())

# --- CAMADA DE DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")

# --- ETAPA 1: BALANCEAMENTO DOS CONJUNTOS DE TREINO E TESTE ---
print("\n--- INICIANDO BALANCEAMENTO DO CONJUNTO DE TREINO ---")
X_train, y_train = balance_dataset(X_train, y_train, data_augmentation)

print("\n--- INICIANDO BALANCEAMENTO DO CONJUNTO DE TESTE ---")
X_test, y_test = balance_dataset(X_test, y_test, data_augmentation)

print("\n--- DISTRIBUIÇÃO DAS CLASSES APÓS BALANCEAMENTO ---")
print("Formato dos dados de treino: ", X_train.shape)
print("Contagem de classes de TREINO:\n", pd.Series(y_train).value_counts())
print("\nFormato dos dados de teste: ", X_test.shape)
print("Contagem de classes de TESTE:\n", pd.Series(y_test).value_counts())

# --- PREPARAÇÃO DOS LABELS (ONE-HOT ENCODING) ---
y_train_new = [labels.index(i) for i in y_train]
y_train = tf.keras.utils.to_categorical(y_train_new)

y_test_new = [labels.index(i) for i in y_test]
y_test = tf.keras.utils.to_categorical(y_test_new)

# --- CONSTRUÇÃO DO MODELO (TRANSFER LEARNING COM EFFICIENTNETB5) ---
# --- MODIFICAÇÃO 2: Usar o EfficientNetB5 como modelo base ---
base_model = EfficientNetB5(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
base_model.trainable = False 

inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
# --- MODIFICAÇÃO 3: Usar a função de pré-processamento do EfficientNet ---
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(rate=0.5)(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# --- FASE 1: TREINAMENTO DA CAMADA DE CLASSIFICAÇÃO ---
print("\n--- INICIANDO FASE 1: TREINAMENTO DA CABEÇA ---")
# --- MODIFICAÇÃO 4: Atualizar nome do arquivo de checkpoint ---
checkpoint = ModelCheckpoint("efficientnetb5_finetuningbalanced.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)

history = model.fit(X_train, y_train, validation_split=0.1, epochs=5, verbose=1, batch_size=32,
                    callbacks=[checkpoint, reduce_lr])

# --- FASE 2: AJUSTE FINO (FINE-TUNING) ---
print("\n--- INICIANDO FASE 2: AJUSTE FINO DO MODELO COMPLETO ---")
base_model.trainable = True
model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              metrics=['accuracy'])
model.summary() 

history_fine_tune = model.fit(X_train, y_train, validation_split=0.1, epochs=10, verbose=1, batch_size=32,
                              initial_epoch=history.epoch[-1] + 1,
                              callbacks=[checkpoint, reduce_lr])

# --- AVALIAÇÃO DO MODELO ---
print("\nCarregando o melhor modelo salvo para avaliação...")
# --- MODIFICAÇÃO 5: Carregar os pesos do modelo correto ---
model.load_weights("efficientnetb5_finetuningbalanced.h5")

pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
print(classification_report(y_test_classes, pred_classes, target_names=labels))

# --- MATRIZ DE CONFUSÃO ---
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sns.heatmap(confusion_matrix(y_test_classes, pred_classes), ax=ax, xticklabels=labels, yticklabels=labels, annot=True,
            fmt='d', cmap='viridis', alpha=0.7, linewidths=2, linecolor='black')
fig.text(s='Heatmap da Matriz de Confusão', size=18, fontweight='bold',
         fontname='monospace', color=colors_dark[1], y=0.92, x=0.28, alpha=0.8)

plt.xlabel("Previsão", fontsize=14)
plt.ylabel("Verdadeiro", fontsize=14)

# Para salvar a imagem da matriz de confusão, descomente a linha abaixo
# plt.savefig("matriz_confusao_efficientnetb5.png")

plt.close(fig)