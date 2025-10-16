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
from datetime import datetime

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

# --- DIVISÃO EM TREINO E TESTE (DADOS ORIGINAIS) ---
X_data, y_data = shuffle(X_data, y_data, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=101)

print("\n--- DISTRIBUIÇÃO DAS CLASSES (SEM BALANCEAMENTO) ---")
print("Formato dos dados de treino: ", X_train.shape)
print("Contagem de classes de TREINO:\n", pd.Series(y_train).value_counts())
print("\nFormato dos dados de teste: ", X_test.shape)
print("Contagem de classes de TESTE:\n", pd.Series(y_test).value_counts())

# --- PREPARAÇÃO DOS LABELS (ONE-HOT ENCODING) ---
y_train_new = [labels.index(i) for i in y_train]
y_train = tf.keras.utils.to_categorical(y_train_new)

y_test_new = [labels.index(i) for i in y_test]
y_test = tf.keras.utils.to_categorical(y_test_new)

# --- CONSTRUÇÃO DO MODELO (TRANSFER LEARNING BÁSICO) ---
base_model = ResNet101V2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
base_model.trainable = True

inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(rate=0.5)(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

# --- TREINAMENTO DO MODELO (FASE ÚNICA) ---
print("\n--- INICIANDO TREINAMENTO BÁSICO ---")
checkpoint = ModelCheckpoint("resnet_basic.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)

history = model.fit(X_train, y_train, validation_split=0.1, epochs=15, verbose=1, batch_size=32,
                      callbacks=[checkpoint, reduce_lr])

# --- AVALIAÇÃO E SALVAMENTO DOS RESULTADOS ---
print("\nCarregando o melhor modelo salvo para avaliação...")
model.load_weights("resnet_basic.h5")

pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Cria uma pasta para os resultados se ela não existir
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Gera um nome base com timestamp para os arquivos
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_filename = f"resnet_basic_{timestamp}"

# Define os nomes completos dos arquivos de saída
report_filename = os.path.join(output_dir, f"relatorio_{base_filename}.txt")
matrix_filename = os.path.join(output_dir, f"matriz_confusao_{base_filename}.png")

# --- 1. SALVANDO O RELATÓRIO DE CLASSIFICAÇÃO ---
print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
report = classification_report(y_test_classes, pred_classes, target_names=labels)
print(report)

with open(report_filename, 'w') as f:
    f.write(f"Relatório de Classificação para o modelo: {base_filename}\n\n")
    f.write(report)
print(f"Relatório de classificação salvo em: {report_filename}")

# --- 2. SALVANDO A MATRIZ DE CONFUSÃO ---
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sns.heatmap(confusion_matrix(y_test_classes, pred_classes), ax=ax, xticklabels=labels, yticklabels=labels, annot=True,
            fmt='d', cmap='viridis', alpha=0.7, linewidths=2, linecolor='black')
fig.text(s='Heatmap da Matriz de Confusão (Modelo Básico)', size=18, fontweight='bold',
         fontname='monospace', color=colors_dark[1], y=0.92, x=0.28, alpha=0.8)

plt.xlabel("Previsão", fontsize=14)
plt.ylabel("Verdadeiro", fontsize=14)

plt.savefig(matrix_filename, dpi=300, bbox_inches='tight')
print(f"Matriz de confusão salva em: {matrix_filename}")

plt.close(fig)