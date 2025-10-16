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
# --- MODIFICAÇÃO 1: Importar EfficientNetB5 ---
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
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

# --- DIVISÃO E BALANCEAMENTO INICIAL ---
X_data, y_data = shuffle(X_data, y_data, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=101)

print("\n--- DISTRIBUIÇÃO INICIAL DAS CLASSES ---")
print("Contagem de classes de TREINO (antes do balanceamento):\n", pd.Series(y_train).value_counts())
print("\nContagem de classes de TESTE (antes do balanceamento):\n", pd.Series(y_test).value_counts())

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")

print("\n--- INICIANDO BALANCEAMENTO DO CONJUNTO DE TREINO ---")
X_train, y_train = balance_dataset(X_train, y_train, data_augmentation)

print("\n--- INICIANDO BALANCEAMENTO DO CONJUNTO DE TESTE ---")
X_test, y_test = balance_dataset(X_test, y_test, data_augmentation)

print("\n--- DISTRIBUIÇÃO DAS CLASSES APÓS BALANCEAMENTO ---")
print("Formato dos dados de treino: ", X_train.shape)
print("Contagem de classes de TREINO:\n", pd.Series(y_train).value_counts())
print("\nFormato dos dados de teste: ", X_test.shape)
print("Contagem de classes de TESTE:\n", pd.Series(y_test).value_counts())

# --- AUMENTO DE DADOS ADICIONAL PARA TREINO ---
target_size_per_class = 2500
print(f"\n--- AUMENTANDO O DATASET DE TREINO PARA {target_size_per_class * len(labels)}+ IMAGENS ---")
df_train_balanced = pd.DataFrame({'image': list(X_train), 'label': list(y_train)})
final_train_df = df_train_balanced.copy()

for label in labels:
    class_df = df_train_balanced[df_train_balanced['label'] == label]
    current_count = len(class_df)
    n_to_generate = target_size_per_class - current_count
    
    if n_to_generate > 0:
        print(f"Gerando {n_to_generate} imagens adicionais para a classe '{label}'...")
        class_images = np.array(class_df['image'].tolist())
        
        augmented_images = []
        for _ in tqdm(range(n_to_generate)):
            random_index = np.random.randint(0, len(class_images))
            image_to_augment = tf.expand_dims(class_images[random_index], 0)
            augmented_image = data_augmentation(image_to_augment, training=True)
            augmented_images.append(np.squeeze(augmented_image.numpy().astype('uint8')))
        
        augmented_df = pd.DataFrame({'image': augmented_images, 'label': [label] * n_to_generate})
        final_train_df = pd.concat([final_train_df, augmented_df])

X_train = np.array(final_train_df['image'].tolist())
y_train = np.array(final_train_df['label'].tolist())
X_train, y_train = shuffle(X_train, y_train, random_state=101)

print("\n--- DATASET FINAL DE TREINO ---")
print(f"Formato final dos dados de treino: {X_train.shape}")
print(f"Contagem final de classes de treino:\n{pd.Series(y_train).value_counts()}")

# --- PREPARAÇÃO DOS LABELS (ONE-HOT ENCODING) ---
y_train_new = [labels.index(i) for i in y_train]
y_train = tf.keras.utils.to_categorical(y_train_new)

y_test_new = [labels.index(i) for i in y_test]
y_test = tf.keras.utils.to_categorical(y_test_new)

# --- CONSTRUÇÃO DO MODELO ---
# --- MODIFICAÇÃO 2: Usar EfficientNetB5 como modelo base ---
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
# --- MODIFICAÇÃO 4: Atualizar nome do arquivo do checkpoint ---
checkpoint = ModelCheckpoint("efficientnetb5_finetuning_data-aug_balanced.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
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

# --- AVALIAÇÃO E SALVAMENTO DOS RESULTADOS ---
print("\nCarregando o melhor modelo salvo para avaliação...")
# --- MODIFICAÇÃO 5: Carregar os pesos do modelo correto ---
model.load_weights("efficientnetb5_finetuning_data-aug_balanced.h5")

pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Cria uma pasta para os resultados se ela não existir
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Gera um nome base com timestamp para os arquivos
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# --- MODIFICAÇÃO 6: Atualizar nome base dos arquivos de saída ---
base_filename = f"efficientnetb5_finetuning_data-aug_balanced_{timestamp}"

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
fig.text(s='Heatmap da Matriz de Confusão (Fine-Tuning)', size=18, fontweight='bold',
         fontname='monospace', color=colors_dark[1], y=0.92, x=0.28, alpha=0.8)

plt.xlabel("Previsão", fontsize=14)
plt.ylabel("Verdadeiro", fontsize=14)

plt.savefig(matrix_filename, dpi=300, bbox_inches='tight')
print(f"Matriz de confusão salva em: {matrix_filename}")

plt.close(fig)