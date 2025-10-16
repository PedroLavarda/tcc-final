import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle

def balance_dataset(X, y, augmentation_layer):
    df = pd.DataFrame({'image': list(X), 'label': list(y)})
    class_counts = df['label'].value_counts()
    max_count = class_counts.max()
    
    balanced_df = df.copy()
    
    for label in class_counts.index:
        count = class_counts[label]
        n_to_generate = max_count - count
        if n_to_generate > 0:
            print(f"Gerando {n_to_generate} imagens para a classe '{label}'...")
            class_df = df[df['label'] == label]
            class_images = np.array(class_df['image'].tolist())
            
            augmented_images = []
            for _ in tqdm(range(n_to_generate)):
                random_index = np.random.randint(0, len(class_images))
                image_to_augment = tf.expand_dims(class_images[random_index], 0)
                augmented_image = augmentation_layer(image_to_augment, training=True)
                augmented_images.append(np.squeeze(augmented_image.numpy().astype('uint8')))
            
            augmented_df = pd.DataFrame({'image': augmented_images, 'label': [label] * n_to_generate})
            balanced_df = pd.concat([balanced_df, augmented_df])
            
    X_balanced = np.array(balanced_df['image'].tolist())
    y_balanced = np.array(balanced_df['label'].tolist())
    
    return shuffle(X_balanced, y_balanced, random_state=101)