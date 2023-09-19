import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_and_preprocess_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            images.append(img)
            labels.append(label)
    return images, labels

def prepare_data(pokemon_folder, digimon_folder):
    pokemon_images, pokemon_labels = load_and_preprocess_images(pokemon_folder, 0)
    digimon_images, digimon_labels = load_and_preprocess_images(digimon_folder, 1)
    
    images = pokemon_images + digimon_images
    labels = pokemon_labels + digimon_labels
    
    X = np.array(images)
    y = np.array(labels)
    
    return X, y

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # Prepare data
    pokemon_folder = "./data/training-data/pokemon"
    digimon_folder = "./data/training-data/digimon"
    X, y = prepare_data(pokemon_folder, digimon_folder)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Create and compile model
    model = create_model()
    
    # Train model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
    
    # Save model
    model.save('pokemon_vs_digimon.h5')

if __name__ == '__main__':
    train_model()
