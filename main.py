# Scott Ratchford, 2023.09.19

import numpy as np
import pandas as pd
import os
import cv2
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

KAGGLE_MODE = False

if KAGGLE_MODE:
    POKEMON_FOLDER = "/kaggle/input/pokemon-vs-digimon-image-dataset/automlpoke/pokemon/"
    DIGIMON_FOLDER = "/kaggle/input/pokemon-vs-digimon-image-dataset/automlpoke/digimon/"
    OTHER_FOLDER = "/kaggle/input/random-images/dataset/train/"
else:
    POKEMON_FOLDER = "./data/automlpoke/pokemon/"
    DIGIMON_FOLDER = "./data/automlpoke/digimon/"
    OTHER_FOLDER = "./data/random-images/train/"

def load_and_preprocess_images(folder_path, label):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            images.append(img)
            labels.append(label)
            filenames.append(filename)
    return images, labels, filenames

def prepare_data(pokemon_folder, digimon_folder, other_folder):
    pokemon_images, pokemon_labels, pokemon_filenames = load_and_preprocess_images(pokemon_folder, 0)
    digimon_images, digimon_labels, digimon_filenames = load_and_preprocess_images(digimon_folder, 1)
    other_images, other_labels, other_filenames = load_and_preprocess_images(other_folder, 2)
    
    images = pokemon_images + digimon_images + other_images
    labels = pokemon_labels + digimon_labels + other_labels
    filenames = pokemon_filenames + digimon_filenames + other_filenames
    
    X = np.array(images)
    y = np.array(labels)
    
    return X, y, filenames

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model():
    # Prepare data
    X, y, filenames = prepare_data(POKEMON_FOLDER, DIGIMON_FOLDER, OTHER_FOLDER)
    
    # Split data
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(X, y, filenames, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val, filenames_train, filenames_val = train_test_split(X_train, y_train, filenames_train, test_size=0.25, random_state=42)
    
    # Create a DataFrame to store file details
    df = pd.DataFrame({
        "Filename": filenames_train + filenames_val + filenames_test,
        "Label": y_train.tolist() + y_val.tolist() + y_test.tolist(),
        "Phase": ["Train"] * len(filenames_train) + ["Validation"] * len(filenames_val) + ["Test"] * len(filenames_test)
    })
    
    # Save the DataFrame to a CSV file. If the file already exists, adjust the filename.
    if KAGGLE_MODE:
        df.to_csv("/kaggle/working/training_file_details.csv", index=False)
    else:
        i = 1
        while os.path.exists(f"training_file_details_{i}.csv"):
            i += 1
        df.to_csv(f"training_file_details_{i}.csv", index=False)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=3)
    y_val = to_categorical(y_val, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)
    
    # Create and compile model
    model = create_model()
    
    # Train model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}")
    
    # Save model to a file
    model.save("./results/pokemon_vs_digimon_CNN_latest.h5")

def load_and_test_model():
    # Load the model based on KAGGLE_MODE
    try:
        if KAGGLE_MODE:
            loaded_model = load_model("/kaggle/working/pokemon_vs_digimon_CNN_latest.h5")
        else:
            loaded_model = load_model("./models/pokemon_vs_digimon_CNN_latest.h5")
    except:
        try:
            if KAGGLE_MODE:
                print("Could not find /kaggle/working/pokemon_vs_digimon_CNN_latest.h5")
            else:
                print("Could not find ./models/pokemon_vs_digimon_CNN_latest.h5")
            model_path = input("Enter the path to the .h5 model file: ")
            loaded_model = load_model(model_path)
        except:
            return
    
    # Load a test image
    test_image_path = input("Enter the path to the test image: ")
    test_image = cv2.imread(test_image_path)
    
    if test_image is None:
        print("Could not load the image. Please check the file path.")
        return
    
    # Preprocess the image
    test_image = cv2.resize(test_image, (128, 128))
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    
    # Make prediction
    prediction = loaded_model.predict(test_image)
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label] * 100  # Convert to percentage
    
    # Output the result
    if predicted_label == 0:
        print(f"The model predicts this image is a Pokemon with {confidence:.2f}% confidence.")
    elif predicted_label == 1:
        print(f"The model predicts this image is a Digimon with {confidence:.2f}% confidence.")
    else:
        print(f"The model predicts this image is 'Other' with {confidence:.2f}% confidence.")


def bulk_classify():
    # Enter the folder containing the test images (all images in the folder must have the same label)
    test_folder = input("Enter the path to the folder containing the test images: ")
    # Enter the label of the test images (0 for Pokemon, 1 for Digimon)
    test_label = int(input("Enter the label of the test images (0 for Pokemon, 1 for Digimon): "))
    if test_label != 0 and test_label != 1:
        print("Invalid label. Exiting.")
        return
    
    # Load the model based on KAGGLE_MODE
    try:
        if KAGGLE_MODE:
            loaded_model = load_model("/kaggle/working/pokemon_vs_digimon_CNN_latest.h5")
        else:
            loaded_model = load_model("./models/pokemon_vs_digimon_CNN_latest.h5")
    except:
        try:
            if KAGGLE_MODE:
                print("Could not find /kaggle/working/pokemon_vs_digimon_CNN_latest.h5")
            else:
                print("Could not find ./models/pokemon_vs_digimon_CNN_latest.h5")
            model_path = input("Enter the path to the .h5 model file: ")
            loaded_model = load_model(model_path)
        except:
            return
        
    # Load and preprocess every image in the test folder
    test_images = []
    test_filenames = []
    for filename in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            test_images.append(img)
            test_filenames.append(filename)
    
    # Convert the list of images to a NumPy array
    test_images = np.array(test_images)

    # Make predictions
    predictions = loaded_model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1) * 100  # Convert to percentage
    
    # Create a DataFrame to store file details
    df = pd.DataFrame({
        "Filename": test_filenames,
        "Label": [test_label] * len(test_filenames),
        "Predicted Label": predicted_labels,
        "Confidence": confidences
    })
    
    # Save the DataFrame to a CSV file. If the file already exists, adjust the filename.
    if KAGGLE_MODE:
        df.to_csv("/kaggle/working/bulk_classification_results.csv", index=False)
    else:
        i = 1
        while os.path.exists(f"bulk_classification_results_{i}.csv"):
            i += 1
        df.to_csv(f"./results/bulk_classification_results_{i}.csv", index=False)

    # Output a summary of the results
    print(f"Total images: {len(test_filenames)}")
    print(f"Correct predictions: {len(df[df['Label'] == df['Predicted Label']])}")
    print(f"Accuracy: {len(df[df['Label'] == df['Predicted Label']]) / len(test_filenames) * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        choice = input("Do you want to train a new model, classify a single image, or classify a group of images with one label? (train/classify/classify-bulk): ")
    else:
        choice = sys.argv[1]
    
    if choice.lower() == "train":
        train_model()
    elif choice.lower() == "classify":
        load_and_test_model()
    elif choice.lower() == "classify-bulk":
        bulk_classify()
    else:
        print("Invalid choice. Exiting.")
