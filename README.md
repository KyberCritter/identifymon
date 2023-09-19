# identifymon by Scott Ratchford
Using a conventional neural network to classify images Pokemon and Digimon. 

## Requirements
- Python 3.11.5+
- NumPy 1.24.3+
- scikit-learn 1.3.0+
- tensorflow 2.13.0+
- pandas 2.1.0+

## Program Setup
1. Clone this repository.
2. Open the terminal and navigate to the directory where the repository is located.
3. Create a folder called `data` in the root of the repository.
4. Download the [Pokemon vs. Digimon dataset](https://www.kaggle.com/datasets/vsvale/pokemon-vs-digimon-image-dataset) from Kaggle. Unzip the data and copy the folder `automlpoke` to the `data` folder you just created.

## Training the Model
This process is not required, since the trained model can be found as `models\pokemon_vs_digimon_CNN_latest.h5`. Training the model will create (or overwrite) the file `results\pokemon_vs_digimon_CNN_latest.h5`, as well as create a CSV file containing information about the training process.
1. Run the following command from the root of your repository: `python main.py train`

## Classify a Single Image
1. Run the following command from the root of your repository: `python main.py classify`
2. Input the relative or absolute path of the image file you want to classify.

## Classify a Group of Pokemon or Digimon
This process will output a file called `bulk_classification_results_X.csv` where X is the smallest positive integer that does not cause a naming conflict.
1. Create a folder exclusively containing images of Pokemon or a folder exclusively containing images of Digimon.
2. Run the following command from the root of your repository: `python main.py classify-bulk`
3. Input the relative or absolute path of the folder of images you want to classify.
