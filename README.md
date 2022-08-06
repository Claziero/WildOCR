# OCR reader
OCR implementation using Convolutional Neural Networks for text and Italian license plate recognition.

## Installation
Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/ .

Also, install all requirements via PIP using `pip install -r requirements.txt`.

For a better experience, is recommended to install CUDA and cuDNN packages.

## Usage
- Generate dataset for OCR NN training following the driver instructions executing `py characterGenerator.py` inside `CharacterGenerator/` folder.

- Once character images have been generated, create the dataset following the driver instructions executing
`py datasetGenerator.py` inside `OCR/` folder.
Generated datasets will be saved into `OCR/` folder.

- Once datasets have been generated, use the OCR CNN following the driver instructions executing
`py driver.py` inside `OCR/` folder, which allows to train and test the CNN for OCR functionality.
NN outputs will be saved into `OCR/` folder.

- Once OCR CNN has been trained, use the License Plate Detector NN following the driver instructions executing `py detect.py` inside `Detector/` folder, which allows to train and test the Plate Detector NN.
    + To train the NN, train and test images must be placed into `Detector/Tensorflow/workspace/images/` folder, inside `train/` and `test/` folder respectively.
    + Trained model will be saved into `Detector/Tensorflow/workspace/models/my_ssd_mobnet/` folder, and will be saved using multiple checkpoints. When restoring the model state, be sure to load the latest checkpoint.

- Once both NN have been trained, use `py driver.py` in the root folder to use both Detector ad OCR NN, following the driver instructions.
    + Every input (images, videos) to the final program must be placed into `data/input` folder (if not exists it will be created from the driver).
    + Every output of this program will be stored into `data/output/` folder.

## Credits
File `Detector/detect.py` contains code readapted from https://github.com/nicknochnack/TFODCourse/blob/main/2.%20Training%20and%20Detection.ipynb .
