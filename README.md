# License Plate Reader
OCR implementation using Convolutional Neural Networks for Italian license plate recognition.

## Identifiable plates
Possible plate types to generate and recognise:
<table align="center">
    <tr>
        <td>
            <img src="LicensePlateGenerator/assets/full-plate-auto.png?raw=true" alt="Auto"><br>
            <figcaption>Auto</figcaption>
        </td><td>
            <img src="LicensePlateGenerator/assets/full-plate-aeronautica-mil.png?raw=true" alt="Aeronautica Militare"><br>
            <figcaption>Aeronautica Militare</figcaption>
        </td><td>
            <img src="LicensePlateGenerator/assets/full-plate-carabinieri.png?raw=true" alt="Carabinieri"><br>
            <figcaption>Carabinieri</figcaption>
        </td>
    </tr><tr>
        <td>
            <img src="LicensePlateGenerator/assets/full-plate-esercito.png?raw=true" alt="Esercito Italiano"><br>
            <figcaption>Esercito Italiano</figcaption>
        </td><td>
            <img src="LicensePlateGenerator/assets/full-plate-marina-mil.png?raw=true" alt="Marina Militare"><br>
            <figcaption>Marina Militare</figcaption>
        </td><td>
        <img src="LicensePlateGenerator/assets/full-plate-vigili-fuoco.png?raw=true" alt="Vigili del fuoco"><br>
        <figcaption>Vigili del fuoco</figcaption>
        </td>
    </tr><tr>
        <td>
            <img src="LicensePlateGenerator/assets/full-plate-moto.png?raw=true" alt="Motorbike"><br>
            <figcaption>Motorbike</figcaption>
        </td><td>
            <img src="LicensePlateGenerator/assets/full-plate-special-auto.png?raw=true" alt="Auto speciale"><br>
            <figcaption>Auto speciale</figcaption>
        </td>
    </tr>
</table>

## Installation
Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/ .

Also, install all requirements via PIP using `pip install -r requirements.txt`.

For a better experience, is recommended to install CUDA and cuDNN packages.

## Usage
- Generate plates following the driver instructions executing `py plateGenerator.py` inside `LicensePlateGenerator/` folder.
Generated images will be saved into `LicensePlateGenerator/output/` folder.

- Once plates have been generated, create the dataset following the driver instructions executing
`py datasetGenerator.py` inside `OCR/` folder.
Generated datasets will be saved into `OCR/` folder.

- Once datasets have been generated, use the OCR CNN following the driver instructions executing
`py driver.py` inside `OCR/` folder, which allows to train and test the CNN for OCR functionality.
NN outputs will be saved into `OCR/` folder.

- Once OCR CNN has been trained, use the License Plate Detector NN following the driver instructions executing `py detect.py` inside `PlateDetector/` folder, which allows to train and test the Plate Detector NN.
    + To train the NN, train and test images must be placed into `PlateDetector/Tensorflow/workspace/images/` folder, inside `train/` and `test/` folder respectively.
    + Trained model will be saved into `PlateDetector/Tensorflow/workspace/models/my_ssd_mobnet/` folder, and will be saved using multiple checkpoints. When restoring the model state, be sure to load the latest checkpoint.

- Once both NN have been trained, use `py driver.py` in the root folder to use both Detector ad OCR NN, following the driver instructions.
    + Every input (images, videos) to the final program must be placed into `data/` folder (if not exists it will be created from the driver).
    + Every output of this program will be stored into `output/` folder.

## Credits
File `PlateDetector/detect.py` contains code readapted from https://github.com/nicknochnack/TFODCourse/blob/main/2.%20Training%20and%20Detection.ipynb .
