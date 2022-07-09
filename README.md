# License Plate Reader
OCR implementation using Convolutional Neural Networks for Italian license plate recognition.

## Plate generation
For generating plates follow the driver instructions executing `py LicensePlateGenerator/lpg.py`.
Generated images will be saved into `LicensePlateGenerator/output/` folder.

## Usage
Once plates have been generated, create the dataset following the driver instructions executing
`py OCR/datasetGenerator.py`.
Generated datasets will be saved into `OCR/` folder.

Once datasets have been generated, use the NN following the driver instructions executing
`OCR/driver.py`. NN outputs will be saved into `OCR/` folder.
