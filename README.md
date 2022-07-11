# License Plate Reader
OCR implementation using Convolutional Neural Networks for Italian license plate recognition.

## Plate generation
For generating plates follow the driver instructions executing `py LicensePlateGenerator/plateGenerator.py`.
Generated images will be saved into `LicensePlateGenerator/output/` folder.

Possible plate types to generate:
<div align="center" width="100%" style="display:flex; justify-content:center">
    <div style="margin-right:auto; margin-left:auto;">
        <img src="LicensePlateGenerator/assets/full-plate-auto.png?raw=true" alt="Auto">
        <figcaption>Auto</figcaption>
    </div>
    <div style="margin-right:auto; margin-left:auto;">
        <img src="LicensePlateGenerator/assets/full-plate-aeronautica-mil.png?raw=true" alt="Aeronautica Militare">
        <figcaption>Aeronautica Militare</figcaption>
    </div>
    <div style="margin-right:auto; margin-left:auto;">
        <img src="LicensePlateGenerator/assets/full-plate-carabinieri.png?raw=true" alt="Carabinieri">
        <figcaption>Carabinieri</figcaption>
    </div>
</div>

<div align="center" width="100%" style="display:flex; justify-content:center">
    <div style="margin-right:auto; margin-left:auto;">
        <img src="LicensePlateGenerator/assets/full-plate-esercito.png?raw=true" alt="Esercito Italiano">
        <figcaption>Esercito Italiano</figcaption>
    </div>
    <div style="margin-right:auto; margin-left:auto;">
        <img src="LicensePlateGenerator/assets/full-plate-marina-mil.png?raw=true" alt="Marina Militare">
        <figcaption>Marina Militare</figcaption>
    </div>
    <div style="margin-right:auto; margin-left:auto;">
        <img src="LicensePlateGenerator/assets/full-plate-vigili-fuoco.png?raw=true" alt="Vigili del fuoco">
        <figcaption>Vigili del fuoco</figcaption>
    </div>
</div>

<div align="center" width="100%" style="display:flex; justify-content:center">
    <div style="margin-right:auto; margin-left:auto;">
        <img src="LicensePlateGenerator/assets/full-plate-moto.png?raw=true" alt="Motorbike">
        <figcaption>Motorbike</figcaption>
    </div>
    <div style="margin-right:auto; margin-left:auto;">
        <img src="LicensePlateGenerator/assets/full-plate-special-auto.png?raw=true" alt="Auto speciale">
        <figcaption>Auto speciale</figcaption>
    </div>
</div>

## Usage
Once plates have been generated, create the dataset following the driver instructions executing
`py OCR/datasetGenerator.py`.
Generated datasets will be saved into `OCR/` folder.

Once datasets have been generated, use the NN following the driver instructions executing
`py OCR/driver.py`. NN outputs will be saved into `OCR/` folder.
