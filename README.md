# License Plate Reader
OCR implementation using Convolutional Neural Networks for Italian license plate recognition.

## Plate generation
For generating plates follow the driver instructions executing `py LicensePlateGenerator/plateGenerator.py`.
Generated images will be saved into `LicensePlateGenerator/output/` folder.

Possible plate types to generate:
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

## Usage
Once plates have been generated, create the dataset following the driver instructions executing
`py OCR/datasetGenerator.py`.
Generated datasets will be saved into `OCR/` folder.

Once datasets have been generated, use the NN following the driver instructions executing
`py OCR/driver.py`. NN outputs will be saved into `OCR/` folder.
