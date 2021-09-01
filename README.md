# SMorph

This library automates the morphological analysis of cells and classify
different subgroups based on the extracted morphometric parameters.

- The notebook [single_cell_analysis.ipynb](./single_cell_analysis.ipynb)
includes the visual analysis to explore various morphological parameters of a
single cell.
- And the notebook [group_analysis.ipynb](./group_analysis.ipynb)
includes the group level analysis of cells of the Nervous System using
Principal Component Analysis (PCA) which helps to distinguish the differences
between different classes of cells based on their morphological parameters.

---

## Published Version

*Please note that this is an update to the initial published version. If you*
*prefer to use the published version please redirect to following commit*
*hyperlink.*

### [Published Version](https://github.com/swanandlab/SMorph/tree/bc8a1cc20d66eca755cb1f0621a4df72ca665bda)

- Follow the instructions there to setup and run the published version.

`Git Commit ID: bc8a1cc20d66eca755cb1f0621a4df72ca665bda`

*Please note that the published version is not packaged as a library.*

---

## Latest version usage

## Quickstart

The package can be easily used in local and Colaboratory environment.

### Colaboratory

Just go to following for:

- Single cell analysis: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swanandlab/SMorph/blob/main/single_cell_analysis.ipynb)

- Group Cells analysis: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swanandlab/SMorph/blob/main/group_analysis.ipynb)

You'll have to upload you data to *Colaboratory* environment either directly to
session storage or to your *Google Drive*.

Instructions for linking your *Google Drive* dataset:

- Either select `Mount Drive` from sidebar `Files` browser or manually execute
the following code in a cell.

```python
from google.colab import drive
drive.mount('/content/drive')
```

- After completing the authorization, paste the authorization code in the input.

- After confirmation, refresh the *Files* sidebar. Your drive would now be
visible in it.

- Follow the same instructions in the following ***Usage*** instructions for
organization of your data.

---

The same notebook can also run on your local environment.

## Installation from source

The code has been tested for Python 3.7.11 and above, if you don't have it
installed, please download the binaries from
[here](https://www.python.org/downloads/) and
follow the installation guide.

SMorph uses [Poetry](https://python-poetry.org) package manager.
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Poetry.
And install the dependencies by typing following in the command line:

```sh
pip install -r requirements.txt
poetry install
```

If the above commands throw errors on windows, please try:

```sh
python -m pip install -r requirements.txt
poetry install
```

## Setup for analysis on local environment

To run the notebooks, execute following from command line and locate to the
desired notebook using browser.

```sh
poetry shell
jupyter notebook
```

If the above command throw errors on windows, please try:

```sh
poetry shell
python -m notebook
```

---

## Usage

### Group analysis

- Place your image data folders inside the `Datasets` folder, with each group's
images organized in their respective folders.

- Run the analysis by replacing the value of `groups_folders` variable in
`group_analysis.ipynb` with path to each of your cell groups.

### Single cell analysis

- Run the analysis by replacing the value of `cell_image` variable in
`single_cell_analysis.ipynb` with path to your cell image.

---

## Help & Support

- If you encounter a previously unreported bug/code issue, please post here
(we encourage you to search issues first):
https://github.com/swanandlab/SMorph/issues

---

## References

If you use this code and find it useful, we kindly ask that you please cite
> Parul Sethi, Garima Virmani, Kushaan Gupta, Surya Chandra Rao
> Thumu, Narendrakumar Ramanan, Swananda Marathe.
> *Automated morphometric analysis with SMorph software reveals plasticity*
> *induced by antidepressant therapy in hippocampal astrocytes*. J Cell Sci 15 June 2021; 134 (12): jcs258430
> https://doi.org/10.1242/jcs.258430
