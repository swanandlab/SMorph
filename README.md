# SMorph

This library automates the morphological analysis of astrocytes and classify
different subgroups based on the extracted morphometric parameters.

The notebook [single_cell_analysis.ipynb](./single_cell_analysis.ipynb)
includes the visual analysis to explore various morphological parameters of a
single cell. And the notebook [group_analysis.ipynb](./group_analysis.ipynb)
includes the group level analysis of cells of the Nervous System using
Principal Component Analysis (PCA) which helps to distinguish the differences
between different classes of cells based on their morphological parameters.

## Quickstart

The package can be easily used in local and Colaboratory environment.

### Colaboratory

Just go to
[single_cell_analysis.ipynb](https://colab.research.google.com/github/parulsethi/SMorph/blob/master/single_cell_analysis.ipynb)
for Single Cell analysis, and
[group_analysis.ipynb](https://colab.research.google.com/github/parulsethi/SMorph/blob/master/group_analysis.ipynb)
for Group Cells analysis. You'll have to upload you data to Colab Colaboratory
environment either directly to session storage or to your Google Drive account.

Instructions for linking your *Google Drive* dataset:

- Either select `Mount Drive` from sidebar `Files` browser or manually execute
the following code in a cell.

```python
from google.colab import drive
drive.mount('/content/drive')
```

- After completing the authorization, paste the authorization code in the input.

- After confirmation, refresh the Files sidebar. Your drive would now be visible
in it.

- Follow the same instructions in the following Usage section for
organization of your data.

The same notebook can also run on your local environment.

## Installation from source

The code has been tested for Python 3.6.1 and above, if you don't have it
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
jupyter notebook
```

If the above command throw errors on windows, please try:

```sh
python -m notebook
```

## Usage

### Group analysis

Place your image data folders inside the `Datasets` folder, with each group's
images organized in their respective folders.

Run the analysis by replacing the value of `groups_folders` variable in
`group_analysis.ipynb` with path to each of your cell groups.

### Single cell analysis

Run the analysis by replacing the value of `cell_image` variable in
`single_cell_analysis.ipynb` with path to your cell image.
