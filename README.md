# SMorph

This library automates the morphological analysis of astrocytes and classify
different subgroups based on the extracted morphometric parameters.

The notebook `single_cell_analysis.ipynb` includes the visual analysis to
explore various morphological parameters of a single cell. And the notebook
`group_analysis.ipynb` includes the group level analysis of cells of the
Nervous System using Principal Component Analysis (PCA) which helps to
distinguish the differences between different classes of cells based on their
morphological parameters.

## Installation

The code has been tested for Python 3.7 and above, if you don't have it
installed, please download the binaries from
[Python releases](https://www.python.org/downloads/release/python-370) and
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

## Usage

### Group analysis

Place your image data folders inside the `Datasets` folder, with each group's
images organized in their respective folders.

Run the analysis by replacing the value of `groups_folders` variable in
`group_analysis.ipynb` with path to each of your cell groups.

### Single cell analysis

Run the analysis by replacing the value of `cell_image` variable in
`single_cell_analysis.ipynb` with path to your cell image.

To run the notebooks, execute following from command line and locate to the
desired notebook using browser.

```sh
jupyter notebook
```

If the above commands throw errors on windows, please try:

```sh
python -m notebook
```
