# Morphometric-analysis

This library automates the morphological analysis of astrocytes and classify different subgroups based on the extracted morphometric parameters.

The notebook single_cell_analysis.ipynb includes the visual analysis to explore various morphological parameters of a single cell. Notebook group_analysis.ipynb includes the group level analysis of astrocytes using Principal Component Analysis (PCA) which helps to distinguish the differences between different classes of astrocytes based on their morphological parameters.

## Installation

The code has been tested for python 3.7, if you don't have it installed, please download the binaries from [Python releases](https://www.python.org/downloads/release/python-370/) and follow the installation guide.

Install the dependencies by typing following in the command line:

`
pip install -r requirements.txt
`

Download the Morphometric-analysis repository using following commands:

`
git clone morphometric-analysis
`

This will store the repository on your local computer and you can then place your own data folders inside the Morphometric analysis folder and implement analysis by replacing the input data folder names.

To run the notebooks, execute `jupyter notebook` from command line and locate to the desired notebook using browser.