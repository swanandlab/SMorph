# SMorph

This library automates the morphological analysis of astrocytes and classify different subgroups based on the extracted morphometric parameters.

The notebook single_cell_analysis.ipynb includes the visual analysis to explore various morphological parameters of a single cell. Notebook group_analysis.ipynb includes the group level analysis of astrocytes using Principal Component Analysis (PCA) which helps to distinguish the differences between different classes of astrocytes based on their morphological parameters.


## Installation

The code has been tested for python 3.7, if you don't have it installed, please download the binaries from [Python releases](https://www.python.org/downloads/release/python-370/) and follow the installation guide.

Install the dependencies by typing following in the command line:

```
pip install -r requirements.txt
```

If you have a GitHub account, you can clone (or fork) the repository by running:

```
git clone https://github.com/swanandlab/SMorph.git
```

If you are not familiar with git or don’t have a GitHub account, you can download the repository as a zip file by going to the GitHub repository (https://github.com/swanandlab/SMorph/) in the browser and click the “Download” button on the upper right corner.

This will store the repository on your local computer and you can then place your own data folders inside the Morphometric analysis folder and implement analysis by replacing the input data folder names.

To run the notebooks, execute following from command line and locate to the desired notebook using browser.

```
jupyter notebook
```

If the above commands throw errors on windows, please try:

```
python -m pip install -r requirements.txt
python -m notebook
```
