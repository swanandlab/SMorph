from os import getcwd, listdir, mkdir, path, remove
from pickle import dump

import skimage.io as io


def read_groups_folders(groups_folders):
    """Synchronously read list of folders for images.

    Parameters
    ----------
    groups_folders : list
        A list of strings containing path of each folder with image dataset.

    Returns
    -------
    file_names : list
        A Python list of names of cell image files.
    dataset : list
        A Python list of ndarrays containing cell image data.

    """
    file_names, dataset = [], []

    for group in groups_folders:
        group_data = []
        for file in listdir(group):
            if not file.startswith('.'):  # skip hidden files
                name = group + '/' + file
                file_names.append(name)
                image = io.imread(name)
                group_data.append(image)
        dataset.append(group_data)

    return file_names, dataset


def df_to_csv(df, folder, out_file_name):
    """Export DataFrame to a CSV file.

    Parameters
    ----------
    df : DataFrame
        Data to export into file.
    folder : str
        Name of the folder in which the exported file will reside.
    out_file_name : str
        Name of the file to export with extension.

    """
    DIR = getcwd() + folder
    FILE = DIR + out_file_name
    if not (path.exists(DIR) and path.isdir(DIR)):
        mkdir(DIR)
    if path.exists(FILE):
        remove(FILE)

    df.to_csv(DIR + out_file_name, index=False, mode='w')


def dict_to_pickle(data, folder, out_file_name):
    """Export a Python dictionary to a pickle file.

    Parameters
    ----------
    data : dict
        Data to export into file.
    folder : str
        Name of the folder in which the exported file will reside.
    out_file_name : str
        Name of the file to export with extension.

    """
    DIR = getcwd() + folder
    FILE = DIR + out_file_name
    if not (path.exists(DIR) and path.isdir(DIR)):
        mkdir(DIR)
    if path.exists(FILE):
        remove(FILE)

    with open(DIR + out_file_name, 'wb') as file:
        dump(data, file, -1)
