from errno import ENOENT
from os import getcwd, listdir, mkdir, path, remove

import skimage.io as io


def silent_remove_file(filename):
    try:
        remove(filename)
    except OSError as e:
        if e.errno != ENOENT:
            raise


def mkdir_if_not(name):
    """Collision-free mkdir"""
    if not (path.exists(name) and path.isdir(name)):
        mkdir(name)


def read_group_dirs(group_dirs):
    """Synchronously read list of folders for images.

    Parameters
    ----------
    group_dirs : list
        A list of strings containing path of each folder with image dataset.

    Returns
    -------
    file_names : list
        A Python list of names of cell image files.
    dataset : list
        A Python list of ndarrays containing cell image data.

    """
    file_names, dataset = [], []

    for group in group_dirs:
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
    DIR = path.join(getcwd(), folder)
    FILE = path.join(DIR, out_file_name)
    if not (path.exists(DIR) and path.isdir(DIR)):
        mkdir(DIR)
    if path.exists(FILE):
        remove(FILE)

    df.to_csv(FILE, index=False, mode='w')


def savefig(fig, name):
    fig.savefig(path.join(getcwd(), name), transparent=False, facecolor='w')
