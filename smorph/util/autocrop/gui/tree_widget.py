import numpy as np

from magicclass.widgets import FreeWidget
from pandas import DataFrame
from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex
from PyQt5.QtWidgets import QTreeView

from .viewer import (
    _get_roi_scaled_points,
    _read_images,
)
from .._io import imread

def gen_dict_extract(key, var):
    # https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-dictionaries-and-lists
    if hasattr(var,'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result


def reconstruct_labels(bounds, images, out):
    for itr in range(len(images)):
        minz, miny, minx, maxz, maxy, maxx = bounds[itr]
        out[minz:maxz, miny:maxy, minx:maxx] += images[itr] * (itr + 1)
    return out


class ImageTree(FreeWidget):
    def __init__(self, parent, file_names):
        super().__init__()
        self.wdt = make_tree_widget(parent, file_names)
        self.set_widget(self.wdt)


def make_tree_widget(parent, file_names=[]):
    tree, dataset = _read_images(file_names)
    headers = ["Images"]

    tree_view = QTreeView()  # Instantiate the View
    # Set the models
    model = TreeModel(headers, tree)
    tree_view.setModel(model)
    tree_view.expandAll()
    tree_view.resizeColumnToContents(0)
    tissue_sholl = parent.fig.ax.plot([], [], label='tissue')[0]
    cell_sholl = parent.fig.ax.plot([], [], label='cell')[0]

    # set up callbacks whenever the selection changes
    selection = tree_view.selectionModel()
    @selection.currentChanged.connect
    def _on_change(current, previous):
        key = model.data(current)

        if key in dataset.keys():  # is an tissue image path
            if key != parent.current_tissue:
                parent.parent_viewer.layers.clear()
                tissue, scale, metadata = imread(key, channel_interest=0)
                parent.parent_viewer.add_image(tissue, scale=scale)
                parent.current_tissue = key
                parent.current_scale = scale

            child_cells = list(dataset[parent.current_tissue].values())[0].keys()
            all_ims = []
            all_bounds = []
            all_sholl = []

            for cell in child_cells:
                queries = list(gen_dict_extract(cell, dataset))
                bounds, centroid_pts, _ = _get_roi_scaled_points(queries)
                all_bounds.extend(bounds)
                im, _, _ = imread(cell)
                all_ims.append(im > 0)
                sholl = queries[0]['smorph']['sholl']
                all_sholl.extend(list(zip(sholl['radii'], sholl['nintersections'])))

            all_sholl = DataFrame(all_sholl, columns=['radii', 'nintersections'])
            mean_sholl = all_sholl.groupby('radii').sum() / len(child_cells)
            tissue_sholl.set_xdata(mean_sholl.index.tolist())
            tissue_sholl.set_ydata(mean_sholl['nintersections'].tolist())
            parent.fig.ax.relim()
            parent.fig.ax.autoscale()
            parent.fig.ax.legend()
            parent.fig.draw()

            labels = np.zeros_like(parent.parent_viewer.layers['tissue'].data, dtype=int)
            labels = reconstruct_labels(all_bounds, all_ims, out=labels)

            layer_names = [layer.name for layer in parent.parent_viewer.layers]
            if 'labels' in layer_names:
                parent.parent_viewer.layers['labels'].data = labels
            else:
                parent.parent_viewer.add_labels(
                        labels, rendering='translucent', opacity=.5,
                        scale=parent.current_scale
                        )
            parent.parent_viewer.reset_view()

        if key.endswith('.tif'):
            if key in list(dataset[parent.current_tissue].values())[0].keys():
                queries = list(gen_dict_extract(key, dataset))
                bounds, centroid_pts, _ = _get_roi_scaled_points(queries)

                im, _, _ = imread(key)
                print(key)
                labels = np.zeros_like(parent.parent_viewer.layers['tissue'].data, dtype=int)
                print(bounds, im.shape)
                labels = reconstruct_labels(bounds, [im > 0], out=labels)

                layer_names = [layer.name for layer in parent.parent_viewer.layers]
                if 'labels' in layer_names:
                    parent.parent_viewer.layers['labels'].data = labels
                else:
                    parent.parent_viewer.add_labels(
                            labels, rendering='translucent', opacity=.5,
                            scale=parent.current_scale
                            )

                for i, c in enumerate(centroid_pts):
                    scaled_c = np.array(c) * np.array(queries[i]['scale'])
                    parent.parent_viewer.camera.center = scaled_c
                    sholl = queries[i]['smorph']['sholl']
                    cell_sholl.set_xdata(sholl['radii'])
                    cell_sholl.set_ydata(sholl['nintersections'])
                    parent.fig.ax.relim()
                    parent.fig.ax.autoscale()
                    parent.fig.ax.legend()
                    parent.fig.draw()

    return tree_view


# https://stackoverflow.com/questions/60443167/qt-for-python-qtreeview-from-a-dictionary
# https://forum.image.sc/t/napari-file-browser-widget/63086/4
class TreeModel(QAbstractItemModel):
    def __init__(self, headers, data, parent=None):
        super(TreeModel, self).__init__(parent)
        """ subclassing the standard interface item models must use and 
                implementing index(), parent(), rowCount(), columnCount(), and data()."""

        rootData = [header for header in headers]
        self.rootItem = TreeNode(rootData)
        indent = -1
        self.parents = [self.rootItem]
        self.indentations = [0]
        self.createData(data, indent)

    def createData(self, data, indent):
        if type(data) == dict:
            indent += 1
            position = 4 * indent
            for dict_keys, dict_values in data.items():
                if position > self.indentations[-1]:
                    if self.parents[-1].childCount() > 0:
                        self.parents.append(self.parents[-1].child(self.parents[-1].childCount() - 1))
                        self.indentations.append(position)
                else:
                    while position < self.indentations[-1] and len(self.parents) > 0:
                        self.parents.pop()
                        self.indentations.pop()
                parent = self.parents[-1]
                parent.insertChildren(parent.childCount(), 1, parent.columnCount())
                parent.child(parent.childCount() - 1).setData(0, dict_keys)
                if type(dict_values) != dict:
                    parent.child(parent.childCount() - 1).setData(1, str(dict_values))
                self.createData(dict_values, indent)

    def index(self, row, column, index=QModelIndex()):
        """ Returns the index of the item in the model specified by the given row, column and parent index """

        if not self.hasIndex(row, column, index):
            return QModelIndex()
        if not index.isValid():
            item = self.rootItem
        else:
            item = index.internalPointer()

        child = item.child(row)
        if child:
            return self.createIndex(row, column, child)
        return QModelIndex()

    def parent(self, index):
        """ Returns the parent of the model item with the given index
                If the item has no parent, an invalid QModelIndex is returned """

        if not index.isValid():
            return QModelIndex()
        item = index.internalPointer()
        if not item:
            return QModelIndex()

        parent = item.parentItem
        if parent == self.rootItem:
            return QModelIndex()
        else:
            return self.createIndex(parent.childNumber(), 0, parent)

    def rowCount(self, index=QModelIndex()):
        """Returns the number of rows under the given parent
        When the parent is valid it means that rowCount is returning
        the number of children of parent """

        if index.isValid():
            parent = index.internalPointer()
        else:
            parent = self.rootItem
        return parent.childCount()

    def columnCount(self, index=QModelIndex()):
        """Returns the number of columns for the children of the given parent"""
        return self.rootItem.columnCount()

    def data(self, index, role=Qt.DisplayRole):
        """Returns the data stored under the given role for the item referred to by the index"""
        if index.isValid() and role == Qt.DisplayRole:
            return index.internalPointer().data(index.column())
        elif not index.isValid():
            return self.rootItem.data(index.column())

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """ Returns the data for the given role & section in the header with the specified orientation"""
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.rootItem.data(section)


class TreeNode(object):
    def __init__(self, data, parent=None):
        self.parentItem = parent
        self.itemData = data
        self.children = []

    def child(self, row):
        return self.children[row]

    def childCount(self):
        return len(self.children)

    def childNumber(self):
        if self.parentItem is not None:
            return self.parentItem.children.index(self)

    def columnCount(self):
        return len(self.itemData)

    def data(self, column):
        return self.itemData[column]

    def insertChildren(self, position, count, columns):
        if position < 0 or position > len(self.children):
            return False
        for row in range(count):
            data = [v for v in range(columns)]
            item = TreeNode(data, self)
            self.children.insert(position, item)

    def parent(self):
        return self.parentItem

    def setData(self, column, value):
        if column < 0 or column >= len(self.itemData):
            return False
        self.itemData[column] = value
