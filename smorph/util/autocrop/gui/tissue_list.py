from magicclass import magicclass
from magicclass.widgets import FreeWidget
from napari._qt.containers import QtListView, QtListModel
from napari.utils.events import SelectableEventedList
from qtpy.QtCore import QModelIndex, Qt


class TissueElement:
    def __init__(self, name):
        self.name = name

class TissueModel(QtListModel):
    def __init__(self, root: SelectableEventedList[TissueElement]):
        super().__init__(root)

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data at `index` for the requested `role`."""
        if role == Qt.DisplayRole:
            my_gene = index.data(Qt.UserRole)
            return my_gene.name
        return super().data(index, role)

class TissueListView(QtListView):
    def __init__(self, root, parent=None):
        super().__init__(root=root, parent=parent)
        self.setModel(TissueModel(root))

class MagicTissueList(FreeWidget):
    def __init__(self, tissue_list=[]):
        super().__init__()
        selectable_evented_tissues = SelectableEventedList(tissue_list)
        my_list_widget = TissueListView(selectable_evented_tissues)
        self.selectable_evented_tissues = selectable_evented_tissues
        selectable_evented_tissues.select_all()
        self.wdt = my_list_widget
        self.set_widget(self.wdt)

@magicclass(widget_type="groupbox", layout="horizontal")
class GroupsClassifier:
    all = MagicTissueList()

    @magicclass(widget_type="none")
    class Groups:
        @magicclass(widget_type="none", layout="horizontal")
        class Group1:
            def add(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__
                for selection in grandparent.all.selectable_evented_tissues.selection:
                    self.group_list.selectable_evented_tissues.append(selection)
                grandparent.all.selectable_evented_tissues.remove_selected()
            group_list = MagicTissueList()

        @magicclass(widget_type="none", layout="horizontal")
        class Group2:
            def add(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__
                for selection in grandparent.all.selectable_evented_tissues.selection:
                    self.group_list.selectable_evented_tissues.append(selection)
                grandparent.all.selectable_evented_tissues.remove_selected()
            group_list = MagicTissueList()
