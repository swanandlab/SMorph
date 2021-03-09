import sys
import warnings
warnings.filterwarnings('ignore')

from smorph.util import viewer as vw


try:
    droppedFile = sys.argv[1]
    vw.identify_cell_in_tissue(droppedFile)
except IndexError:
    print("No file dropped")
