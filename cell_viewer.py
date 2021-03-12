import sys
import warnings
warnings.filterwarnings('ignore')

from smorph.util import viewer as vw


try:
    vw.identify_cells_in_tissue(sys.argv[1:])
except IndexError:
    print("No file dropped")
