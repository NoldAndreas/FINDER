"""splits localizations into inside and outside cell and clusters subset"""
from Modules.cell_segmentation import CellSegmentation
from Modules.Definitions import get_datafolder
import os


CellSegmentation(os.path.join(get_datafolder('Data_AnalysisOrganized'),"TTX_control_new")).segmentation()
    
