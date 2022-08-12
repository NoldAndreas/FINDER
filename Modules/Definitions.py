import os


def get_datafolder(subdatafolder='Data_Figures'):
    """returns path to folder with figure data"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..',subdatafolder)


hue_order = ['FINDER_1D_loop', 'FINDER_1D', 'dbscan', 'CAML_87B144', 'CAML_07VEJJ', 'OPTICS']

data_folder = get_datafolder() + os.path.sep
base_folder = get_datafolder("") + os.path.sep
