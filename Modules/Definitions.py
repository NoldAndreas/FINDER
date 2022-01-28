import os 

def get_datafolder():
    """returns path to folder with figure data"""    
    return os.path.dirname(os.path.abspath(__file__))+'/../../Data_Figures/'

hue_order = ['FINDER_1D_loop','FINDER_1D','dbscan','CAML_87B144','CAML_07VEJJ','OPTICS'];