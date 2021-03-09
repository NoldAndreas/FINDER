#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:34:38 2020

@author: andreas
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


hue_order = ['FINDER_1D_loop','FINDER_1D','dbscan','CAML_87B144','CAML_07VEJJ','OPTICS'];

#e8a358
my_pal = {'CAML_07VEJJ':'#eabe8e',\
          'CAML_87B144':'#d67d1d',\
          'FINDER_1D_loop':'#701ac0',\
          'FINDER_1D':'#af6eeb',\
          'dbscan':'dimgrey',\
          'OPTICS':'lightgrey',\
        };    

def PlotScatter(labels,XC,ax=[],filename=[]):
 
        if(labels == []):
            labels = -np.ones((len(XC),));
            
        # Get correctly detected:
        if(ax == []):        
            fig,ax = plt.subplots();
        mark = (labels==-1);
        sns.scatterplot(x=XC[mark,0],y=XC[mark,1],color='grey',alpha=0.2,ax=ax);
        mark = (labels>=0);
        sns.scatterplot(x=XC[mark,0],y=XC[mark,1],hue=labels[mark],palette='deep',
                        size=0.2,legend=False,ax=ax);
        ax.set_aspect('equal');
        
        
        x_0 = 0;
        y_0 = np.min(XC[:,1]) - 100;
        ax.plot([x_0,x_0+100],[y_0,y_0],'k')
        ax.annotate('$100nm$',(x_0+50,y_0+20),fontsize='large',ha='center');         
        ax.set_aspect(1);
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.axis('off');

        
        if(ax==[]):
            plt.show();
            
        if(not(filename == [])):
            plt.savefig(filename)