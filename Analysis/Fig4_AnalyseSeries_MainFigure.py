#!/usr/bin/env python3

import sys
sys.path.append("Modules/")

import pandas as pd
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

from Clustering import Clustering
from Definitions import get_datafolder


basefolder = get_datafolder()
hue_order = ['FINDER_1D_loop','CAML_07VEJJ','CAML_87B144']

show_algos = ['FINDER_1D_loop','CAML_87B144','CAML_07VEJJ']

dict_algo_names_ = {"OPTICS":"OPTICS",
                    "dbscan":"DBSCAN",
                    "CAML_07VEJJ":"CAML (07VEJJ)",
                    "CAML_87B144":"CAML (87B144)",
                    "FINDER_1D_loop":"FINDER",
                    "FINDER_1D":"FINDER  with DBSCAN"
                    }


my_pal = {'CAML_07VEJJ':'#eabe8e',\
          'CAML_87B144':'#d67d1d',\
          'FINDER_1D_loop':'#701ac0',\
          'FINDER_1D':'#af6eeb',\
          'dbscan':'dimgrey',\
          'OPTICS':'lightgrey',\
        }

def PlotScatter(labels,XC,ax=[],optionScaleBarText = False):
 
        if len(labels) == 0:
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
        y_0 = np.min(XC[:,1]) - 120;
        ax.plot([x_0,x_0+200],[y_0,y_0],'k')
        if(optionScaleBarText):
            ax.annotate('$200nm$',(x_0+100,y_0+25),fontsize='large',ha='center');         
            
        ax.set_aspect(1);
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.axis('off');



      
def AnalyseSeries(df,params,filename,axs,optionScaleBarText):
   
    filename_pickle = filename+".pickle";
    all_data = []
    infile = open(filename_pickle, "rb")
    while 1:
        try:
            all_data.append(pickle.load(infile))
        except (EOFError, pickle.UnpicklingError):
            break
    infile.close()


    #********************************************************************
    
    #Filter: Only get last instance

    x_name = params['var_1_name'];
    
    #axs[0].set_title(dict_xlabel[x_name]);  

    if(x_name == 'noise_ratio'):
        x_to_display = 1.0;
        for d in all_data:
            if(getattr(d.Geometry,'noise_ratio') == x_to_display):
                 data_to_display = d;
                 break;     
    elif(x_name == 'N_clusters'):
        x_to_display = 25;
        for d in all_data:
            if(getattr(d.Geometry,'N_clusters') == x_to_display):
                 data_to_display = d;
                 break;                      
    elif(x_name == 'Delta_ratio'):
        x_to_display = 1.0;        
        for d in all_data:
            if(d.Geometry.parameters['Delta_ratio'] == x_to_display):
                 data_to_display = d;
                 break;
    else:
        x_to_display    = np.asarray(df[x_name])[-1]; #We select last instance, i.e. case with highest perturbation
        data_to_display = all_data[-1];
    mark         = (df[x_name]==x_to_display);
    df = df[mark];


    PlotScatter(data_to_display.Geometry.labels_groundtruth,data_to_display.Geometry.XC,ax=axs[0],optionScaleBarText=optionScaleBarText);    

    ax     = axs[1];
    y_name = 'true_positives_ratio';
    sns.violinplot(ax=ax,data=df,x=x_name,y=y_name,hue='algo',hue_order=hue_order,palette=my_pal,dodge=True,inner=None,cut=0);    
    sns.stripplot(ax=ax,x=x_name, y=y_name, data=df,hue='algo',hue_order=hue_order,size=4,palette=my_pal, linewidth=0.5,dodge=True)

    ax.legend([],[], frameon=False);
    ax.set_ylabel("");
    ax.set_xlabel("");
    ax.set_xticks([]);
    #ax.set_xlabel(dict_xlabel[x_name]);        
    ax.set_ylim(-0.05,1.05);        

    ax     = axs[2];
    y_name = 'false_positives_ratio';
    boxplot_return = sns.violinplot(ax=ax,data=df,x=x_name,y=y_name,hue='algo',hue_order=hue_order,palette=my_pal,dodge=True,inner=None,cut=0);    
    sns.stripplot(ax=ax,x=x_name, y=y_name, data=df,hue='algo',hue_order=hue_order,size=4,palette=my_pal, linewidth=1,dodge=True)
 #   boxplot_return = sns.stripplot(ax=ax,x=x_name, y=y_name, data=df,hue='algo',hue_order=hue_order,size=4,color='k', linewidth=0.5,dodge=True)
    ax.legend([],[], frameon=False);
    ax.locator_params(axis='y',nbins=5)
    
    ax.set_ylabel("");
    ax.set_xticks([]);
    ax.set_xlabel("");
    ax.set_ylim(-0.05,1.05);

    return boxplot_return


basefolder = basefolder+'/Results_Fig4/';

filenamesList = ['Results_5.txt',
                 'Results_4.txt',
                 'Results_0.txt',
                 'Results_3.txt',
                 'Results_1.txt',
                 'Results_2.txt'];
                 
dict_ylabel = {'true_positives_ratio':'True positives (ratio)',\
               'false_positives_ratio':'False positives (ratio)',\
                'compute_time':'Computation time [seconds]'};
    
dict_xlabel = {'noise_ratio':'Noise vs cluster localizations',\
               'Delta_ratio':'Relative distance between clusters',\
               'N_clusters':'Number of clusters'};
    
gs_kw = dict(width_ratios=np.ones(len(filenamesList),), height_ratios=[4,4,4,0.5]);
fig,axs = plt.subplots(nrows=4,ncols=len(filenamesList),gridspec_kw=gs_kw,figsize=(16,9));#gridspec_kw=gs_kw

gs = axs[-1, -1].get_gridspec()
# remove the underlying axes
for ax in axs[-1,:]:
    ax.remove()
ax_legend = fig.add_subplot(gs[-1,:])
   
#fig,axs = plt.subplots(3,len(filenamesList),figsize=(16,8));

pos_ = np.linspace(0.11,0.79,6)
plt.text(pos_[0], 0.9, 'a', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[1], 0.9, 'b', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[2], 0.9, 'c', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[3], 0.9, 'd', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[4], 0.9, 'e', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[5], 0.9, 'f', fontsize=14, transform=plt.gcf().transFigure)

d__ = +0.025;
plt.text(pos_[0]+d__, 0.9, 'High overlap', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[1]+d__, 0.9, 'High noise', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[2]+d__, 0.9, 'Unstructured', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[3]+d__, 0.9, 'High overlap', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[4]+d__, 0.9, 'High noise', fontsize=14, transform=plt.gcf().transFigure)
plt.text(pos_[5]+d__, 0.9, 'Unstructured', fontsize=14, transform=plt.gcf().transFigure)


plt.text(0.25, 0.96, 'Low density clusters', fontsize=14, transform=plt.gcf().transFigure)
plt.text(0.65, 0.96, 'High density clusters', fontsize=14, transform=plt.gcf().transFigure)

for i,filename in enumerate(filenamesList):
    #Load corresponding parameter file:
    #fn_ = os.path.basename(filename);
    filename_json = basefolder + filename[:-4]+'_Parameters.json';
    with open(filename_json, 'r') as fp:
        params = json.load(fp)
        

    df = pd.read_csv(basefolder + filename);


    mark = np.zeros(len(df),);
    for ii in np.arange(len(df)):
        if(df['algo'][ii] in show_algos):
            mark[ii] = 1;

    df = df[mark == 1];
    
    if(i==0):
        optionScaleBarText = True;
    else:
        optionScaleBarText = False;

    boxplot_return = AnalyseSeries(df,params,basefolder+filename[:-4],axs[:,i],optionScaleBarText);
    axs[1,0].set_ylabel(dict_ylabel['true_positives_ratio']);
    axs[2,0].set_ylabel(dict_ylabel['false_positives_ratio']);


lines, labels = axs[1,1].get_legend_handles_labels();
ax_legend.set_xticks([]);
ax_legend.set_yticks([]);
ax_legend.axis('off');

labels_properNames = [dict_algo_names_[l] for l in labels];
ax_legend.legend(lines[:3], labels_properNames[:3], loc = 'center',ncol=5)

plt.subplots_adjust(wspace=0.35);    
plt.savefig(basefolder + "Fig4_MainFigure_analysis.pdf",bbox_inches="tight");
print("File saved in : "+basefolder + "Fig4_MainFigure_analysis.pdf");