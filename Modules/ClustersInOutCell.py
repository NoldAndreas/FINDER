import numpy as np
import json
import os.path
import pickle
import pandas as pd
import seaborn as sns
import scipy.stats as stats

import matplotlib.pyplot as plt
from Finder_1d import Finder_1d


from SimilarityScore import getSimilarityScore,getClusterSizesAll,getSimilarityScore_ij

from FigY_Functions import GetDensity,GetOverlay
from FigY_Functions import LoadPoints,FilterPoints
from FigY_Functions import DefineCleanedLabels,GetLineOfOptima
from FigY_Functions import PlotScatter


class ClustersInOutCell:

    def __init__(self,basefolder,algo='DbscanLoop',points_per_dimension=15):

        #Load Parameter file
        parameters= {'algo':algo,\
                     'points_per_dimension':points_per_dimension};

        if not basefolder.endswith(os.path.sep):
            basefolder += os.path.sep

        with open(basefolder+'parameters_clusterBasing.json', 'w') as fp:
            json.dump(parameters,fp,indent=4);

        if(os.path.isfile(basefolder+"X_incell_window.txt") and \
            os.path.isfile(basefolder+"X_outcell_window.txt")):
            self.XC_incell  = LoadPoints(basefolder+"X_incell_window.txt");
            self.XC_outcell = LoadPoints(basefolder+"X_outcell_window.txt");
        else:
            print("X_incell_window or X_outcell_window in folder "+basefolder+" not found!");

        self.basefolder              = basefolder;
        self.parameters              = parameters;
        self.save_name               = basefolder + 'analysis';

    def GetClusterings_InOutCell(self,skipSimilarityScore=True):

        parameters = self.parameters;
        filename   = self.save_name+"clustering";

        #******************************************************************************************
        # Load or Compute Clustering within cell
        #******************************************************************************************
        if(os.path.exists(filename+'_incell.pickle')):
            with open(filename+'_incell.pickle', 'rb') as fr:
                FD_load = pickle.load(fr);
            FD     = FD_load['FD'];

            print("Loaded Clustering results from "+filename+'_incell.pickle');
        else:
            FD      = Finder_1d(algo=parameters['algo'],\
                                points_per_dimension=parameters['points_per_dimension']);
            labels  = FD.fit(self.XC_incell,skipSimilarityScore=skipSimilarityScore);

            with open(filename+'_incell.pickle','wb') as handle:
                pickle.dump({'FD':FD}, handle,protocol=pickle.HIGHEST_PROTOCOL)
            print("Computed and saved Clustering results in "+filename+'_incell.pickle');

        #******************************************************************************************
        # Load or Compute Clustering outside cell
        #******************************************************************************************
        if(os.path.exists(filename+'_outcell.pickle')):
            with open(filename+'_outcell.pickle', 'rb') as fr:
                FD_load = pickle.load(fr);
            FD_ref  = FD_load['FD'];

            print("Loaded Clustering results from "+filename+'_outcell.pickle');
        else:
            FD_ref      = Finder_1d(algo=parameters['algo'],\
                                    points_per_dimension=parameters['points_per_dimension']);
            labels_ref  = FD_ref.fit(self.XC_outcell,self.XC_incell,skipSimilarityScore=skipSimilarityScore);

            with open(filename+'_outcell.pickle','wb') as handle:
                pickle.dump({'FD':FD_ref}, handle,protocol=pickle.HIGHEST_PROTOCOL)
            print("Computed and saved Clustering results in "+filename+'_outcell.pickle');

        #******************************************************************************************
        # Assemble data
        #******************************************************************************************
        phasespace_all                    = FD.phasespace;
        phasespace_all['labels_ref']      = FD_ref.phasespace['labels']
        phasespace_all['no_clusters_ref'] = FD_ref.phasespace['no_clusters'];
        phasespace_all['time_ref']        = FD_ref.phasespace['time'];
        self.phasespace_all               = phasespace_all;

        df_clusterSizes     = FD.clusterInfo;#GetClusterSizesAll(FD);
        df_clusterSizes_ref = FD_ref.clusterInfo;#'GetClusterSizesAll(FD_ref);

        df_clusterSizes['type']     = 'incell';
        df_clusterSizes_ref['type'] = 'outcell';
        self.df_clusterSizes_all    = df_clusterSizes.append(df_clusterSizes_ref, ignore_index=True);

    def GetSimilarityAlongOptima(self):

        filename   = self.save_name+"clusterDataWithSimilarityScores";
        if(os.path.exists(filename+'.pickle')):
            with open(filename+'.pickle', 'rb') as fr:
                data = pickle.load(fr);
            self.df_opt_th           = data['df_opt_th'];
            self.df_clusters_opt_th  = data['df_clusters_opt_th'];

            print("Loaded clusterDataWithSimilarityScores results from "+filename+'.pickle');
        else:

            self.df_opt_th   = GetLineOfOptima(self.phasespace_all[['sigma', 'threshold','no_clusters']],'threshold','no_clusters');
            phasespace_sub   = self.phasespace_all.loc[self.df_opt_th['idx'],:];
            df_clusters_opt_th_in  = getClusterSizesAll(self.XC_incell,phasespace_sub);


            print('Computing similarity score along line of optima for in cell');
            cli_similarityScore_in,similarityScore_in = getSimilarityScore(self.XC_incell,\
                                    phasespace_sub,df_clusters_opt_th_in);

            print('Computing similarity score along line of optima for out cell');
            phasespace_out                 = phasespace_sub.copy();
            phasespace_out['labels']       = phasespace_out['labels_ref'];
            phasespace_out['no_clusters']  = phasespace_out['no_clusters_ref'];
            df_clusters_opt_th_out = getClusterSizesAll(self.XC_outcell,phasespace_out);
            cli_similarityScore_out,similarityScore_out = getSimilarityScore(self.XC_outcell,\
                                    phasespace_out,df_clusters_opt_th_out);

            self.df_opt_th['similarityScore']     = similarityScore_in;
            self.df_opt_th['similarityScore_ref'] = similarityScore_out;

            df_clusters_opt_th_in['similarityScore']  = cli_similarityScore_in;
            df_clusters_opt_th_out['similarityScore'] = cli_similarityScore_out;

            df_clusters_opt_th_in['type']  = 'incell';
            df_clusters_opt_th_out['type'] = 'outcell';
            self.df_clusters_opt_th        = df_clusters_opt_th_in.append(df_clusters_opt_th_out,ignore_index=True);

            with open(filename+'.pickle','wb') as handle:
                data = {'df_opt_th':self.df_opt_th,\
                        'df_clusters_opt_th':self.df_clusters_opt_th};
                pickle.dump(data, handle,protocol=pickle.HIGHEST_PROTOCOL)
            print("Computed and saved clusterDataWithSimilarityScores results in "+filename+'.pickle');

        self.df_opt_th['threshold']          = (self.df_opt_th['threshold']).astype(int);
        self.df_clusters_opt_th['threshold'] = (self.df_clusters_opt_th['threshold']).astype(int);

        self.df_opt_th['similarityScore']          = (self.df_opt_th['similarityScore']).astype(int);
        self.df_clusters_opt_th['similarityScore'] = (self.df_clusters_opt_th['similarityScore']).astype(int);

    def GetReferenceClustering(self,bestRequiredRate=1.0,computeSimilarityScores=False,generalLimit=True):

        #*********************************************
        # Get limit and filter by cluster size
        #*********************************************
        phasespace_all_aboveT = DefineCleanedLabels(self.df_clusterSizes_all,self.phasespace_all,criterion='clusterSize',bestRequiredRate=bestRequiredRate,generalLimit=generalLimit);
        clusterInfo_aboveT    = getClusterSizesAll(self.XC_incell,phasespace_all_aboveT);

        if(computeSimilarityScores==True):
            cli_similarityScore,similarityScore      = getSimilarityScore(self.XC_incell,phasespace_all_aboveT,clusterInfo_aboveT);
            phasespace_all_aboveT['similarityScore'] = similarityScore;
            clusterInfo_aboveT['similarityScore']    = cli_similarityScore;

        self.phasespace_all_aboveT = phasespace_all_aboveT;
        self.clusterInfo_aboveT    = clusterInfo_aboveT;
        #self.df_opt_th_aboveT_ncl  = GetLineOfOptima(self.phasespace_all_aboveT[['sigma', 'threshold','no_clusters']],'threshold','no_clusters');
        self.df_opt_th_aboveT_ncl = GetLineOfOptima(self.phasespace_all_aboveT[['sigma', 'threshold','no_clusters','percent_locsIncluded']],'threshold','no_clusters');


    def GetClustering(self,criterion='percent_locsIncluded'): #'no_clusters'

        #df_opt_th_aboveT_ncl = GetLineOfOptima(self.phasespace_all_aboveT[['sigma', 'threshold','no_clusters','percent_locsIncluded']],'threshold','no_clusters');
        i_choose             = self.df_opt_th_aboveT_ncl.loc[self.df_opt_th_aboveT_ncl[criterion].argmax(),'idx'];
        #i_choose = np.argmax(self.phasespace_all_aboveT[criterion]);
        #i_check = 56;
        if(False):
            v = [];
            for i in np.arange(len(self.phasespace_all_aboveT)):
                s1= getSimilarityScore_ij(i_choose,i,self.phasespace_all_aboveT,self.clusterInfo_aboveT);
            #    print(s1)

                if(type(s1)!= bool):
                    v.append(np.sum(s1[0]));
                else:
                    v.append(0);
             #       print(np.sum(s1[0]),np.sum(s1[1]));
            self.phasespace_all_aboveT['similarityScoreChosen'] = v;
        print(self.phasespace_all_aboveT.loc[i_choose,:]);

        return self.phasespace_all_aboveT.loc[i_choose,:];

    def get_info_phasespace_approx(this,sigma,threshold):
        """returns the phasespace info closest to sigma and threshold coordinates"""

        idx = np.argmax((this.phasespace_all['sigma']==sigma) & \
            (this.phasespace_all['threshold']==threshold));

        return this.phasespace_all.loc[idx,:];

    def __get_indices_in_phasespace(this, sigma, threshold):

        sigmas = np.sort(np.unique(np.unique(this.phasespace_all['sigma'])))
        thresholds = np.sort(np.unique(this.phasespace_all['threshold']))

        sigma_index = np.zeros_like(sigma)
        for i,s in enumerate(sigma):
            sigma_index[i] = np.where(sigmas==s)[0][0]

        th_index = np.zeros_like(threshold)
        for i,t in enumerate(threshold):
            th_index[i] = np.where(thresholds==t)[0][0]

        return sigma_index,th_index

    def plot_phasespace(this, ax, value="no_clusters", type="incell", criterion="clusterSize", min_criterion=0):

        df1 = this.df_clusterSizes_all[(this.df_clusterSizes_all[criterion]>min_criterion) & (this.df_clusterSizes_all['type']==type)]

        if value=="no_clusters":
            df1 = df1[['sigma','threshold','type']]
            df1 = df1.groupby(['threshold','sigma']).agg([('value','count')])
        elif value=="similarityScore":
            df1 = df1[['sigma','threshold','similarityScore']]
            df1 = df1.groupby(['threshold','sigma']).agg([('value','sum')])

        df1.columns=df1.columns.droplevel(0)
        df1.reset_index(inplace=True)

        heatmap1_data = pd.pivot_table(df1, values='value',
                            index=['threshold'],
                            columns='sigma')
        heatmap1_data.index = heatmap1_data.index.astype(int)
        heatmap1_data.columns = np.round(heatmap1_data.columns,2)

        ax = sns.heatmap(heatmap1_data,ax=ax)
        ax.invert_yaxis()
        ax.set_title(f"{value} for {criterion} > {min_criterion} for {type}");

        sigma_idx, th_idx = this.__get_indices_in_phasespace(this.df_opt_th['sigma'],this.df_opt_th['threshold'])
        ax.plot(sigma_idx+0.5, th_idx+0.5,'b')

    def plot_phasespace_in_vs_outcell(this,value='no_clusters',criterion="clusterSize", min_criterion=0):
        fig,axs = plt.subplots(1,2,figsize=(12,6))

        this.plot_phasespace(axs[0], value=value, type="incell", criterion=criterion, min_criterion=min_criterion)
        this.plot_phasespace(axs[1], value=value, type="outcell", criterion=criterion, min_criterion=min_criterion)
        plt.tight_layout()
        plt.savefig(f"{this.save_name}_{value}_for_{criterion}_greaterthan_{min_criterion}.pdf",bbox_inches="tight")
        plt.close('all')


    def plot_clustersizes_along_optima(this,min_clustersize=30):
        """Plot clustersize distribution along line of optima in phasespace"""

        df_clusters_opt_th = this.df_clusters_opt_th

        df_in  = df_clusters_opt_th[(df_clusters_opt_th['type']=='incell')  & (df_clusters_opt_th['clusterSize']>=min_clustersize)]
        df_out = df_clusters_opt_th[(df_clusters_opt_th['type']=='outcell') & (df_clusters_opt_th['clusterSize']>=min_clustersize)]

        fig,axs = plt.subplots(1,2,figsize=(12,6))

        ax = axs[0]
        sns.swarmplot(data=df_in,y='clusterSize',x='threshold',ax=ax,size=2,color='k');
        sns.boxplot(data=df_in,y='clusterSize',x='threshold',ax=ax)
        ax.set_title('in cell')
        ax.set_ylim(0,100)

        ax = axs[1]
        sns.swarmplot(data=df_out,y='clusterSize',x='threshold',ax=ax,size=2,color='k');
        sns.boxplot(data=df_out,y='clusterSize',x='threshold',ax=ax)
        ax.set_title('out cell')
        ax.set_ylim(0,100)

        plt.tight_layout()
        plt.savefig(f"{this.save_name}_ClusterSize_ResultsAlongOptima_minClustersize{min_clustersize}.pdf",bbox_inches="tight");

    def plot_statistics(this):
        """plots several statistical measures along line of optima"""

        df_grouped = this.df_clusters_opt_th.groupby(['threshold','type'])['clusterSize'].\
        agg([('median',np.median),\
            ('count','count'),\
            ('min','min'),\
            ('max','max'),\
            ('std','std'),\
            ('cv',stats.variation),\
            ('skewness',stats.skew),\
            ('kurtosis',stats.kurtosis),\
            ('fano',lambda d_ : np.var(d_)/np.mean(d_))])

        fig,axs = fig,axs = plt.subplots(3,3,figsize=(14,12));

        for key,ax in zip(df_grouped.keys(),axs.flatten()):
            sns.lineplot(data=df_grouped,x='threshold',y=key,hue='type',ax=ax)
            ax.set_ylabel(key)

        plt.tight_layout()
        plt.savefig(f"{this.save_name}_Statistics_alongLineOfOptima.pdf",bbox_inches="tight");

    def plot_scatter_selection(this,XC,labels,df_clusterlist,ax,scalebar):
        labels_c = labels.copy();
        list_of_labels_to_include = list(df_clusterlist['labels']);
        for l in np.unique(labels):
            if(not (l in list_of_labels_to_include)):
                labels_c[labels_c == l] = -1;
        PlotScatter(XC,labels_c,ax=ax,scalebar=scalebar);

    def plot_gif_along_optima(this,type_,criterion='clusterSize',min_criterion=0):

        png_folder  = this.save_name+"GoThroughOptima"+type_+"/";
        df_clusters = this.df_clusters_opt_th;

        if(type_ == "incell"):
            XC   = this.XC_incell;
            labels_name = 'labels';
        # df_clusters = clusterInfo_in;
        else:
            XC   = this.XC_outcell;
            labels_name = 'labels_ref';
            #df_clusters = clusterInfo_out;

        if(not os.path.exists(png_folder)):
            os.mkdir(png_folder);

        count = 0;
        for idx,row in this.df_opt_th.iterrows():
            fig,axs = plt.subplots(1,1,figsize=(12,12));
            ax = axs;
            idx = row['idx'];

            mark = (df_clusters['index']==idx) & (df_clusters['type']==type_)\
                        & (df_clusters[criterion]>=min_criterion);


            this.plot_scatter_selection(XC,this.phasespace_all.loc[idx,labels_name],df_clusters.loc[mark,:],ax,scalebar=True);

            xymin    = np.min(XC,axis=0);
            xymax    = np.max(XC,axis=0);
            delta_xy = np.max(XC,axis=0) - np.min(XC,axis=0);
            x        = xymax[0] - delta_xy[0]*2/10;
            y        = xymax[1] + delta_xy[1]*1/10;
            ax.text(x,y,'minPts = '+str(int(row['threshold']))+\
                    " , r = "+str(np.round(row['sigma'],3)),ha='center', va='center');

            plt.savefig(png_folder+"goThroughOptima_"+str(count)+".png",bbox_inches="tight");
            plt.close('all')
            count += 1;

        filename= this.save_name+"_video_"+type_+"_clusterings_min"+criterion+"_"+str(min_criterion)+".gif"
        cmd_ = "convert -delay 50 $(ls "+png_folder+"*.png | sort -V) "+ filename;
        print(os.popen(cmd_).read())
        print(os.popen("rm -r "+png_folder).read())


    def plot_overview_scatterplots(this):

        fig,axs = plt.subplots(1,2,figsize=(12,5));
        PlotScatter(this.XC_incell,ax=axs[0],scalebar=True)
        PlotScatter(this.XC_outcell,ax=axs[1],scalebar=True)

        axs[0].set_title('in cell,'+str(len(this.XC_incell))+' points');

        axs[1].set_title('out cell, '+str(len(this.XC_outcell))+' points');
        for ax in axs:
            ax.set_aspect('equal');

        if(True):
            axs[0].axis('off');
            axs[1].axis('off')

            axs[0].set_title('In cell');
            axs[1].set_title('Out cell');

        plt.savefig(this.save_name+"_localizations_incell_vs_outcell.pdf",bbox_inches="tight")
        plt.close('all')

    def plot_similarity_results_along_optima(this,criterion='clusterSize',min_criterion=0):

        df_in  = this.df_clusters_opt_th[(this.df_clusters_opt_th['type']=='incell')  & (this.df_clusters_opt_th[criterion]>=min_criterion)];
        df_out = this.df_clusters_opt_th[(this.df_clusters_opt_th['type']=='outcell') & (this.df_clusters_opt_th[criterion]>=min_criterion)];

        fig,axs = plt.subplots(3,2,figsize=(12,15))

        ax = axs[0,0]
        sns.boxplot(data=df_out,y='similarityScore',x='threshold',ax=ax)
        ax.set_title('Out cell')

        ax = axs[0,1]
        sns.boxplot(data=df_in,y='similarityScore',x='threshold',ax=ax)
        ax.set_title('In cell')

        ax = axs[1,0]
        sns.scatterplot(data=df_out,y='similarityScore',x='clusterSize',hue='threshold',ax=ax)

        ax = axs[1,1]
        sns.scatterplot(data=df_in,y='similarityScore',x='clusterSize',hue='threshold',ax=ax)

        ax = axs[2,0]
        sns.histplot(df_out,x='threshold',ax=ax,binwidth=1)

        ax = axs[2,1]
        sns.histplot(df_in,x='threshold',ax=ax,binwidth=1);

        plt.tight_layout()
        plt.savefig(f"{this.save_name}_SimilarityResultsAlongOptima_{criterion}_greaterThan_{min_criterion}.pdf",bbox_inches="tight");

    def plot_included_localizations(this,criterion='clusterSize',min_criterion=0):

        df_in  = this.df_clusters_opt_th[(this.df_clusters_opt_th['type']=='incell')  & (this.df_clusters_opt_th[criterion]>=min_criterion)];
        df_out = this.df_clusters_opt_th[(this.df_clusters_opt_th['type']=='outcell') & (this.df_clusters_opt_th[criterion]>=min_criterion)];

        fig,axs = plt.subplots(1,2,figsize=(12,5))
