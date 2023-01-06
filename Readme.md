# FINDER

-------------------------------

# Installing required packages

Set up a python environment and install the dependencies, e.g. using the
following commands:

```shell
python3 -m venv finder_env
source finder_env/bin/activate
pip install -r requirements.txt
```
This was tested using python version 3.8.12.

## Using FINDER

Using `FINDER` is really simple.
To run the `FINDER` algorithm on your own localization data, add

```python
from Finder import Finder

FD      = Finder()
labels  = FD.fit(XC)
result_ = FD.selected_parameters
```

to your code, analogous to DBSCAN in the `sklearn.cluster` package.
FINDER will choose global clustering parameters according to the overall noise levels / the robustness detected in the dataset.

We created two notebooks to guide you through it:

* [synthetic_data.ipynb](https://github.com/NoldAndreas/FINDER/blob/master/synthetic_data.ipynb) will guide you through the creation of synthetic datasets (based on true recordings!) in which you can control the level
of noise and arange the clusters in various forms. The dataset will be later clustered using `FINDER`.

* [real_data.ipynb](https://github.com/NoldAndreas/FINDER/blob/master/real_data.ipynb) applies `FINDER` to a true recording, for which the ground truth is not known.


## Using the FINDER app

The `FINDER` app allows you to apply finder to your data directly.
Make sure you have installed all the required packages (as explained above, see _Installing required packages_).
To use the app, simply navigate with your shell to the directory `./app` inside the `./Finder` folder.

Then type on the terminal:

```shell
python3 app.py
```
The app should launch automatically, but if it does not simply click on the link displayed on the terminal and open it.
Using the app, you can browse your computer for the data you want to cluster and select the parameter you want to use.
`FINDER` will be applied to your data and the labels will be returned. For the app to work properly, your data must be saved in a text file.

-----------------------------------------

## How to file to reproduce existing figures

1. To reproduce Figures of the manuscript ["Unbiased choice of global clustering parameters in single-molecule localization microscopy"](https://www.biorxiv.org/content/10.1101/2021.02.22.432198v1), run the respective python files in the "ProduceFigures/" Subfolder. This will analyse the precomputed clustering results in the "../Data_Figures/" folder. **NOTE: The working directory is assumed to be ```Code/```. To run the files from the command line, change ```sys.path.append("Modules/")``` to ```sys.path.append("../Modules/")```**

2. To re-compute clustering results:
	- copy the parameter file ending on ```_Parameters.json``` from the subfolder of "../Data_Figures/" (eg. ```../Data_Figures/Results_Fig3/Results_3mers_Parameters.json``` into the ```../Data_Figures/Input``` folder. Make sure only one file is in the folder.
	- Then go to the ```Modules``` folder and run ```ComputeSeries.py``` or ```Modules/RunAll.py```. ```Modules/RunAll.py``` processes all input files in the Input folder. A folder with the date of the computation will be created in the ```../Data_Figures/```-folder (e.g. ```Results_2022_01_19_15_31_31_0```).
	- To test this, leave the file ```Fig3_a_3mers.json``` as only file in the input folder, then run ```python3 ComputeSeries.py```. This re-computes the clustering results shown in Fig. 3a.
	- Runtime: The example ```Fig3_a_3mers.json``` runs in <2 minutes on a local machine (2,3 GHz Quad-Core Intel Core i5, 8 GB 2133 MHz LPDDR3). Running the ```*.json``` files from ```Results_Fig4``` takes >1hr for each file. The localization-source data for Figure 5 is not included in the repository, but processed files can be found in the respective folders.


****************************************************************************************************
NOTE:
For the purpose of reproducing the results in the manuscript, the files include the CAML-code and pre-trained models published on [Gitlab](https://gitlab.com/quokka79/caml), see also
Williamson et al. "Machine learning for cluster analysis of localization
microscopy data", Nat. Comm. (2020) .
****************************************************************************************************

## Analysis of localization data

1. We first segment the image obtained from the localizations into two a low-density region (outcell) and a high-density region (incell). We then select part of the image and perform cluster analysis with DBSCAN or DBSCANLoop for a full range of clustering parameters. For a given dataset, localizations are read from the file ```XC.hdf5``` located in the Input folder. (e.g. ```TTX_24hr_2/Input/XC.hdf5```. Results are saved in files ```Output/X_incell.txt``` and ```Output/XC_outcell.hdf5.```
	- To do this, run ```Code/dash-split-cell/app.py```, select parameters and save the split.
	- Alternatively, run ```Analysis/analysis_1_DefineInOutCell.py``` with the respective subfolder of ```Data_AnalysisOrganized```, and with adapted parameters if necessary. Parameters are loaded from the file ```Input/parameters_splitInOutCell.json```.

2. To define a ROI and run the clustering analysis, run ```Analysis/analysis_2_Clustering.py```

3. To analyze the clustering results, run either ```Dash/dash-show-clustering``` or ```Analysis/analysis_3_plotting.ipynb```.

4. An alternative, exploratory analysis is given in ```Analysis/FigY1_Exploration_SingleDataset.ipynb```, where a square window is analyzed. This is not optimized to work with the folders in ```Data_AnalysisOrganized```, and runs with a precomputed example in ```Data_Other/MikeData/Analysis_dataWindow_1```.
