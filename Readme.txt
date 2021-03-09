
****************************************************************************************************
How to file to reproduce results of the manuscript

"Unbiased choice of global clustering parameters in single-molecule localization microscopy"
https://www.biorxiv.org/content/10.1101/2021.02.22.432198v1
****************************************************************************************************
MIT License
****************************************************************************************************1. To reproduce Figures of the manuscript, run the respective python files in the "Code/" Subfolder. This will analyse the precomputed clustering results in the "Data/" folder. 

2. To re-compute a clustering results, copy the respective "*_Parameters.json" file found in the subfolder of "Data/" (eg. "Data/Results_Fig3/Results_3mers_Parameters.json" into the "Data/Input" folder. Make sure only one file is in the folder. Then go to the "Code" folder and run 

"python3 ComputeSeries.py" 

from the command line. A folder with the date of the computation will be created in the "Data/"-folder. To test, leave the file "Fig3_a_3mers.json" as only file in the input folder, then run "python3 ComputeSeries.py". This re-computes the clustering results shown in Fig. 3a. The example "Fig3_a_3mers.json" runs in <2 minutes on a local machine (2,3 GHz Quad-Core Intel Core i5, 8 GB 2133 MHz LPDDR3). Running the "*.json" files from "Results_Fig4" takes >1hr for each file. The localization-source data for Figure 5 is not included in the repository, but processed files can be found in the respective folders.

3. To run the FINDER algorithm on your own data "XC", add 

"from Finder_1d import Finder_1d

FD      = Finder_1d();
labels  = FD.fit(XC);                
result_ = FD.selected_parameters;"

to your code, analogous to DBSCAN in the sklearn.cluster package. FINDER will choose global clustering parameters according to the overall noise levels / the robustness detected in the dataset.

****************************************************************************************************
NOTE: 
For the purpose of reproducing the results in the manuscript, the files include the CAML-code and pre-trained models published under
https://gitlab.com/quokka79/caml
Williamson et al. "Machine learning for cluster analysis of localization
microscopy data", Nat. Comm. (2020) 
****************************************************************************************************
The code was tested with Python 3.8.5, on MAC OS 10.15.7. 
****************************************************************************************************