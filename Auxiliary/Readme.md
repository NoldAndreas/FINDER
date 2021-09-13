

Steps to produce noise and signal files:

Step1_Txt2Heatmap.py
Saves point file as a heatmap
[filename] => [filename + "_heatmap_D.txt"]


Step2_AnalyzeImages:
Identifies ROI (for signal)
[filename + "_heatmap_D.txt"] => [filename + "_heatmap_mask.txt"]

Step3_MaskImage:
Separates points in signal and noise
[filename + "_heatmap_mask.txt"] => [filename + "_X_signal.txt"], [filename + "_X_noise.txt"]

