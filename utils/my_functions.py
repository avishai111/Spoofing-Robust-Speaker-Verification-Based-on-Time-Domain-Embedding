    
import torch
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import time
from time import time
import mlflow

# function to compute the equal error rate with interpolation1d and brentq
def compute_eer(y, y_score): # y is the ground truth, y_score is the prediction score
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1) # pos_label=1 means that the positive class is 1  

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr,kind='linear')(x), 0., 1.) # brentq is a root finding algorithm
    thresh = interp1d(fpr, thresholds)(eer) # interp1d is a linear interpolation function
    return eer, thresh # eer is the equal error rate, thresh is the threshold at eer


def compute_eer_2(label, pred_score, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred_score, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    #print(eer_1)
    #print(eer_2)
    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer_1,eer_2,eer,eer_threshold

# function to plot DET curve with EER point
def DETCurve(fpr, fnr, eer_fpr,eer_fnr,model_name,plot_type="plot",is_mlflow=False):
    plt.figure()
    FPR= fpr 
    TPR =1-fnr
    FNR = fnr
    if plot_type == "plot":
        plt.plot(FPR*100, FNR*100, color='b', lw=2, label='DET Curve')
    elif plot_type == "step":
        plt.step(FPR*100, FNR*100, color='b', lw=2, label='DET Curve')
        

    # Plot EER point
    #eer_index = np.argmin(np.abs(FPR - (1 - TPR)))
    #eer_fpr = FPR[eer_index]*100
    #eer_fnr = FNR[eer_index]*100
    eer_fpr = eer_fpr*100
    eer_fnr = eer_fnr*100
    
    if plot_type == "plot":
        plt.plot(eer_fpr, eer_fnr, 'ro', label=f'EER ({eer_fpr:.4f}%, {eer_fnr:.4f}%)')
    elif plot_type == "step":
        plt.step(eer_fpr, eer_fnr, 'ro', label=f'EER ({eer_fpr:.4f}%, {eer_fnr:.4f}%)')

    # Annotate EER point
    plt.annotate(f'EER ({eer_fpr:.4f}%, {eer_fnr:.4f}%)',
                xy=(eer_fpr, eer_fnr),
                xytext=(eer_fpr - eer_fpr/4, eer_fnr + eer_fnr/4),
                arrowprops=dict(arrowstyle='->'))
    # plt.xlabel('False Positive Rate (FPR) (%)')
    # plt.ylabel('False Negative Rate (FNR) (%)')
    # plt.title('Detection Error Tradeoff (DET) Curve (%) with model name - ' + model_name)
    plt.xlabel('False Acceptance Rate (FAR) (%)')
    plt.ylabel('False Rejection Rate (FRR) (%)')
    plt.title('Detection Error Tradeoff (DET) Curve (%) with model name - ' + model_name)

    plt.legend()
    plt.grid(True)

    # Show the plot
    if is_mlflow == True:
        if "validation" in model_name.lower():
            mlflow.log_figure(plt.gcf(), "DET_Curve_validation.png")
        if "test" in model_name.lower():
            mlflow.log_figure(plt.gcf(), "DET_Curve_test.png")
    plt.show(block = False)


#timing function to calculate the time taken for the function to execute
def timing(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


#function for XAI for RFC to see the feature importance
def get_columns_names_feature_importance(substruct = True): #substruct = True means that we want to substruct the distance from spoof and human
    number_filters = 2

    filter_names = ["gammtone_inv","gammatone"] #filters names 
   # filter_names = ["gammatone_inv","gammatone"]
    number_channels = 10 #number of channels    

    distance_from_spoof_and_human = 2 #distance from spoof and human
    distance_from_spoof_and_human_names = ["d_(p,p_h)","d_(p,p_s)"] #distance from spoof and human names
    number_of_distances = 8 #number of distances
    distance_names = ["Chi-square","Correlation","Hellinger","Intersection","Jensen-Shannon","Symmetrised Kullback-Leibler","Kullback-Leibler Divergence","Modified Kolmogorov-Smirnov"] #distance names

    columns_names = [] #columns names

    max_name_length = number_filters*number_channels*number_of_distances*distance_from_spoof_and_human #max name length


    #for loop to create the columns names when substruct = True 
    if substruct == True:
        for i in range(0,number_filters):
            for j in range(0,number_channels):
                for k in range(0,number_of_distances):
                        #print(f"filter-{filter_names[i]}-channel-{j+1}-distance-{distance_names[k]}-[d_(p,p_s)-d(p,p_h)] - {max_name_length//2}")
                            columns_names.append(f"filter-{filter_names[i]}-channel-{j+1}-distance-{distance_names[k]}-[d_(p,p_s)-d(p,p_h)]")


    #for loop to create the columns names when substruct = False
    if substruct == False:
        for i in range(0,number_filters):
            for j in range(0,number_channels):
                    for l in range(0,distance_from_spoof_and_human):
                        for k in range(0,number_of_distances):
                            #print(f"filter{filter_names[i]}-channel-{j}-distance-{distance_names[k]}-{distance_from_spoof_and_human_names[l]} - {max_name_length}")
                            columns_names.append(f"filter{filter_names[i]}-channel-{j+1}-distance-{distance_names[k]}-{distance_from_spoof_and_human_names[l]}")
                            
    return columns_names,max_name_length
                        
#function to get the real channel number                        
def get_real_channel(channels,num_of_channels):
    if num_of_channels <= 1:
        real_channel = 0
        return real_channel
    
    real_channel = []
    for i in range(len(channels)):
        real_channel.append(channels[i] - num_of_channels/2);
        if real_channel[i] <= 0:
            real_channel[i] = real_channel[i] + num_of_channels
    return real_channel
