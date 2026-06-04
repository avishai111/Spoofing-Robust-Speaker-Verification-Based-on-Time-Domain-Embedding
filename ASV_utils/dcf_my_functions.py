
import torch
import torch.nn.functional as F 
from sklearn.metrics.pairwise import cosine_similarity
import torchaudio
import tqdm
from ASV_utils.ASV_my_classes import speaker_class,enrollment_file
import pandas as pd
import numpy as np
import os
import utils.eval_metrics as eval_metrics
#######################################################################################################################################################
#######################################################################################################################################################

def dcf_formula(Pfa_asv,Pmiss_asv,Prior_target,Prior_non_target,cost_model_dcf,is_print=True):
    dcf = cost_model_dcf['Cmiss_asv']*Prior_target*Pmiss_asv + cost_model_dcf['Cfa_asv']*Prior_non_target*Pfa_asv
    if is_print:
        print("The DCF is: ",dcf)
    return dcf

#######################################################################################################################################################
#######################################################################################################################################################
def calculate_dcf(list_files,df_protocol,enrollment,embeddings,asv_score_list,Prior_target,Prior_non_target,cost_model_dcf,is_print=True): # CALCULATE THE DCF FOR THE ASV SYSTEM
 
    path_names = pd.Series(np.array(list_files)) # Get the path names of the data set # Get the path names of the data set
    all_speakers_id_enroll = np.array(enrollment.get_all_speakers_id())  # Get all the speakers id in the enrollment data set
    
    labels_list = []; scores_list = []; # labels_list and scores_list for each asv score
    for _, row in tqdm.tqdm(df_protocol.iterrows(), total=len(df_protocol)): # over each row in the protocol file
        
        audio_filename = row['audio_filename'] # Get the audio filename
        speaker_id = row['speaker_id'] # Get the speaker id
        index_embeddings = path_names.str.contains(audio_filename) # Get the index of the current audio filename
        curr_embedding = embeddings[index_embeddings] # Get the current embedding
        label = row['label'] # Get the label of the current audio filename
        max_score = -1 # initialize the max score
        find_speaker_in_enroll = False # initialize the find_speaker_in_enroll flag
        for index_speaker,embedding_in_enroll_per_speaker in enumerate(enrollment.get_all_enrollment_embeddings()): # over each speaker in the enrollment data set
            embedding_in_enroll_per_speaker = np.array(embedding_in_enroll_per_speaker)
            for _,embedding_in_enroll in enumerate(embedding_in_enroll_per_speaker):
                if speaker_id == all_speakers_id_enroll[index_speaker]: # i know that the speaker id of the inferece input
                    find_speaker_in_enroll = True
                    score = cosine_similarity(embedding_in_enroll.squeeze().reshape(1, -1), curr_embedding.reshape(1, -1)) # calculate the cosine similarity between the current embedding and the enrollment embedding
                    
                    if score.item() > max_score: # if the score is higher than the asv score
                        max_score = score.item()
        
        if find_speaker_in_enroll == False:
            raise Exception("The speaker id is not in the enrollment data set")
        
        labels_list.append(label) # the current embedding is in the enrollment embeddings
        scores_list.append(max_score) # the score of the current embedding with the enrollment embedding
                                

    df_results = pd.DataFrame({'label_ground_truth': labels_list,'pred_scores': scores_list})
    
    dcf_list = []
    target_scores = df_results.loc[df_results['label_ground_truth'] == "target"]["pred_scores"]
    nontarget_scores = df_results.loc[df_results['label_ground_truth'] == "nontarget"]["pred_scores"]
    
    eer, thr_err, frr, far = eval_metrics.compute_eer(target_scores,nontarget_scores)
    
    for asv_score in asv_score_list: # over each asv score
        
        print("current asv score is: ",asv_score)
        
        Pfa_asv, Pmiss_asv, _ = eval_metrics.obtain_asv_error_rates(target_scores,nontarget_scores,np.array([]),asv_score)
        dcf = dcf_formula(Pfa_asv,Pmiss_asv,Prior_target = Prior_target,Prior_non_target= Prior_non_target,cost_model_dcf=cost_model_dcf,is_print=is_print)
        dcf_list.append(dcf)
        
    return df_results,dcf_list, eer, thr_err, frr, far

#######################################################################################################################################################
#######################################################################################################################################################
def calculate_dcf_more_details(list_files,df_protocol,enrollment,embeddings,asv_score_list,Prior_target,Prior_non_target,cost_model_dcf,is_print=True): # CALCULATE THE DCF FOR THE ASV SYSTEM
    
    target_count_ground_truth = df_protocol['label'].value_counts().get('target', 0) # Get the number of target in the ground truth
    non_target_count_ground_truth = df_protocol['label'].value_counts().get('nontarget', 0) # Get the number of nontarget and spoof in the ground truth
    
    path_names = pd.Series(np.array(list_files)) # Get the path names of the data set # Get the path names of the data set

    results_list = []; dcf_list = []; #results_list and dcf_list for each asv score
    Pfa_asv_list = []; Pmiss_asv_list = []; # Pfa_asv_list and Pmiss_asv_list for each asv score
    all_speakers_id_enroll = np.array(enrollment.get_all_speakers_id())  # Get all the speakers id in the enrollment data set
    
    for asv_score in asv_score_list: # over each asv score
        print("current asv score is: ",asv_score)
        
        label_ground_truth = []; pred_labels_list = []; scores_list = []; # labels_list and scores_list for each asv score
        fa_asv_count_pred = 0; miss_asv_count_pred = 0; # target_count_pred and non_target_count_pred for each asv score
        for _, row in tqdm.tqdm(df_protocol.iterrows(), total=len(df_protocol)): # over each row in the protocol file
            
            audio_filename = row['audio_filename'] # Get the audio filename
            speaker_id = row['speaker_id'] # Get the speaker id
            index_embeddings = path_names.str.contains(audio_filename) # Get the index of the current audio filename
            curr_embedding = embeddings[index_embeddings] # Get the current embedding
            label = row['label'] # Get the label of the current audio filename
            label_ground_truth.append(label)
            find_speaker_in_enroll = False # flag to check if the current speaker is in the enrollment data set
            max_score_curr_embedding = -1 # the max score of the current embedding with all the enrollment embeddings
            
            for index_speaker,embedding_in_enroll_per_speaker in enumerate(np.array(enrollment.get_all_enrollment_embeddings())): # over each speaker in the enrollment data set
                for _,embedding_in_enroll in enumerate(embedding_in_enroll_per_speaker):
                    if speaker_id == all_speakers_id_enroll[index_speaker]:
                        score = cosine_similarity(embedding_in_enroll.squeeze().reshape(1, -1), curr_embedding.reshape(1, -1)) # calculate the cosine similarity between the current embedding and the enrollment embedding
                        
                        if score.item() > max_score_curr_embedding: # if the current score is bigger than the max score
                            max_score_curr_embedding = score.item() # update the max score
                        
                        
                        if score.item() > asv_score: # if the current score is bigger than the asv score
                            find_speaker_in_enroll = True # update the flag
                            pred_labels_list.append("target") # the current embedding is in the enrollment embeddings
                            scores_list.append(score.item()) # the score of the current embedding with the enrollment embedding
                            if label == 'nontarget': # if the current label is nontarget, then update the fa asv count
                                fa_asv_count_pred += 1
                            break
                        
            if find_speaker_in_enroll == False:
                pred_labels_list.append("nontarget") # the current embedding is not in the enrollment embeddings
                scores_list.append(max_score_curr_embedding) # the max score of the current embedding with all the enrollment embeddings
                if label == 'target': #if the current label is target, then update the miss asv count
                        miss_asv_count_pred += 1
        
        df_results = pd.DataFrame({'label_ground_truth': label_ground_truth,'pred_labels': pred_labels_list,'scores': scores_list})
        
        Pfa_asv = fa_asv_count_pred / non_target_count_ground_truth
        
        Pmiss_asv = miss_asv_count_pred / target_count_ground_truth
        
        dcf = dcf_formula(Pfa_asv,Pmiss_asv,Prior_target,Prior_non_target,cost_model_dcf,is_print=True)
        
        Pfa_asv_list.append(Pfa_asv)
        Pmiss_asv_list.append(Pmiss_asv)
        dcf_list.append(dcf)        
        results_list.append(df_results) 
    
    target_scores = df_results.loc[df_results['label_ground_truth'] == "target"]["pred_scores"]
    nontarget_scores = df_results.loc[df_results['label_ground_truth'] == "nontarget"]["pred_scores"]
    
    eer, thr_err, frr, far = eval_metrics.compute_eer(target_scores,nontarget_scores)
        
    return results_list,dcf_list,Pfa_asv_list,Pmiss_asv_list,eer, thr_err, frr, far
