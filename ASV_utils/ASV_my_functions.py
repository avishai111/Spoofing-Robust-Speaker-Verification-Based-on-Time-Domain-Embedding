import csv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 
from sklearn.metrics.pairwise import cosine_similarity
import torchaudio
import tqdm
from ASV_utils.ASV_my_classes import speaker_class,enrollment_file
import pandas as pd
import numpy as np
import os

######################################################################################################################

def make_enrollment(file_path,gender):
    '''
    This function gets a file path of enrollment file 
    and return a list of the enrollment data
    '''
    speakers = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=' ')
        for row in csv_reader:
            speaker = row[0]
            files = row[1].split(",")
            speakers.append(speaker_class(speaker_id = speaker,enrollment_files = files,gender = gender))
    enrollment = enrollment_file(speakers)     
    return enrollment

######################################################################################################################

def extract_embeddings(list_files_embeddings,embeddings,enrollment):
    '''
    This function gets a list of enrollment data and return a list of the enrollment embeddings
    '''
    enrollment_embeddings = []
    for speaker in enrollment.speakers_list:
        is_subset = np.isin(np.array(list_files_embeddings), np.array(speaker.get_enrollment_files()))
        indices = np.where(is_subset)[0]
        enrollment_embeddings.append(embeddings[indices])
        speaker.indexes_big_embeddings_list = indices
  
    if enrollment_embeddings is None:
        raise Exception("The enrollment embeddings is None, dont find the speaker in the embeddings list")
    
    enrollment_embeddings = np.array(enrollment_embeddings)
    
    for index , speaker in enumerate(enrollment.speakers_list):    
        speaker.enrollment_embeddings = enrollment_embeddings[index]
    
    if np.array(enrollment.get_all_enrollment_files()).shape != np.array(enrollment_embeddings).shape[0:2]:
        raise Exception("The enrollment embeddings shape is not equal to the enrollment files shape \n the shape of the enrollment files is: {} \n the shape of the enrollment embeddings is: {}".format(np.array(enrollment.get_all_enrollment_files()).shape,np.array(enrollment_embeddings).shape))
     
    return enrollment

######################################################################################################################

def remove_enrollment_embeddings_from_embeddings_list(list_files_embeddings,embeddings,enrollment):
    '''
    This function gets a list of enrollment data and remove a list of the enrollment embeddings
    '''
    indices = np.concatenate(enrollment.get_all_indexes_big_embeddings_list())
    embeddings = np.delete(embeddings,indices,axis = 0)
    list_files_embeddings = np.delete(list_files_embeddings,indices,axis = 0)
    return list_files_embeddings,embeddings

#######################################################################################################################

def CM_inferernce_time_embeddings(data,gender_model,spoof_model_male,spoof_model_female,device,thr_gender_RF,thr_spoof_male,thr_spoof_female,verbose = False):
    its_spoof = None
    its_male = None
    with torch.no_grad():
        if gender_model.predict_proba([data])[:,1] > thr_gender_RF: #need to normalize the data
            its_male = True
            if verbose:
                print(f"it's male in the record  {gender_model.predict_proba([data])[:,1]}")
            spoof_score = torch.sigmoid(spoof_model_male(torch.Tensor(data).unsqueeze(0)).cpu())
            spoof_score = spoof_score.item()
            if verbose:
                print(f"it's male spoof with score: {spoof_score}")
            if spoof_score <= thr_spoof_male:
                its_spoof = False
                if verbose:
                    print("it's not spoof")
            else:
                its_spoof = True
                if verbose:
                    print("it's spoof")
            
        else:
            its_male = False
            if verbose:
                print(f"it's female in the record  {gender_model.predict_proba([data])[:,1]}")
            spoof_model_female = spoof_model_female.to(device)
            test_output = spoof_model_female(torch.Tensor(data).unsqueeze(0).to(device))
            _ , spoof_score = spoof_model_female.loss(torch.Tensor(test_output).to(device),None) # fix this tomorrow!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            spoof_score = -1*spoof_score.item()
            if verbose:
                print(f"it's female spoof with prob: {spoof_score}")
            if spoof_score <= thr_spoof_female:
                its_spoof = False
                if verbose:
                    print("it's not spoof")
            else:
                its_spoof = True
                if verbose:
                    print("it's spoof")
                    
                    
       
        if its_spoof is None:
            raise Exception("its_spoof is None, need to check the inference function")
        
        if its_male is None:
            raise Exception("its_male is None, need to check the inference function")
        
        return its_male,its_spoof, spoof_score  

   ######################################################################################################################
def CM_inferernce_time_embeddings_know_gender(data,spoof_model,device,thr_spoof,gender,verbose = False):
    its_spoof = None
    spoof_score = None
    with torch.no_grad():
        if gender == 'male':
            spoof_score = torch.sigmoid(spoof_model(torch.Tensor(data).unsqueeze(0)).cpu())
            spoof_score = spoof_score.item()
        elif gender == 'female':
            output = spoof_model(torch.Tensor(data).unsqueeze(0).to(device))
            _ , spoof_score = spoof_model.loss(torch.Tensor(output).to(device),None)  # fix this tomorrow!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            spoof_score = torch.Tensor(spoof_score).cpu()
            spoof_score = -1*spoof_score.item()
        else:
            raise Exception("Problem in gender type!")
        if verbose:
            print(f"it's spoof with score: {spoof_score}")
        if spoof_score <= thr_spoof:
            its_spoof = False
            if verbose:
                print("it's not spoof")
        else:
            its_spoof = True
            if verbose:
                print("it's spoof")
        
        if its_spoof is None:
            raise Exception("its_spoof is None, need to check the inference function")
    
    return its_spoof, spoof_score
######################################################################################################################
def CM_inferernce_time_embeddings_without_gender(data,spoof_model,device,thr_spoof,verbose = False):
    its_spoof = None
    with torch.no_grad():
            spoof_score = torch.sigmoid(spoof_model(torch.Tensor(data).unsqueeze(0)).cpu())
            spoof_score = spoof_score.item()
            if verbose:
                print(f"it's male spoof with score: {spoof_score}")
            if spoof_score <= thr_spoof:
                its_spoof = False
                if verbose:
                    print("it's not spoof")
            else:
                its_spoof = True
                if verbose:
                    print("it's spoof")
            
            if its_spoof is None:
                raise Exception("its_spoof is None, need to check the inference function")
        
    return its_spoof, spoof_score

#########################################################################################################################   
    
    
def ASV_verification_ECAPA_TDNN(example_1,example_2,its_male_1,its_male_2,its_spoof_1,its_spoof_2,verification_model,verbose):
    if its_male_1 == its_male_2 and its_spoof_1 == its_spoof_2 == False:
        score, prediction = verification_model.verify_files(example_1['path'], example_2['path']) # Same Speaker
        if prediction == True:
            if verbose:
                print("it's the same speaker, the score is: {score}")
    else:
        prediction = None
        score = None
        if verbose:
            print("it's not the same speaker")
            
        if score is None:
            raise Exception("score is None, need to check the inference function")
        
        if prediction is None:
            raise Exception("prediction is None, need to check the inference function")
    
    return score, prediction

######################################################################################################################
######################################################################################################################

def inference_system_cm_asV(txt_file_path,csv_file_path, asv_score,thr_gender_RF,thr_spoof_male,thr_spoof_female,enrollment_male, enrollment_female, list_files, data_set, gender_model, spoof_model_male, spoof_model_female, asv_model, device, verbose = False, verbose_txt = False,verbose_csv = False):
    '''
    This function is recognize the speaker in the data set, when the speaker don't say who he is, this task is like recognize the speaker. (Recognition task).
    '''
    
    # Initialize the dictionary with empty lists of data
    results = {  
        "file_name": [],
        "its_male": [],
        "its_spoof": [],
        "spoof_score": [],
        "spoof_ground_truth": [],
        "max_cos_score": [],
        "dataset_speaker_id": [],
        "pred_speaker_id": [],
        "find_speaker_in_enroll": [],
        "more_than_one_speaker": []
    }
    
    if txt_file_path is None: # If the txt file path is None, then don't write to the txt file
        verbose_txt = False
        print("The output file path is None, so the results will not be saved in txt file")
    
    if csv_file_path is None: # If the csv file path is None, then don't write to the csv file
        verbose_csv = False
        print("The output file path is None, so the results will not be saved in csv file")

    if verbose_txt: # Open the txt file      
        txt_file = open(txt_file_path, 'w')
        
    if verbose_csv: # Open the csv file
        csv_file = open(csv_file_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(results.keys())
        
    path_names = pd.Series(np.concatenate(data_set.name.values)) # Get the path names of the data set

    for _, file_name in tqdm.tqdm(enumerate(list_files), total=len(list_files)): # Iterate over the list of the files
        try:
            index_time_embedding = path_names.str.contains(file_name) # Get the index of the file in the data set
            
            if index_time_embedding.size == 0 or index_time_embedding is None: # If the index is empty, then raise an error
                raise ValueError(f"File {file_name} in list files, is not present in in list paths.")

            if verbose: # Print the file name and the index
                print(f"File {file_name} in list files, is at {path_names[index_time_embedding]} in list paths.")
                
        except ValueError: # If the index is empty, then raise an error
            print(f"File {file_name} in list files, is not present in in list paths.")

        time_embedding = data_set.data[index_time_embedding].squeeze() # Get the time embedding of the file

        # this function calculate the CM inference of the time embedding 
        its_male, its_spoof, spoof_score = CM_inferernce_time_embeddings(time_embedding, gender_model, spoof_model_male, spoof_model_female, device,thr_gender_RF,thr_spoof_male,thr_spoof_female) # this function calculate

        if its_male is None: 
            raise Exception("its_male is None, need to check the inference function")

        if its_spoof is None:
            raise Exception("its_spoof is None, need to check the inference function")
        
        if spoof_score is None:
            raise Exception("spoof_score is None, need to check the inference function")
        
        # Append the results to the data dictionary
        results["file_name"].append(file_name)
        results["its_male"].append(its_male)
        results["its_spoof"].append(its_spoof)
        results["spoof_ground_truth"].append(bool(data_set.is_spoofed.values[index_time_embedding][0]))
        results["spoof_score"].append(spoof_score)
        
        #signal, fs = torchaudio.load("Data/data_tests_thesis/ASVspoof2019_LA_dev/wav/LA_D_1000265.wav")
        
        signal, fs = torchaudio.load(str(path_names.values[index_time_embedding][0])) # load the signal from the path
        
        embeddings = asv_model.encode_batch(signal)  # encode the signal
       
        embeddings = embeddings.cpu().numpy().squeeze(0) # convert the embeddings to numpy

        # initialize the helps variables
        find_speaker_in_enroll = False
        max_cos_score = -1
        first_speaker_id_match = None
        flag_find_first_speaker = False
        more_than_one_speaker = False
        
        if its_male == True: # if its male
            
            all_male_speakers_id = np.array(enrollment_male.get_all_speakers_id()) 

            for index_speaker,embedding_in_enroll_per_speaker in enumerate(np.array(enrollment_male.get_all_enrollment_embeddings())):
                for _,embedding_in_enroll in enumerate(embedding_in_enroll_per_speaker):
                    cos_score = cosine_similarity(embedding_in_enroll.squeeze().reshape(1, -1), embeddings.reshape(1, -1)) # calculate the cosine similarity between the embeddings

                    if cos_score.shape != (1, 1): # if the cosine similarity is not scalar, then raise an error
                        raise Exception("cos_score shape is not (1,1) so is not scalar!, need to check the inference function")
                    
                    if cos_score.item() > max_cos_score: # if the cosine similarity is bigger than the max cosine similarity, then update the max cosine similarity
                        max_cos_score = cos_score.item()
                        max_cos_index = index_speaker
                        
                        if not flag_find_first_speaker: # if it's the first speaker, then update the first speaker id
                            first_speaker_id_match = index_speaker
                            flag_find_first_speaker = True
                        if first_speaker_id_match != index_speaker: # if it's not the first speaker, then update the more than one speaker flag
                            more_than_one_speaker = True
                    
            if max_cos_score > asv_score: # if the max cosine similarity is bigger than the asv score, then update the find speaker in enroll flag
                find_speaker_in_enroll = True # update the find speaker in enroll flag
                # Append the results to the data dictionary
                results["max_cos_score"].append(max_cos_score)
                results["dataset_speaker_id"].append(data_set.speaker_id.values[index_time_embedding][0])
                results["pred_speaker_id"].append(all_male_speakers_id[max_cos_index])
                results["more_than_one_speaker"].append(more_than_one_speaker)
                results["find_speaker_in_enroll"].append(find_speaker_in_enroll)
                
        elif its_male == False: # if its female
            
            all_female_speakers_id = np.array(enrollment_female.get_all_speakers_id()) 
            
            for index_speaker,embedding_in_enroll_per_speaker in enumerate(np.array(enrollment_female.get_all_enrollment_embeddings())):
                for _,embedding_in_enroll in enumerate(embedding_in_enroll_per_speaker):
                    cos_score = cosine_similarity(embedding_in_enroll.squeeze().reshape(1, -1), embeddings.reshape(1, -1)) # calculate the cosine similarity between the embeddings
                    
                    if cos_score.shape != (1, 1): # if the cosine similarity is not scalar, then raise an error
                        raise Exception("cos_score shape is not (1,1) so is not scalar!, need to check the inference function")
                    
                    if cos_score.item() > max_cos_score: # if the cosine similarity is bigger than the max cosine similarity, then update the max cosine similarity
                        max_cos_score = cos_score.item()
                        max_cos_index = index_speaker
                        if not flag_find_first_speaker: # if it's the first speaker, then update the first speaker id
                            first_speaker_id_match = index_speaker
                            flag_find_first_speaker = True
                        if first_speaker_id_match != index_speaker: # if it's not the first speaker, then update the more than one speaker flag
                            more_than_one_speaker = True
                    
            if max_cos_score > asv_score: # if the max cosine similarity is bigger than the asv score, then update the find speaker in enroll flag
                find_speaker_in_enroll = True
                results["max_cos_score"].append(max_cos_score)
                results["dataset_speaker_id"].append(data_set.speaker_id.values[index_time_embedding][0])
                results["pred_speaker_id"].append(all_female_speakers_id[max_cos_index])
                results["more_than_one_speaker"].append(more_than_one_speaker)
                results["find_speaker_in_enroll"].append(find_speaker_in_enroll)
                
            
        if not find_speaker_in_enroll: # if the find speaker in enroll flag is false, then update the results with unknown
            results["max_cos_score"].append("unknown")
            results["dataset_speaker_id"].append(data_set.speaker_id.values[index_time_embedding][0])
            results["pred_speaker_id"].append("unknown")
            results["more_than_one_speaker"].append("unknown")
            results["find_speaker_in_enroll"].append(find_speaker_in_enroll)
            
        
        if verbose: # Print the results
            print(f"file_name: {results['file_name'][-1]}, its_male: {results['its_male'][-1]}, its_spoof: {results['its_spoof'][-1]}, spoof_score: {results['spoof_score'][-1]}, spoof_ground_truth: {results['spoof_ground_truth'][-1]},max_cos_score: {results['max_cos_score'][-1]}, dataset_speaker_id: {results['dataset_speaker_id'][-1]} , dataset_speaker_id: {results['dataset_speaker_id'][-1]} , pred_speaker_id: {results['pred_speaker_id'][-1]}, more_than_one_speaker: {results['more_than_one_speaker'][-1]}, find_speaker_in_enroll: {results['find_speaker_in_enroll'][-1]} \n")
            
        if verbose_txt: # Write the results to the txt file
            txt_file.write(f"file_name: {results['file_name'][-1]}, its_male: {results['its_male'][-1]}, its_spoof: {results['its_spoof'][-1]}, spoof_score: {results['spoof_score'][-1]}, spoof_ground_truth: {results['spoof_ground_truth'][-1]},max_cos_score: {results['max_cos_score'][-1]}, dataset_speaker_id: {results['dataset_speaker_id'][-1]} , dataset_speaker_id: {results['dataset_speaker_id'][-1]} , pred_speaker_id: {results['pred_speaker_id'][-1]}, more_than_one_speaker: {results['more_than_one_speaker'][-1]}, find_speaker_in_enroll: {results['find_speaker_in_enroll'][-1]} \n")

        if verbose_csv: # Write the results to the csv file
            last_item = {key: list(values)[-1] for key, values in results.items()} # Get the last item of each list in the dictionary
            csv_writer.writerow(last_item.values()) # Write the last item to the csv file
        
    if verbose_txt: # Close the txt file
        txt_file.close()
        
    if verbose_csv:# Close the csv file
        csv_file.close() 
           
    results_df = pd.DataFrame(results)  # Return the DataFrame
    
    return results_df