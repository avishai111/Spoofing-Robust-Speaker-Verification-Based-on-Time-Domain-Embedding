import os
import scipy
import pandas as pd
import scipy.io
import numpy as np

#loading the female data
def load_data_all(data_path,include_eval=True,include_dev = True):
    embedded_groups_1_1 = scipy.io.loadmat(os.path.join(data_path,'all_embedded_groups_1_1.mat'))['embedded_groups_1_1'];
    
    embedded_groups_1_2 = np.array([])
    if include_dev:
        embedded_groups_1_2 = scipy.io.loadmat(os.path.join(data_path,'all_embedded_groups_2_1.mat'))['embedded_groups_2_1'];
    
    embedded_groups_1_3 = np.array([])
    if include_eval:
        embedded_groups_1_3 = scipy.io.loadmat(os.path.join(data_path,'all_embedded_groups_3_1.mat'))['embedded_groups_3_1'];

    chosen_labels_1_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_is_spoofed.mat'))['chosen_labels_1_1_is_spoofed'];
    chosen_labels_1_1_is_spoofed = pd.Series([item for sublist in chosen_labels_1_1_is_spoofed for item in sublist]);

    chosen_labels_2_1_is_spoofed = pd.Series([])
    if include_dev:
        chosen_labels_2_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_is_spoofed.mat'))['chosen_labels_2_1_is_spoofed'];
        chosen_labels_2_1_is_spoofed = pd.Series([item for sublist in chosen_labels_2_1_is_spoofed for item in sublist]);

    chosen_labels_3_1_is_spoofed = pd.Series([])
    if include_eval:
        chosen_labels_3_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_is_spoofed.mat'))['chosen_labels_3_1_is_spoofed'];
        chosen_labels_3_1_is_spoofed = pd.Series([item for sublist in chosen_labels_3_1_is_spoofed for item in sublist]);

    chosen_labels_numeric_1_1 = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_numeric_1_1.mat'))['chosen_labels_numeric_1_1'][:];
    chosen_labels_numeric_1_1 = pd.Series([item for sublist in chosen_labels_numeric_1_1 for item in sublist]);

    chosen_labels_numeric_2_1 = pd.Series([])
    if include_dev:
        chosen_labels_numeric_2_1 = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_numeric_2_1.mat'))['chosen_labels_numeric_2_1'][:];
        chosen_labels_numeric_2_1 = pd.Series([item for sublist in chosen_labels_numeric_2_1 for item in sublist]);
    
    chosen_labels_numeric_3_1 = pd.Series([])
    if include_eval:
        chosen_labels_numeric_3_1 = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_numeric_3_1.mat'))['chosen_labels_numeric_3_1'][:];
        chosen_labels_numeric_3_1 = pd.Series([item for sublist in chosen_labels_numeric_3_1 for item in sublist]);

    chosen_labels_1_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_attack_logical.mat'))['chosen_labels_1_1_attack_logical'];
    chosen_labels_1_1_attack_logical = pd.Series([item for sublist in chosen_labels_1_1_attack_logical for item in sublist])

    chosen_labels_2_1_attack_logical = pd.Series([])
    if include_dev:
        chosen_labels_2_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_attack_logical.mat'))['chosen_labels_2_1_attack_logical'];
        chosen_labels_2_1_attack_logical = pd.Series([item for sublist in chosen_labels_2_1_attack_logical for item in sublist])

    chosen_labels_3_1_attack_logical = pd.Series([])
    if include_eval:
        chosen_labels_3_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_attack_logical.mat'))['chosen_labels_3_1_attack_logical'];
        chosen_labels_3_1_attack_logical = pd.Series([item for sublist in chosen_labels_3_1_attack_logical for item in sublist])

    chosen_labels_1_1_name = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_name.mat'))['chosen_labels_1_1_name'];
    chosen_labels_1_1_name = pd.Series([item for sublist in chosen_labels_1_1_name for item in sublist])

    chosen_labels_2_1_name = pd.Series([])
    if include_dev:
        chosen_labels_2_1_name = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_name.mat'))['chosen_labels_2_1_name'];
        chosen_labels_2_1_name = pd.Series([item for sublist in chosen_labels_2_1_name for item in sublist])
    
    chosen_labels_3_1_name = pd.Series([])
    if include_eval:
        chosen_labels_3_1_name = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_name.mat'))['chosen_labels_3_1_name'];
        chosen_labels_3_1_name = pd.Series([item for sublist in chosen_labels_3_1_name for item in sublist])

    chosen_labels_1_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_speaker_id.mat'))['chosen_labels_1_1_speaker_id'];
    chosen_labels_1_1_speaker_id = pd.Series([item for sublist in chosen_labels_1_1_speaker_id for item in sublist])

    chosen_labels_2_1_speaker_id = pd.Series([])
    if include_dev:
        chosen_labels_2_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_speaker_id.mat'))['chosen_labels_2_1_speaker_id'];
        chosen_labels_2_1_speaker_id = pd.Series([item for sublist in chosen_labels_2_1_speaker_id for item in sublist])
    
    chosen_labels_3_1_speaker_id = pd.Series([])
    if include_eval:
        chosen_labels_3_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_speaker_id.mat'))['chosen_labels_3_1_speaker_id'];
        chosen_labels_3_1_speaker_id = pd.Series([item for sublist in chosen_labels_3_1_speaker_id for item in sublist])

    chosen_labels_1_1_sex = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_sex.mat'))['chosen_labels_1_1_sex'];
    chosen_labels_1_1_sex = pd.Series([item for sublist in chosen_labels_1_1_sex for item in sublist]);

    chosen_labels_2_1_sex = pd.Series([])
    if include_dev:
        chosen_labels_2_1_sex = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_sex.mat'))['chosen_labels_2_1_sex'];
        chosen_labels_2_1_sex = pd.Series([item for sublist in chosen_labels_2_1_sex for item in sublist]);
    
    chosen_labels_3_1_sex = pd.Series([])
    if include_eval:
        chosen_labels_3_1_sex = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_sex.mat'))['chosen_labels_3_1_sex'];
        chosen_labels_3_1_sex = pd.Series([item for sublist in chosen_labels_3_1_sex for item in sublist]);
    
    
    return  embedded_groups_1_1,embedded_groups_1_2,embedded_groups_1_3,chosen_labels_1_1_is_spoofed,chosen_labels_2_1_is_spoofed,chosen_labels_3_1_is_spoofed,chosen_labels_numeric_1_1,chosen_labels_numeric_2_1,chosen_labels_numeric_3_1,chosen_labels_1_1_attack_logical,chosen_labels_2_1_attack_logical,chosen_labels_3_1_attack_logical,chosen_labels_1_1_name,chosen_labels_2_1_name,chosen_labels_3_1_name,chosen_labels_1_1_speaker_id,chosen_labels_2_1_speaker_id,chosen_labels_3_1_speaker_id,chosen_labels_1_1_sex,chosen_labels_2_1_sex,chosen_labels_3_1_sex   



#loading the male data
def load_data_male(data_path):
    embedded_groups_1_1 = scipy.io.loadmat(os.path.join(data_path,'male_embedded_groups_1_1.mat'))['embedded_groups_1_1'];
    embedded_groups_1_2 = scipy.io.loadmat(os.path.join(data_path,'male_embedded_groups_2_1.mat'))['embedded_groups_2_1'];
    embedded_groups_1_3 = scipy.io.loadmat(os.path.join(data_path,'male_embedded_groups_3_1.mat'))['embedded_groups_3_1'];

    chosen_labels_1_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_1_1_is_spoofed.mat'))['chosen_labels_1_1_is_spoofed'];
    chosen_labels_1_1_is_spoofed = pd.Series([item for sublist in chosen_labels_1_1_is_spoofed for item in sublist]);

    chosen_labels_2_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_2_1_is_spoofed.mat'))['chosen_labels_2_1_is_spoofed'];
    chosen_labels_2_1_is_spoofed = pd.Series([item for sublist in chosen_labels_2_1_is_spoofed for item in sublist]);

    chosen_labels_3_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_3_1_is_spoofed.mat'))['chosen_labels_3_1_is_spoofed'];
    chosen_labels_3_1_is_spoofed = pd.Series([item for sublist in chosen_labels_3_1_is_spoofed for item in sublist]);

    chosen_labels_numeric_1_1 = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_numeric_1_1.mat'))['chosen_labels_numeric_1_1'][:];
    chosen_labels_numeric_1_1 = pd.Series([item for sublist in chosen_labels_numeric_1_1 for item in sublist]);

    chosen_labels_numeric_2_1 = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_numeric_2_1.mat'))['chosen_labels_numeric_2_1'][:];
    chosen_labels_numeric_2_1 = pd.Series([item for sublist in chosen_labels_numeric_2_1 for item in sublist]);

    chosen_labels_numeric_3_1 = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_numeric_3_1.mat'))['chosen_labels_numeric_3_1'][:];
    chosen_labels_numeric_3_1 = pd.Series([item for sublist in chosen_labels_numeric_3_1 for item in sublist]);

    chosen_labels_1_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_1_1_attack_logical.mat'))['chosen_labels_1_1_attack_logical'];
    chosen_labels_1_1_attack_logical = pd.Series([item for sublist in chosen_labels_1_1_attack_logical for item in sublist])

    chosen_labels_2_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_2_1_attack_logical.mat'))['chosen_labels_2_1_attack_logical'];
    chosen_labels_2_1_attack_logical = pd.Series([item for sublist in chosen_labels_2_1_attack_logical for item in sublist])

    chosen_labels_3_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_3_1_attack_logical.mat'))['chosen_labels_3_1_attack_logical'];
    chosen_labels_3_1_attack_logical = pd.Series([item for sublist in chosen_labels_3_1_attack_logical for item in sublist])

    chosen_labels_1_1_name = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_1_1_name.mat'))['chosen_labels_1_1_name'];
    chosen_labels_1_1_name = pd.Series([item for sublist in chosen_labels_1_1_name for item in sublist])

    chosen_labels_2_1_name = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_2_1_name.mat'))['chosen_labels_2_1_name'];
    chosen_labels_2_1_name = pd.Series([item for sublist in chosen_labels_2_1_name for item in sublist])

    chosen_labels_3_1_name = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_3_1_name.mat'))['chosen_labels_3_1_name'];
    chosen_labels_3_1_name = pd.Series([item for sublist in chosen_labels_3_1_name for item in sublist])

    chosen_labels_1_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_1_1_speaker_id.mat'))['chosen_labels_1_1_speaker_id'];
    chosen_labels_1_1_speaker_id = pd.Series([item for sublist in chosen_labels_1_1_speaker_id for item in sublist])

    chosen_labels_2_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_2_1_speaker_id.mat'))['chosen_labels_2_1_speaker_id'];
    chosen_labels_2_1_speaker_id = pd.Series([item for sublist in chosen_labels_2_1_speaker_id for item in sublist])

    chosen_labels_3_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_3_1_speaker_id.mat'))['chosen_labels_3_1_speaker_id'];
    chosen_labels_3_1_speaker_id = pd.Series([item for sublist in chosen_labels_3_1_speaker_id for item in sublist])

    male_chosen_labels_1_1_sex = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_1_1_sex.mat'))['chosen_labels_1_1_sex'];
    male_chosen_labels_1_1_sex = pd.Series([item for sublist in male_chosen_labels_1_1_sex for item in sublist]);

    male_chosen_labels_2_1_sex = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_2_1_sex.mat'))['chosen_labels_2_1_sex'];
    male_chosen_labels_2_1_sex = pd.Series([item for sublist in male_chosen_labels_2_1_sex for item in sublist]);

    male_chosen_labels_3_1_sex = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_3_1_sex.mat'))['chosen_labels_3_1_sex'];
    male_chosen_labels_3_1_sex = pd.Series([item for sublist in male_chosen_labels_3_1_sex for item in sublist]);
    
    return  embedded_groups_1_1,embedded_groups_1_2,embedded_groups_1_3,chosen_labels_1_1_is_spoofed,chosen_labels_2_1_is_spoofed,chosen_labels_3_1_is_spoofed,chosen_labels_numeric_1_1,chosen_labels_numeric_2_1,chosen_labels_numeric_3_1,chosen_labels_1_1_attack_logical,chosen_labels_2_1_attack_logical,chosen_labels_3_1_attack_logical,chosen_labels_1_1_name,chosen_labels_2_1_name,chosen_labels_3_1_name,chosen_labels_1_1_speaker_id,chosen_labels_2_1_speaker_id,chosen_labels_3_1_speaker_id,male_chosen_labels_1_1_sex,male_chosen_labels_2_1_sex,male_chosen_labels_3_1_sex   

#loading the female data
def load_data_female(data_path):
    embedded_groups_1_1 = scipy.io.loadmat(os.path.join(data_path,'female_embedded_groups_1_1.mat'))['embedded_groups_1_1'];
    embedded_groups_1_2 = scipy.io.loadmat(os.path.join(data_path,'female_embedded_groups_2_1.mat'))['embedded_groups_2_1'];
    embedded_groups_1_3 = scipy.io.loadmat(os.path.join(data_path,'female_embedded_groups_3_1.mat'))['embedded_groups_3_1'];

    chosen_labels_1_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_1_1_is_spoofed.mat'))['chosen_labels_1_1_is_spoofed'];
    chosen_labels_1_1_is_spoofed = pd.Series([item for sublist in chosen_labels_1_1_is_spoofed for item in sublist]);

    chosen_labels_2_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_2_1_is_spoofed.mat'))['chosen_labels_2_1_is_spoofed'];
    chosen_labels_2_1_is_spoofed = pd.Series([item for sublist in chosen_labels_2_1_is_spoofed for item in sublist]);

    chosen_labels_3_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_3_1_is_spoofed.mat'))['chosen_labels_3_1_is_spoofed'];
    chosen_labels_3_1_is_spoofed = pd.Series([item for sublist in chosen_labels_3_1_is_spoofed for item in sublist]);

    chosen_labels_numeric_1_1 = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_numeric_1_1.mat'))['chosen_labels_numeric_1_1'][:];
    chosen_labels_numeric_1_1 = pd.Series([item for sublist in chosen_labels_numeric_1_1 for item in sublist]);

    chosen_labels_numeric_2_1 = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_numeric_2_1.mat'))['chosen_labels_numeric_2_1'][:];
    chosen_labels_numeric_2_1 = pd.Series([item for sublist in chosen_labels_numeric_2_1 for item in sublist]);

    chosen_labels_numeric_3_1 = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_numeric_3_1.mat'))['chosen_labels_numeric_3_1'][:];
    chosen_labels_numeric_3_1 = pd.Series([item for sublist in chosen_labels_numeric_3_1 for item in sublist]);

    chosen_labels_1_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_1_1_attack_logical.mat'))['chosen_labels_1_1_attack_logical'];
    chosen_labels_1_1_attack_logical = pd.Series([item for sublist in chosen_labels_1_1_attack_logical for item in sublist])

    chosen_labels_2_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_2_1_attack_logical.mat'))['chosen_labels_2_1_attack_logical'];
    chosen_labels_2_1_attack_logical = pd.Series([item for sublist in chosen_labels_2_1_attack_logical for item in sublist])

    chosen_labels_3_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_3_1_attack_logical.mat'))['chosen_labels_3_1_attack_logical'];
    chosen_labels_3_1_attack_logical = pd.Series([item for sublist in chosen_labels_3_1_attack_logical for item in sublist])

    chosen_labels_1_1_name = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_1_1_name.mat'))['chosen_labels_1_1_name'];
    chosen_labels_1_1_name = pd.Series([item for sublist in chosen_labels_1_1_name for item in sublist])

    chosen_labels_2_1_name = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_2_1_name.mat'))['chosen_labels_2_1_name'];
    chosen_labels_2_1_name = pd.Series([item for sublist in chosen_labels_2_1_name for item in sublist])

    chosen_labels_3_1_name = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_3_1_name.mat'))['chosen_labels_3_1_name'];
    chosen_labels_3_1_name = pd.Series([item for sublist in chosen_labels_3_1_name for item in sublist])

    chosen_labels_1_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_1_1_speaker_id.mat'))['chosen_labels_1_1_speaker_id'];
    chosen_labels_1_1_speaker_id = pd.Series([item for sublist in chosen_labels_1_1_speaker_id for item in sublist])

    chosen_labels_2_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_2_1_speaker_id.mat'))['chosen_labels_2_1_speaker_id'];
    chosen_labels_2_1_speaker_id = pd.Series([item for sublist in chosen_labels_2_1_speaker_id for item in sublist])

    chosen_labels_3_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_3_1_speaker_id.mat'))['chosen_labels_3_1_speaker_id'];
    chosen_labels_3_1_speaker_id = pd.Series([item for sublist in chosen_labels_3_1_speaker_id for item in sublist])

    female_chosen_labels_1_1_sex = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_1_1_sex.mat'))['chosen_labels_1_1_sex'];
    female_chosen_labels_1_1_sex = pd.Series([item for sublist in female_chosen_labels_1_1_sex for item in sublist]);

    female_chosen_labels_2_1_sex = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_2_1_sex.mat'))['chosen_labels_2_1_sex'];
    female_chosen_labels_2_1_sex = pd.Series([item for sublist in female_chosen_labels_2_1_sex for item in sublist]);

    female_chosen_labels_3_1_sex = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_3_1_sex.mat'))['chosen_labels_3_1_sex'];
    female_chosen_labels_3_1_sex = pd.Series([item for sublist in female_chosen_labels_3_1_sex for item in sublist]);
    
    return  embedded_groups_1_1,embedded_groups_1_2,embedded_groups_1_3,chosen_labels_1_1_is_spoofed,chosen_labels_2_1_is_spoofed,chosen_labels_3_1_is_spoofed,chosen_labels_numeric_1_1,chosen_labels_numeric_2_1,chosen_labels_numeric_3_1,chosen_labels_1_1_attack_logical,chosen_labels_2_1_attack_logical,chosen_labels_3_1_attack_logical,chosen_labels_1_1_name,chosen_labels_2_1_name,chosen_labels_3_1_name,chosen_labels_1_1_speaker_id,chosen_labels_2_1_speaker_id,chosen_labels_3_1_speaker_id,female_chosen_labels_1_1_sex,female_chosen_labels_2_1_sex,female_chosen_labels_3_1_sex   




#concatenate the data of male and female to one data
def concatenate_data(male_embedded_groups_1_1,male_embedded_groups_1_2,male_embedded_groups_1_3,
                male_chosen_labels_1_1_is_spoofed,male_chosen_labels_2_1_is_spoofed,male_chosen_labels_3_1_is_spoofed,
                male_chosen_labels_numeric_1_1,male_chosen_labels_numeric_2_1,male_chosen_labels_numeric_3_1,
                male_chosen_labels_1_1_attack_logical,male_chosen_labels_2_1_attack_logical,male_chosen_labels_3_1_attack_logical,
                male_chosen_labels_1_1_name,male_chosen_labels_2_1_name,male_chosen_labels_3_1_name,
                male_chosen_labels_1_1_speaker_id,male_chosen_labels_2_1_speaker_id, male_chosen_labels_3_1_speaker_id,
                male_chosen_labels_1_1_sex,male_chosen_labels_2_1_sex,male_chosen_labels_3_1_sex,
                
                female_embedded_groups_1_1,female_embedded_groups_1_2,female_embedded_groups_1_3,
                female_chosen_labels_1_1_is_spoofed,female_chosen_labels_2_1_is_spoofed,female_chosen_labels_3_1_is_spoofed,
                female_chosen_labels_numeric_1_1,female_chosen_labels_numeric_2_1,female_chosen_labels_numeric_3_1,
                female_chosen_labels_1_1_attack_logical,female_chosen_labels_2_1_attack_logical,female_chosen_labels_3_1_attack_logical,
                female_chosen_labels_1_1_name,female_chosen_labels_2_1_name,female_chosen_labels_3_1_name,
                female_chosen_labels_1_1_speaker_id,female_chosen_labels_2_1_speaker_id,female_chosen_labels_3_1_speaker_id,
                female_chosen_labels_1_1_sex,female_chosen_labels_2_1_sex,female_chosen_labels_3_1_sex):


    embedded_groups_1_1 = np.concatenate((male_embedded_groups_1_1,female_embedded_groups_1_1),axis = 0); 

    embedded_groups_1_2 = np.concatenate((male_embedded_groups_1_2,female_embedded_groups_1_2),axis = 0);

    embedded_groups_1_3 = np.concatenate((male_embedded_groups_1_3,female_embedded_groups_1_3),axis = 0);

    chosen_labels_numeric_1_1 = pd.concat([male_chosen_labels_numeric_1_1,female_chosen_labels_numeric_1_1], ignore_index = True,axis = 0);
    chosen_labels_numeric_2_1 = pd.concat([male_chosen_labels_numeric_2_1,female_chosen_labels_numeric_2_1], ignore_index = True,axis = 0);
    chosen_labels_numeric_3_1 = pd.concat([male_chosen_labels_numeric_3_1,female_chosen_labels_numeric_3_1], ignore_index = True, axis = 0);
    
    chosen_labels_1_1_attack_logical = pd.concat([male_chosen_labels_1_1_attack_logical,female_chosen_labels_1_1_attack_logical], ignore_index = True,axis = 0);
    chosen_labels_1_1_is_spoofed = pd.concat([male_chosen_labels_1_1_is_spoofed,female_chosen_labels_1_1_is_spoofed], ignore_index = True,axis = 0);
    chosen_labels_1_1_name = pd.concat([male_chosen_labels_1_1_name,female_chosen_labels_1_1_name], ignore_index = True, axis = 0);
    chosen_labels_1_1_sex = pd.concat([male_chosen_labels_1_1_sex,female_chosen_labels_1_1_sex], ignore_index = True,axis = 0);
    chosen_labels_1_1_speaker_id = pd.concat([male_chosen_labels_1_1_speaker_id,female_chosen_labels_1_1_speaker_id], ignore_index = True, axis = 0);

    chosen_labels_2_1_attack_logical = pd.concat([male_chosen_labels_2_1_attack_logical,female_chosen_labels_2_1_attack_logical], ignore_index = True,axis = 0);
    chosen_labels_2_1_is_spoofed = pd.concat([male_chosen_labels_2_1_is_spoofed,female_chosen_labels_2_1_is_spoofed], ignore_index = True,axis = 0);
    chosen_labels_2_1_name = pd.concat([male_chosen_labels_2_1_name,female_chosen_labels_2_1_name], ignore_index = True, axis = 0);
    chosen_labels_2_1_sex = pd.concat([male_chosen_labels_2_1_sex,female_chosen_labels_2_1_sex], ignore_index = True,axis = 0);
    chosen_labels_2_1_speaker_id = pd.concat([male_chosen_labels_2_1_speaker_id,female_chosen_labels_2_1_speaker_id], ignore_index = True, axis = 0);

    chosen_labels_3_1_attack_logical = pd.concat([male_chosen_labels_3_1_attack_logical,female_chosen_labels_3_1_attack_logical], ignore_index = True,axis = 0);
    chosen_labels_3_1_is_spoofed = pd.concat([male_chosen_labels_3_1_is_spoofed,female_chosen_labels_3_1_is_spoofed], ignore_index = True,axis = 0);
    chosen_labels_3_1_name = pd.concat([male_chosen_labels_3_1_name,female_chosen_labels_3_1_name], ignore_index = True, axis = 0);
    chosen_labels_3_1_sex = pd.concat([male_chosen_labels_3_1_sex,female_chosen_labels_3_1_sex], ignore_index = True,axis = 0);
    chosen_labels_3_1_speaker_id = pd.concat([male_chosen_labels_3_1_speaker_id,female_chosen_labels_3_1_speaker_id], ignore_index = True, axis = 0);
    
    return  embedded_groups_1_1,embedded_groups_1_2,embedded_groups_1_3,chosen_labels_1_1_is_spoofed,chosen_labels_2_1_is_spoofed,chosen_labels_3_1_is_spoofed,chosen_labels_numeric_1_1,chosen_labels_numeric_2_1,chosen_labels_numeric_3_1,chosen_labels_1_1_attack_logical,chosen_labels_2_1_attack_logical,chosen_labels_3_1_attack_logical,chosen_labels_1_1_name,chosen_labels_2_1_name,chosen_labels_3_1_name,chosen_labels_1_1_speaker_id,chosen_labels_2_1_speaker_id,chosen_labels_3_1_speaker_id,chosen_labels_1_1_sex,chosen_labels_2_1_sex,chosen_labels_3_1_sex   




#concatenate the data of male and female to one data
def concatenate_training_data(embedded_groups_1_1,
                chosen_labels_1_1_is_spoofed,
                chosen_labels_1_1_attack_logical,
                chosen_labels_1_1_name,
                chosen_labels_1_1_speaker_id,
                chosen_labels_1_1_sex,
                embedded_groups_1_2,
                chosen_labels_1_2_is_spoofed,
                chosen_labels_1_2_attack_logical,
                chosen_labels_1_2_name,
                chosen_labels_1_2_speaker_id,
                chosen_labels_1_2_sex):


    embedded_groups_1_1_all = np.concatenate((embedded_groups_1_1,embedded_groups_1_2),axis = 0); 

    is_spoofed_all = pd.concat([chosen_labels_1_1_is_spoofed,chosen_labels_1_2_is_spoofed], ignore_index = True,axis = 0);

    
    chosen_labels_attack_logical_all = pd.concat([chosen_labels_1_1_attack_logical,chosen_labels_1_2_attack_logical], ignore_index = True,axis = 0);
    chosen_labels_name_all = pd.concat([chosen_labels_1_1_name,chosen_labels_1_2_name], ignore_index = True, axis = 0);
    
    chosen_labels_speaker_id_all = pd.concat([chosen_labels_1_1_speaker_id,chosen_labels_1_2_speaker_id], ignore_index = True, axis = 0);
    
    chosen_labels_sex_all = pd.concat([chosen_labels_1_1_sex,chosen_labels_1_2_sex], ignore_index = True, axis = 0);

    return embedded_groups_1_1_all, is_spoofed_all, chosen_labels_attack_logical_all, chosen_labels_name_all, chosen_labels_speaker_id_all, chosen_labels_sex_all


def split_to_male_and_female(elements,chosen_labels_sex):
    if isinstance(chosen_labels_sex[0], list):
        chosen_labels_sex = pd.Series([item[0] for item in chosen_labels_sex])
    male_elements = elements[chosen_labels_sex == 'male']
    female_elements = elements[chosen_labels_sex == 'female']
    return male_elements, female_elements 


def split_all_elements_to_male_and_female(embedded_groups_1_1, chosen_labels_1_1_sex, chosen_labels_1_1_is_spoofed, chosen_labels_1_1_attack_logical, chosen_labels_1_1_name, chosen_labels_1_1_speaker_id):
    
    male_embedded_groups_1_1, female_embedded_groups_1_1 = split_to_male_and_female(embedded_groups_1_1, chosen_labels_1_1_sex)
    male_is_spoofed_1_1, female_is_spoofed_1_1 = split_to_male_and_female(chosen_labels_1_1_is_spoofed, chosen_labels_1_1_sex)
    male_attack_logical_1_1 , female_attack_logical_1_1 = split_to_male_and_female(chosen_labels_1_1_attack_logical, chosen_labels_1_1_sex)
    male_name_1_1, female_name_1_1 = split_to_male_and_female(chosen_labels_1_1_name, chosen_labels_1_1_sex)
    male_speaker_id_1_1, female_speaker_id_1_1 = split_to_male_and_female(chosen_labels_1_1_speaker_id, chosen_labels_1_1_sex)
    male_labels_1_1_sex, female_labels_1_1_sex = split_to_male_and_female(chosen_labels_1_1_sex, chosen_labels_1_1_sex)
    
    return male_embedded_groups_1_1, female_embedded_groups_1_1, male_is_spoofed_1_1, female_is_spoofed_1_1, male_attack_logical_1_1, \
    female_attack_logical_1_1, male_name_1_1, female_name_1_1, male_speaker_id_1_1, female_speaker_id_1_1 , male_labels_1_1_sex, female_labels_1_1_sex

def loading_training_data_augmantion(data_path_1,data_path_2,data_path_3,data_path_4):
    #loading augmantion data:
    embedded_groups_1_1,_,_,is_spoofed_1_1,_,_,_,_,_,attack_logical_1_1,_,_,name_1_1,_,_,speaker_id_1_1,_,_,sex_1_1,_,_ = load_data_all(data_path_1, include_eval = False,include_dev = False)
    sex_1_1 = pd.Series([item[0] for item in sex_1_1])
    
    embedded_groups_1_2,_,_,is_spoofed_1_2,_,_,_,_,_,attack_logical_1_2,_,_,name_1_2,_,_,speaker_id_1_2,_,_,sex_1_2,_,_ = load_data_all(data_path_2, include_eval = False,include_dev = False)
    sex_1_2 = pd.Series([item[0] for item in sex_1_2])

    embedded_groups_1_3,_,_,is_spoofed_1_3,_,_,_,_,_,attack_logical_1_3,_,_,name_1_3,_,_,speaker_id_1_3,_,_,sex_1_3,_,_ = load_data_all(data_path_3, include_eval = False,include_dev = False)
    sex_1_3 = pd.Series([item[0] for item in sex_1_3])
    
    embedded_groups_1_4,_,_,is_spoofed_1_4,_,_,_,_,_,attack_logical_1_4,_,_,name_1_4,_,_,speaker_id_1_4,_,_,sex_1_4,_,_ = load_data_all(data_path_4, include_eval = False,include_dev = False)
    sex_1_4 = pd.Series([item[0] for item in sex_1_4])

    # splitting the data into male and female:

    male_embedded_groups_1_1, female_embedded_groups_1_1, male_is_spoofed_1_1, female_is_spoofed_1_1, male_attack_logical_1_1, \
    female_attack_logical_1_1, male_name_1_1, female_name_1_1, male_speaker_id_1_1, female_speaker_id_1_1 ,male_sex_1_1, female_sex_1_1 = split_all_elements_to_male_and_female(embedded_groups_1_1, sex_1_1, is_spoofed_1_1, attack_logical_1_1, name_1_1, speaker_id_1_1)

    male_embedded_groups_1_2, female_embedded_groups_1_2, male_is_spoofed_1_2, female_is_spoofed_1_2, male_attack_logical_1_2, \
    female_attack_logical_1_2, male_name_1_2, female_name_1_2, male_speaker_id_1_2, female_speaker_id_1_2 ,male_sex_1_2, female_sex_1_2 = split_all_elements_to_male_and_female(embedded_groups_1_2, sex_1_2, is_spoofed_1_2, attack_logical_1_2, name_1_2, speaker_id_1_2)

    male_embedded_groups_1_3, female_embedded_groups_1_3, male_is_spoofed_1_3, female_is_spoofed_1_3, male_attack_logical_1_3, \
    female_attack_logical_1_3, male_name_1_3, female_name_1_3, male_speaker_id_1_3, female_speaker_id_1_3  , male_sex_1_3, female_sex_1_3 = split_all_elements_to_male_and_female(embedded_groups_1_3, sex_1_3, is_spoofed_1_3, attack_logical_1_3, name_1_3, speaker_id_1_3)

    male_embedded_groups_1_4, female_embedded_groups_1_4, male_is_spoofed_1_4, female_is_spoofed_1_4, male_attack_logical_1_4, \
    female_attack_logical_1_4, male_name_1_4, female_name_1_4, male_speaker_id_1_4, female_speaker_id_1_4 , male_sex_1_4, female_sex_1_4 = split_all_elements_to_male_and_female(embedded_groups_1_4, sex_1_4, is_spoofed_1_4, attack_logical_1_4, name_1_4, speaker_id_1_4)

    #concatenate the male data:
    male_embedded_all_1, male_is_spoofed_all_1, male_chosen_labels_attack_logical_all_1, \
    male_chosen_labels_name_all_1, male_chosen_labels_speaker_id_all_1, male_chosen_labels_sex_all_1 = concatenate_training_data(male_embedded_groups_1_1,male_is_spoofed_1_1,male_attack_logical_1_1,male_name_1_1,male_speaker_id_1_1,male_sex_1_1, \
                                                                                                            male_embedded_groups_1_2, male_is_spoofed_1_2, male_attack_logical_1_2, male_name_1_2, male_speaker_id_1_2, male_sex_1_2)

    male_embedded_all_2, male_is_spoofed_all_2, male_chosen_labels_attack_logical_all_2, \
    male_chosen_labels_name_all_2, male_chosen_labels_speaker_id_all_2, male_chosen_labels_sex_all_2 = concatenate_training_data(male_embedded_groups_1_3,male_is_spoofed_1_3,male_attack_logical_1_3,male_name_1_3,male_speaker_id_1_3,male_sex_1_3, \
                                                                                                            male_embedded_groups_1_4, male_is_spoofed_1_4, male_attack_logical_1_4, male_name_1_4, male_speaker_id_1_4,male_sex_1_4)

    male_embedded_all, male_is_spoofed_all, male_chosen_labels_attack_logical_all, \
    male_chosen_labels_name_all, male_chosen_labels_speaker_id_all, male_chosen_labels_sex_all = concatenate_training_data(male_embedded_all_1,male_is_spoofed_all_1,male_chosen_labels_attack_logical_all_1,male_chosen_labels_name_all_1,male_chosen_labels_speaker_id_all_1,male_chosen_labels_sex_all_1, \
                                                                                                            male_embedded_all_2, male_is_spoofed_all_2, male_chosen_labels_attack_logical_all_2, male_chosen_labels_name_all_2, male_chosen_labels_speaker_id_all_2, male_chosen_labels_sex_all_2)

    #concatenate the female data:
    female_embedded_all_1, female_is_spoofed_all_1, female_chosen_labels_attack_logical_all_1, \
    female_chosen_labels_name_all_1, female_chosen_labels_speaker_id_all_1, female_chosen_labels_sex_all_1 = concatenate_training_data(female_embedded_groups_1_1,female_is_spoofed_1_1,female_attack_logical_1_1,female_name_1_1,female_speaker_id_1_1,female_sex_1_1, \
                                                                                                            female_embedded_groups_1_2, female_is_spoofed_1_2, female_attack_logical_1_2, female_name_1_2, female_speaker_id_1_2, female_sex_1_2)

    female_embedded_all_2, female_is_spoofed_all_2, female_chosen_labels_attack_logical_all_2, \
    female_chosen_labels_name_all_2, female_chosen_labels_speaker_id_all_2, female_chosen_labels_sex_all_2 = concatenate_training_data(female_embedded_groups_1_3,female_is_spoofed_1_3,female_attack_logical_1_3,female_name_1_3,female_speaker_id_1_3,female_sex_1_3, \
                                                                                                            female_embedded_groups_1_4, female_is_spoofed_1_4, female_attack_logical_1_4, female_name_1_4, female_speaker_id_1_4,female_sex_1_4)

    female_embedded_all, female_is_spoofed_all, female_chosen_labels_attack_logical_all, \
    female_chosen_labels_name_all, female_chosen_labels_speaker_id_all, female_chosen_labels_sex_all = concatenate_training_data(female_embedded_all_1,female_is_spoofed_all_1,female_chosen_labels_attack_logical_all_1,female_chosen_labels_name_all_1,female_chosen_labels_speaker_id_all_1,female_chosen_labels_sex_all_1, \
                                                                                                            female_embedded_all_2,female_is_spoofed_all_2,female_chosen_labels_attack_logical_all_2,female_chosen_labels_name_all_2,female_chosen_labels_speaker_id_all_2,female_chosen_labels_sex_all_2)
    
    
    return male_embedded_all,female_embedded_all, \
        male_is_spoofed_all,female_is_spoofed_all, \
        male_chosen_labels_attack_logical_all,female_chosen_labels_attack_logical_all, \
        male_chosen_labels_name_all,female_chosen_labels_name_all, \
        male_chosen_labels_speaker_id_all,female_chosen_labels_speaker_id_all, \
        male_chosen_labels_sex_all,female_chosen_labels_sex_all 
        
        
if __name__ == "__main__":
    data_path_1 = './Data/pmf_augmantion_ASVSpoof2019/1/'
    data_path_2 = './Data/pmf_augmantion_ASVSpoof2019/2/'
    data_path_3 = './Data/pmf_augmantion_ASVSpoof2019/3/'
    data_path_4 = './Data/pmf_augmantion_ASVSpoof2019/4/'
    male_embedded_all,female_embedded_all, \
        male_is_spoofed_all,female_is_spoofed_all, \
        male_chosen_labels_attack_logical_all,female_chosen_labels_attack_logical_all, \
        male_chosen_labels_name_all,female_chosen_labels_name_all, \
        male_chosen_labels_speaker_id_all,female_chosen_labels_speaker_id_all, \
        male_chosen_labels_sex_all,female_chosen_labels_sex_all  = loading_training_data_augmantion(data_path_1,data_path_2,data_path_3,data_path_4)