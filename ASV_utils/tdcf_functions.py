import numpy as np
import utils.eval_metrics as eval_metrics
# parameters:

# Example Input Data:

#for male:

def compute_t_dcf(bonafide_score_cm,spoof_score_cm,Prior_spoof,target_scores,nontarget_scores,spoof_scores,list_asv_score,type):

    if type != 'constrained' and type != 'unconstrained' and type != 'constrained_ver2':
        raise ValueError('type must be either "constrained" or "unconstrained" or "constrained_ver2"')
    
    # Parameters of the t-DCF cost model
    if type == 'constrained':
        cost_model = {
            'Ptar':(1-Prior_spoof)*0.99,         # Prior probability of target speaker
            'Pnon':(1-Prior_spoof)*0.01,         # Prior probability of nontarget speaker (zero-effort impostor)
            'Pspoof': Prior_spoof,        # Prior probability of spoofing attack
            'Cmiss_asv': 1,      # Cost of ASV falsely rejecting target
            'Cfa_asv': 10,       # Cost of ASV falsely accepting nontarget
            'Cmiss_cm': 1,       # Cost of CM falsely rejecting target
            'Cfa_cm': 10          # Cost of CM falsely accepting spoof
        }
    elif type == 'unconstrained':
        cost_model = {
            'Ptar':(1-Prior_spoof)*0.99,         # Prior probability of target speaker
            'Pnon':(1-Prior_spoof)*0.01,         # Prior probability of nontarget speaker (zero-effort impostor)
            'Pspoof': Prior_spoof,              # Prior probability of spoofing attack
            'Cmiss': 1,
            'Cfa': 10,                          # Cost of ASV falsely accepting nontarget
            'Cfa_spoof':10,
        }
    elif type == 'constrained_ver2':
        cost_model = {
            'Ptar':(1-Prior_spoof)*0.99,         # Prior probability of target speaker
            'Pnon':(1-Prior_spoof)*0.01,         # Prior probability of nontarget speaker (zero-effort impostor)
            'Pspoof': Prior_spoof,        # Prior probability of spoofing attack
            'Cmiss': 1,  
            'Cfa': 10,       # Cost of ASV falsely accepting nontarget
            'Cfa_spoof':10,
        }
    list_tDCF_norm = []
    list_CM_thresholds = []
    list_tdcf = []
    for idx,asv_score in enumerate(list_asv_score):

        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = eval_metrics.obtain_asv_error_rates(target_scores,nontarget_scores,spoof_scores,asv_score)

        # Flag to print a summary of the cost parameters and the t-DCF cost function
        print_cost = True
        print("The t-DCF evaluation type is:",type)
        if type == 'constrained':
            tDCF_norm, CM_thresholds,tdcf = eval_metrics.compute_tDCF_constrained(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost)
        elif type == 'unconstrained':
            tDCF_norm, CM_thresholds,tdcf = eval_metrics.compute_tDCF_Unconstrained(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,Pfa_spoof_asv, cost_model, print_cost)
        elif type == 'constrained_ver2':
            tDCF_norm, CM_thresholds,tdcf = eval_metrics.compute_tDCF_constrained_ver2(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,Pfa_spoof_asv, cost_model, print_cost)
        
        list_tdcf.append(tdcf)
        list_tDCF_norm.append(tDCF_norm)
        list_CM_thresholds.append(CM_thresholds)

        if idx ==0:
            print("the asv threshold is from eer on asv development set ",asv_score)
        
        if idx ==1:
            print("the asv threshold is from eer on asv evaluation set ",asv_score)
        elif idx>1:
            print("the asv threshold is from eer on asv set ",asv_score)
        
        print("the CM thresholds is: ",CM_thresholds)

        print("the CM threshold min is:",CM_thresholds[np.argmin(tDCF_norm)])

        print("the tDCF_norm is:",tDCF_norm)

        print("the min tDCF_norm is:",min(tDCF_norm))
        
        print("the tDCF is:",tdcf)
        
        print("the min tdcf is:",min(tdcf))
        
    return list_tDCF_norm,list_CM_thresholds,list_tdcf