    
import torch
import pandas as pd
import numpy as np
from time import time
import utils.my_functions as my_functions
from sklearn.metrics import det_curve
from alive_progress import alive_bar
import sklearn.metrics
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.pytorchtools import EarlyStopping
import torch.nn.functional as F 
import utils.sphere_plots
import scipy
import random
import utils.AMSloss
from utils.pytorchtools import EarlyStopping
import torch.nn.functional as F 
import utils.sphere_plots
import matplotlib
import pickle
#########################################################################
def set_random_seeds(seed=42):
    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch (CPU and GPU, if available)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic flags for PyTorch (if available)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#################################################################################################################################################

#################################################################################################################################################
# get number of correct predictions
def number_of_correct(prediction, labels):
    # count number of correct predictions
    return prediction.eq(labels).sum().item()
#################################################################################################################################################


#################################################################################################################################################
# test the model on the test set
def test_model(model, test_dataloader ,model_name = "TestSet DNN"):
    '''
    Args:
        model: model to test
        test_dataloader: test dataloader
        model_name: model name
    Returns:
        balanced_acc_percentage: balanced accuracy in percentage
        f1_score: f1 score
        matrix_confusion: confusion matrix
        total_prob: total probability
        total_labels: total labels
    '''
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    model.eval()
    balanced_acc_batch = [] # balanced accuracy per batch
    f1_score_batch = [] # f1 score per batch
    matrix_confusion = 0 # confusion matrix
    total_prob = [] # total probability
    total_labels = [] # total labels
    
    with alive_bar(total=len(test_dataloader)) as pbar_test:
        for _, (data_iter, labels_iter) in enumerate(test_dataloader): # for each batch
            data_iter = data_iter.to(device) # send data to device
            labels_iter = labels_iter.to(device) # send labels to device

            # apply transform and model on whole batch directly on device
            output = model(data_iter.float())
            prob = torch.sigmoid(output) # probability
            labels_iter = labels_iter.unsqueeze(1) 

            pbar_test() # update progress bar
            total_prob.append(prob.cpu().detach().numpy()) # append probability
            total_labels.append(labels_iter.cpu().detach().numpy()) # append labels
    
    total_labels = np.concatenate(total_labels) # concatenate all labels
    total_prob = np.concatenate(total_prob) # concatenate all probabilities
            
    eer, thresh = my_functions.compute_eer(total_labels,total_prob) # compute equal error rate
    fpr, fnr, _ = det_curve(total_labels, total_prob) # compute false positive rate and false negative rate
    my_functions.DETCurve(fpr = fpr, fnr = fnr, eer_fpr = eer,eer_fnr = eer, model_name = model_name,plot_type="step") # plot DET curve
    
    prediction = np.array([1 if total_prob_element > thresh else 0 for total_prob_element in total_prob]) # prediction for specific threshold

    matrix_confusion = sklearn.metrics.confusion_matrix(total_labels, prediction) # confusion matrix
    
    balanced_acc_batch.append(sklearn.metrics.balanced_accuracy_score(total_labels,prediction))  # balanced accuracy per batch

    f1_score_batch.append(sklearn.metrics.f1_score(total_labels, prediction, average='micro'))  # f1 score per batch
    
    f1_score = np.mean(f1_score_batch)  # average validation f1 score per epoch
    balanced_acc_percentage = 100. * np.mean(balanced_acc_batch)  # average validation balanced accuracy per epoch in percentage
    
    print(f"\tTest Balanced Accuracy: ({balanced_acc_percentage:.5f}%) \tF1 Score: {f1_score:.5f}")
  
    return balanced_acc_percentage, f1_score, matrix_confusion, total_prob, total_labels
#################################################################################################################################################

#################################################################################################################################################
import mlflow

# Plot loss and balanced accuracy and eer per epoch 
def plot_loss_accuracy(num_epoch,train_losses,validation_losses,train_acc_percentage,validation_acc_percentage,f1_score_validation,validation_total_eer,train_total_err, title ="",is_mlflow=False):
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    axs[0, 0].plot(range(1,train_losses.shape[0]+1),train_losses, "tab:blue") # plot train loss per batch
    axs[0, 0].plot(range(1,validation_losses.shape[0]+1),validation_losses, "tab:orange") # plot validation loss per batch
    axs[0, 0].legend(["train loss per epoch","validation loss per epoch"], loc ="upper right")
    axs[0, 0].set_title("loss per epoch")
    axs[0, 1].step(range(1,num_epoch+1),f1_score_validation, "tab:pink") # plot train balanced accuracy per epoch
    axs[0, 1].set_title("validation f1 score per epoch")
    axs[1, 0].step(range(1,num_epoch+1),train_acc_percentage, "tab:green") # plot train balanced accuracy per epoch
    #axs[1, 0].set_yscale('log')
    axs[1, 0].set_title("train balanced accuracy per epoch")
    axs[1, 1].step(range(1,num_epoch+1),validation_acc_percentage, "tab:green") # plot validation balanced accuracy per epoch
    #axs[1, 1].set_yscale('log')
    axs[1, 1].set_title("validation balanced accuracy per epoch")
    axs[0, 2].step(range(1,num_epoch+1),train_total_err, "tab:orange") # plot train eer per epoch
    axs[0, 2].set_yscale('log')
    axs[0, 2].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[0, 2].set_title("train eer per epoch")
    axs[1, 2].step(range(1,num_epoch+1),validation_total_eer, "tab:orange") # plot validation eer per epoch
    axs[1, 2].set_yscale('log')
    axs[1, 2].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[1, 2].set_title(f"validation eer per epoch")
    fig.suptitle(title)
    fig.tight_layout()
    if is_mlflow == True:
        mlflow.log_figure(fig, "Plots_graphs.png")
    fig.show()
   

#################################################################################################################################################

# Train the model with 1 neuruon in the last layer with sigmoid activation function in the last layer
from utils.pytorchtools import EarlyStopping
import torch.nn.functional as F 
import mlflow
def train_model_new(training_type,model,train_dataloader,validation_dataloader ,num_epoch,checkpoint_path=None,patience = None):
    '''
    Args:
        training_type: '1_neorun_end_sigmoid' or '2_neorun_end_softmax' or '2_neorun_end_AMS' or '2_neorun_end_OCS'
        model: model to train
        train_dataloader: train dataloader
        validation_dataloader: validation dataloader
        num_epoch: number of epochs
        checkpoint_path: path to save the model
        patience: patience for early stopping
    Returns:
        model: trained model
        train_losses_total: all the train loss
        validation_losses_total: all the validation loss
        train_balanced_acc_total: all the train balanced accuracy
        valid_balanced_acc_total: all the validation balanced accuracy
        f1_score_validation_total: all the validation f1 score
        validation_total_eer: all the validation eer
        train_total_eer: all the train eer
    '''
    # check if training_type is valid
    if training_type != "1_neorun_end_sigmoid" and training_type != "2_neorun_end_softmax" and training_type != "2_neorun_end_AMS" and training_type != "2_neorun_end_OCS":
        raise ValueError("training_type must be '1_neorun_end_sigmoid' or '2_neorun_end_softmax' or '2_neorun_end_AMS' or '2_neorun_end_OCS'")
    
    utils.DNN_functions.set_random_seeds(seed=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    
    if training_type == "2_neorun_end_AMS":
        embedding_dim = model.loss.embedding_dim # embedding dimension for AMS
    elif training_type == "2_neorun_end_OCS":
        embedding_dim = model.loss.feat_dim # embedding dimension for OCS
    else:
        embedding_dim = None # embedding dimension for sigmoid and softmax

    if patience is None:
        patience = num_epoch  
        
    criterion = model.loss # loss function    

    optimizer = model.optimizer # optimizer
        
    scheduler = model.scheduler
    
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True,onnx=False)

    if checkpoint_path is not None:
        early_stopping = EarlyStopping(patience=patience, verbose=False,path=checkpoint_path,onnx=False)

    train_losses_total = []; train_balanced_acc_total = []; validation_losses_total = []; valid_balanced_acc_total = []; f1_score_validation_total = [];
    
    validation_total_eer = []; train_total_eer = [];
    
    train_norm_all_outputs = None
    
    validation_norm_all_outputs = None

    with alive_bar(total=num_epoch) as pbar:
        model.train()
        if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS": 
            criterion.train()
        for epoch in range(1, num_epoch+1 ):
            pbar(skipped=True)
            loss_epoch = [] # loss per epoch
            train_total_labels_epoch = [] # all the labels per epoch
            train_total_score_epoch = []   # all the score per epoch
            if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS":
                train_norm_all_outputs = np.zeros((1,embedding_dim)) # all the outputs per epoch

            for _ , (data_iter, labels_iter) in enumerate(train_dataloader):
                optimizer.zero_grad()# Initializes the weights
                data_iter = data_iter.to(device) # send data to device
                labels_iter = labels_iter.to(device).type(torch.LongTensor) # send labels to device
                output = model(data_iter.float()); # forward propagation
                output = output.to(device).type(torch.Tensor) # send output to device
                
                if training_type == "1_neorun_end_sigmoid":
                    labels_iter = labels_iter.float() # convert labels to float
                    loss = criterion(torch.squeeze(output).to(device), labels_iter.to(device))
                    loss.backward() #Back_propagation
                    optimizer.step() #Updating the optimizer
                    score = torch.sigmoid(output).squeeze()  #Apply sigmoid to the output
               
                if training_type == "2_neorun_end_softmax":
                    loss = criterion(output.to(device), labels_iter.to(device))
                    loss.backward() #Back_propagation
                    optimizer.step() #Updating the optimizer
                    score = torch.softmax(output,dim = 1).squeeze()  #Apply softmax to the output
            
                if training_type == "2_neorun_end_AMS":
                    loss, _ ,logits = criterion(output.to(device), labels_iter.to(device))  #Calculate the loss
                    loss.backward() #Back_propagation
                    optimizer.step() #Updating the optimizer
                    train_norm_output  = F.normalize(output)
                    score = torch.softmax(logits,dim = 1).squeeze()  #Apply softmax to the output
                
                if training_type == "2_neorun_end_OCS":
                    loss, score = criterion(output.to(device), labels_iter.to(device))
                    score = -1*score #because the probability of the positive class is when the score is bigger than thrheshold

                    loss.backward() #Back_propagation
                    optimizer.step() #Updating the optimizer
                    train_norm_output  = F.normalize(output)
                    
                
                labels_iter = labels_iter.squeeze() # squeeze labels
                
                loss_epoch.append(loss.item()) # loss per batch
                
                train_total_score_epoch.append(score.cpu().detach().numpy()) # all the score per epoch
                train_total_labels_epoch.append(labels_iter.cpu().detach().numpy()) # all the labels per epoch
                
                #print(f"Train Epoch: {epoch} [{batch_idx * len(data_iter)}/{len(train_dataloader.dataset)}] ,Train Loss: {loss.item():.5f}") # print loss per batch
           

                if training_type == "2_neorun_end_AMS" or  training_type == "2_neorun_end_OCS":
                    train_norm_all_outputs = np.vstack((train_norm_all_outputs, train_norm_output.detach().numpy()))
            
            train_total_score_epoch = np.concatenate(train_total_score_epoch) # concatenate all the score per epoch
            train_total_labels_epoch = np.concatenate(train_total_labels_epoch) # concatenate all the labels per epoch
            
            if training_type == "2_neorun_end_softmax" or training_type == "2_neorun_end_AMS": 
                    train_total_score_epoch = train_total_score_epoch[:,1]
          
            eer, thresh = my_functions.compute_eer(train_total_labels_epoch,train_total_score_epoch) # compute equal error rate
           
            train_total_eer.append(eer) # all the train eer
            
            prediction_epoch = np.array([1 if total_score_element > thresh else 0 for total_score_element in train_total_score_epoch]) # prediction for specific threshold
 
            train_balanced_acc_total.append(sklearn.metrics.balanced_accuracy_score(train_total_labels_epoch,prediction_epoch)) ## total acurrecy 
                
                
            train_acc_percentage_epoch = (100. * train_balanced_acc_total[epoch-1]); # train balanced accuracy per epoch in percentage
            
            print(f"\nTrain Balanced Accuracy:  {train_acc_percentage_epoch:.2f}%")
            print(f"Train EER:  {100*eer:.2f}%\n")
            
        
            train_losses_total.append(np.average(loss_epoch)) # all the train loss 
            if scheduler is not None:
                scheduler.step() # update learning rate

            if validation_dataloader is not None: # if validation dataloader is not None then do validation after each epoch
                model.eval()
                if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS": 
                    criterion.eval()
                
                val_loss_epoch = []  # loss per epoch
                validation_losses_epoch = [] # average validation loss per epoch
                    
                valid_balanced_acc_epoch = [] # balanced accuracy per epoch
            
                valid_f1_score_epoch = [] # f1 score per epoch
                
                validation_total_labels_epoch = [] # all the labels per epoch
                validation_total_score_epoch = []  # all the score per epoch
                
                if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS":
                    validation_norm_all_outputs = np.zeros((1,embedding_dim)) # all the outputs per epoch
                    
                with torch.no_grad():
                    for _, (data_iter_val, labels_iter_val) in enumerate(validation_dataloader):
                        data_iter_val = data_iter_val.to(device) # send data to device
                        labels_iter_val = labels_iter_val.to(device).type(torch.LongTensor) # send labels to device
                        output = model((data_iter_val).float()) # forward propagation
                        output = output.to(device).type(torch.Tensor) # send output to device
                        if training_type == "1_neorun_end_sigmoid":
                            labels_iter_val = labels_iter_val.float()
                            loss = criterion(torch.squeeze(output).to(device),labels_iter_val.to(device))  #Calculate the loss
                            score = torch.sigmoid(output)  #Apply sigmoid to the output
                        
                        if training_type == "2_neorun_end_softmax":
                            loss = criterion(output.to(device),labels_iter_val.to(device))  #Calculate the loss
                            score = torch.softmax(output,dim = 1)  #Apply softmax to the output
                        
                        if training_type == "2_neorun_end_AMS":
                            loss, _ ,logits = criterion(output.to(device), labels_iter_val.to(device))  #Calculate the loss
                            validation_norm_output  = F.normalize(output)
                            score = torch.softmax(logits,dim = 1).squeeze()  #Apply softmax to the output
                        
                        if training_type == "2_neorun_end_OCS":
                            loss, score = criterion(output.to(device), labels_iter_val.to(device))
                            score = -1*score #because the probability of the positive class is when the score is bigger than threshold
                            validation_norm_output  = F.normalize(output)
                            
                            
                        if torch.cuda.is_available():
                            loss = loss.cpu() # send loss to cpu
                            val_loss_epoch.append(loss.item()) # loss per batch

                            
                            validation_total_score_epoch.append(score.cpu().detach().numpy()) # all the score per epoch
                            
                            validation_total_labels_epoch.append(labels_iter_val.cpu().detach().numpy()) # all the labels per epoch
                            
                            if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS":
                                validation_norm_all_outputs = np.vstack((validation_norm_all_outputs, validation_norm_output)) # all the outputs per epoch

                        
                    validation_total_score_epoch = np.concatenate(validation_total_score_epoch) # concatenate all the score per epoch
                    validation_total_labels_epoch = np.concatenate(validation_total_labels_epoch) # concatenate all the labels per epoch
                    
                    if training_type == "2_neorun_end_softmax" or training_type == "2_neorun_end_AMS":
                         validation_total_score_epoch = validation_total_score_epoch[:,1]
                    
                    eer, thresh = my_functions.compute_eer(validation_total_labels_epoch,validation_total_score_epoch) # compute equal error rate
                    
                    prediction_epoch = np.array([1 if total_score_element > thresh else 0 for total_score_element in validation_total_score_epoch]) # prediction for specific threshold

                        
                    valid_balanced_acc_epoch.append(sklearn.metrics.balanced_accuracy_score(validation_total_labels_epoch,prediction_epoch)) ## acurrecy
                            
                    valid_f1_score_epoch.append(sklearn.metrics.f1_score(validation_total_labels_epoch,prediction_epoch,average = 'micro')) ## f1 score
        

                    validation_losses_epoch = np.average(val_loss_epoch)     # average validation loss per epoch      
                    validation_losses_total.append(validation_losses_epoch)           # all the validation loss 
                    
                    valid_balanced_acc_total.append(valid_balanced_acc_epoch) # all the validation balanced accuracy
                    
                    f1_score_validation_total.append(valid_f1_score_epoch) # all the validation f1 score
                    
                    validation_total_eer.append(eer) # all the validation eer
                    
                    valid_balanced_acc_percentage_epoch = (100. * valid_balanced_acc_epoch[0])  # average validation balanced accuracy per epoch in percentage

                    print(f"\nValidation {epoch:.0f} ,Validation Loss {validation_losses_epoch:.5f} , Validation Balanced Accuracy: {valid_balanced_acc_percentage_epoch:.2f}%")
                    print(f"Validation EER:  {100*eer:.2f}%")
                    print(f"Validation f1 Score: {100*valid_f1_score_epoch[0]:.2f}% \n")
                    
                    early_stopping(validation_losses_epoch, model,batch_size = validation_dataloader.batch_size, num_features = data_iter_val.shape[1] , num_epoch=epoch) # early stopping
                    print("\n")

            #if (checkpoint_path is not None): # save model
            #       torch.save(model.state_dict(), checkpoint_path.format(epoch))
     
    if training_type  == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS":                
        train_norm_all_outputs = train_norm_all_outputs[1::] # all the train outputs per last epoch
        validation_norm_all_outputs = validation_norm_all_outputs[1::] # all the valdation outputs per last epoch

        #utils.sphere_plots.sphere_plot(train_norm_all_outputs,train_total_labels_epoch,scale= model.scale, margin = model.margin ,title = "Train Samples",figure_path=None) 

        #utils.sphere_plots.sphere_plot(validation_norm_all_outputs,validation_total_labels_epoch,scale= model.scale, margin = model.margin ,title = "Validation Samples",figure_path=None) 
    
    model.loss = criterion # loss function   
    model.optimizer = optimizer # optimizer  
    model.scheduler = scheduler # scheduler  
    # convert list to numpy array               
    train_losses_total       = np.array(train_losses_total).ravel()         # all the train loss
    validation_losses_total  = np.array(validation_losses_total).ravel()    # all the validation loss
    train_balanced_acc_total = np.array(train_balanced_acc_total).ravel()   # all the train balanced accuracy
    valid_balanced_acc_total = np.array(valid_balanced_acc_total).ravel()   # all the validation balanced accuracy
    f1_score_validation_total= np.array(f1_score_validation_total).ravel()  # all the validation f1 score
    train_total_eer          = np.array(train_total_eer).ravel()            #  all the train eer   
    validation_total_eer     = np.array(validation_total_eer).ravel()       # all the validation eer
    
    
    if checkpoint_path is not None:
        model = pickle.load(open(checkpoint_path, 'rb'))
        model = model.to(device)
    
    else:
        checkpoint_path = 'checkpoint.pkl'
        model = pickle.load(open(checkpoint_path, 'rb'))
        model = model.to(device)

    return model,train_losses_total,validation_losses_total,train_balanced_acc_total,valid_balanced_acc_total,f1_score_validation_total,validation_total_eer,train_total_eer


#################################################################################################################################################




from utils.pytorchtools import EarlyStopping
import torch.nn.functional as F 
import mlflow
def train_model_new_diffrenet_optimazers_for_losses(training_type,model,train_dataloader,validation_dataloader ,num_epoch,checkpoint_path=None,patience = None):
    '''
    Args:
        training_type: '1_neorun_end_sigmoid' or '2_neorun_end_softmax' or '2_neorun_end_AMS' or '2_neorun_end_OCS'
        model: model to train
        train_dataloader: train dataloader
        validation_dataloader: validation dataloader
        num_epoch: number of epochs
        checkpoint_path: path to save the model
        patience: patience for early stopping
    Returns:
        model: trained model
        train_losses_total: all the train loss
        validation_losses_total: all the validation loss
        train_balanced_acc_total: all the train balanced accuracy
        valid_balanced_acc_total: all the validation balanced accuracy
        f1_score_validation_total: all the validation f1 score
        validation_total_eer: all the validation eer
        train_total_eer: all the train eer
    '''
    # check if training_type is valid
    if training_type != "1_neorun_end_sigmoid" and training_type != "2_neorun_end_softmax" and training_type != "2_neorun_end_AMS" and training_type != "2_neorun_end_OCS":
        raise ValueError("training_type must be '1_neorun_end_sigmoid' or '2_neorun_end_softmax' or '2_neorun_end_AMS' or '2_neorun_end_OCS'")
    
    utils.DNN_functions.set_random_seeds(seed=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    
    if training_type == "2_neorun_end_AMS":
        embedding_dim = model.loss.embedding_dim # embedding dimension for AMS
    elif training_type == "2_neorun_end_OCS":
        embedding_dim = model.loss.feat_dim # embedding dimension for OCS
    else:
        embedding_dim = None # embedding dimension for sigmoid and softmax

    if patience is None:
        patience = num_epoch  
        
    criterion = model.loss # loss function    

    optimizer = model.optimizer # optimizer
        
    scheduler = model.scheduler
    
    loss_optimizer = model.loss_optimizer
    
    loss_scheduler = model.loss_scheduler
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True,onnx=False)

    if checkpoint_path is not None:
        early_stopping = EarlyStopping(patience=patience, verbose=False,path=checkpoint_path,onnx=False)

    train_losses_total = []; train_balanced_acc_total = []; validation_losses_total = []; valid_balanced_acc_total = []; f1_score_validation_total = [];
    
    validation_total_eer = []; train_total_eer = [];
    
    train_norm_all_outputs = None
    
    validation_norm_all_outputs = None

    with alive_bar(total=num_epoch) as pbar:
        model.train()
        if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS": 
            criterion.train()
        for epoch in range(1, num_epoch+1 ):
            pbar(skipped=True)
            loss_epoch = [] # loss per epoch
            train_total_labels_epoch = [] # all the labels per epoch
            train_total_score_epoch = []   # all the score per epoch
            if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS":
                train_norm_all_outputs = np.zeros((1,embedding_dim)) # all the outputs per epoch

            for _ , (data_iter, labels_iter) in enumerate(train_dataloader):
                optimizer.zero_grad()# Initializes the weights
                data_iter = data_iter.to(device) # send data to device
                labels_iter = labels_iter.to(device).type(torch.LongTensor) # send labels to device
                output = model(data_iter.float()); # forward propagation
                output = output.to(device).type(torch.Tensor) # send output to device
                
                if training_type == "1_neorun_end_sigmoid":
                    labels_iter = labels_iter.float() # convert labels to float
                    loss = criterion(torch.squeeze(output).to(device), labels_iter.to(device))
                    loss.backward() #Back_propagation
                    optimizer.step() #Updating the optimizer
                    score = torch.sigmoid(output).squeeze()  #Apply sigmoid to the output
               
                if training_type == "2_neorun_end_softmax":
                    loss = criterion(output.to(device), labels_iter.to(device))
                    loss.backward() #Back_propagation
                    optimizer.step() #Updating the optimizer
                    score = torch.softmax(output,dim = 1).squeeze()  #Apply softmax to the output
            
                if training_type == "2_neorun_end_AMS":
                    loss_optimizer.zero_grad()
                    loss, _ ,logits = criterion(output.to(device), labels_iter.to(device))  #Calculate the loss
                    loss.backward() #Back_propagation
                    optimizer.step() #Updating the optimizer
                    loss_optimizer.step() #Updating the optimizer
                    train_norm_output  = F.normalize(output)
                    score = torch.softmax(logits,dim = 1).squeeze()  #Apply softmax to the output
                
                if training_type == "2_neorun_end_OCS":
                    loss_optimizer.zero_grad() # Initializes the weights
                    loss, score = criterion(output.to(device), labels_iter.to(device))
                    score = -1*score #because the probability of the positive class is when the score is bigger than thrheshold

                    loss.backward() #Back_propagation
                    optimizer.step() #Updating the optimizer
                    loss_optimizer.step() #Updating the optimizer
                    train_norm_output  = F.normalize(output)
                    
                
                labels_iter = labels_iter.squeeze() # squeeze labels
                
                loss_epoch.append(loss.item()) # loss per batch
                
                train_total_score_epoch.append(score.cpu().detach().numpy()) # all the score per epoch
                train_total_labels_epoch.append(labels_iter.cpu().detach().numpy()) # all the labels per epoch
                
                #print(f"Train Epoch: {epoch} [{batch_idx * len(data_iter)}/{len(train_dataloader.dataset)}] ,Train Loss: {loss.item():.5f}") # print loss per batch
           

                if training_type == "2_neorun_end_AMS" or  training_type == "2_neorun_end_OCS":
                    train_norm_all_outputs = np.vstack((train_norm_all_outputs, train_norm_output.detach().numpy()))
            
            train_total_score_epoch = np.concatenate(train_total_score_epoch) # concatenate all the score per epoch
            train_total_labels_epoch = np.concatenate(train_total_labels_epoch) # concatenate all the labels per epoch
            
            if training_type == "2_neorun_end_softmax" or training_type == "2_neorun_end_AMS": 
                    train_total_score_epoch = train_total_score_epoch[:,1]
          
            eer, thresh = my_functions.compute_eer(train_total_labels_epoch,train_total_score_epoch) # compute equal error rate
           
            train_total_eer.append(eer) # all the train eer
            
            prediction_epoch = np.array([1 if total_score_element > thresh else 0 for total_score_element in train_total_score_epoch]) # prediction for specific threshold
 
            train_balanced_acc_total.append(sklearn.metrics.balanced_accuracy_score(train_total_labels_epoch,prediction_epoch)) ## total acurrecy 
                
                
            train_acc_percentage_epoch = (100. * train_balanced_acc_total[epoch-1]); # train balanced accuracy per epoch in percentage
            
            print(f"\nTrain Balanced Accuracy:  {train_acc_percentage_epoch:.2f}%")
            print(f"Train EER:  {100*eer:.2f}%\n")
            
        
            train_losses_total.append(np.average(loss_epoch)) # all the train loss 
            if scheduler is not None:
                scheduler.step() # update learning rate
            if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS":
                loss_scheduler.step() # update learning rate

            if validation_dataloader is not None: # if validation dataloader is not None then do validation after each epoch
                model.eval()
                if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS": 
                    criterion.eval()
                
                val_loss_epoch = []  # loss per epoch
                validation_losses_epoch = [] # average validation loss per epoch
                    
                valid_balanced_acc_epoch = [] # balanced accuracy per epoch
            
                valid_f1_score_epoch = [] # f1 score per epoch
                
                validation_total_labels_epoch = [] # all the labels per epoch
                validation_total_score_epoch = []  # all the score per epoch
                
                if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS":
                    validation_norm_all_outputs = np.zeros((1,embedding_dim)) # all the outputs per epoch
                    
                with torch.no_grad():
                    for _, (data_iter_val, labels_iter_val) in enumerate(validation_dataloader):
                        data_iter_val = data_iter_val.to(device) # send data to device
                        labels_iter_val = labels_iter_val.to(device).type(torch.LongTensor) # send labels to device
                        output = model((data_iter_val).float()) # forward propagation
                        output = output.to(device).type(torch.Tensor) # send output to device
                        if training_type == "1_neorun_end_sigmoid":
                            labels_iter_val = labels_iter_val.float()
                            loss = criterion(torch.squeeze(output).to(device),labels_iter_val.to(device))  #Calculate the loss
                            score = torch.sigmoid(output)  #Apply sigmoid to the output
                        
                        if training_type == "2_neorun_end_softmax":
                            loss = criterion(output.to(device),labels_iter_val.to(device))  #Calculate the loss
                            score = torch.softmax(output,dim = 1)  #Apply softmax to the output
                        
                        if training_type == "2_neorun_end_AMS":
                            loss, _ ,logits = criterion(output.to(device), labels_iter_val.to(device))  #Calculate the loss
                            validation_norm_output  = F.normalize(output)
                            score = torch.softmax(logits,dim = 1).squeeze()  #Apply softmax to the output
                        
                        if training_type == "2_neorun_end_OCS":
                            loss, score = criterion(output.to(device), labels_iter_val.to(device))
                            score = -1*score #because the probability of the positive class is when the score is bigger than threshold
                            validation_norm_output  = F.normalize(output)
                            
                            
                        if torch.cuda.is_available():
                            loss = loss.cpu() # send loss to cpu
                            val_loss_epoch.append(loss.item()) # loss per batch

                            
                            validation_total_score_epoch.append(score.cpu().detach().numpy()) # all the score per epoch
                            
                            validation_total_labels_epoch.append(labels_iter_val.cpu().detach().numpy()) # all the labels per epoch
                            
                            if training_type == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS":
                                validation_norm_all_outputs = np.vstack((validation_norm_all_outputs, validation_norm_output)) # all the outputs per epoch

                        
                    validation_total_score_epoch = np.concatenate(validation_total_score_epoch) # concatenate all the score per epoch
                    validation_total_labels_epoch = np.concatenate(validation_total_labels_epoch) # concatenate all the labels per epoch
                    
                    if training_type == "2_neorun_end_softmax" or training_type == "2_neorun_end_AMS":
                         validation_total_score_epoch = validation_total_score_epoch[:,1]
                    
                    eer, thresh = my_functions.compute_eer(validation_total_labels_epoch,validation_total_score_epoch) # compute equal error rate
                    
                    prediction_epoch = np.array([1 if total_score_element > thresh else 0 for total_score_element in validation_total_score_epoch]) # prediction for specific threshold

                        
                    valid_balanced_acc_epoch.append(sklearn.metrics.balanced_accuracy_score(validation_total_labels_epoch,prediction_epoch)) ## acurrecy
                            
                    valid_f1_score_epoch.append(sklearn.metrics.f1_score(validation_total_labels_epoch,prediction_epoch,average = 'micro')) ## f1 score
        

                    validation_losses_epoch = np.average(val_loss_epoch)     # average validation loss per epoch      
                    validation_losses_total.append(validation_losses_epoch)           # all the validation loss 
                    
                    valid_balanced_acc_total.append(valid_balanced_acc_epoch) # all the validation balanced accuracy
                    
                    f1_score_validation_total.append(valid_f1_score_epoch) # all the validation f1 score
                    
                    validation_total_eer.append(eer) # all the validation eer
                    
                    valid_balanced_acc_percentage_epoch = (100. * valid_balanced_acc_epoch[0])  # average validation balanced accuracy per epoch in percentage

                    print(f"\nValidation {epoch:.0f} ,Validation Loss {validation_losses_epoch:.5f} , Validation Balanced Accuracy: {valid_balanced_acc_percentage_epoch:.2f}%")
                    print(f"Validation EER:  {100*eer:.2f}%")
                    print(f"Validation f1 Score: {100*valid_f1_score_epoch[0]:.2f}% \n")
                    
                    early_stopping(validation_losses_epoch, model,batch_size = validation_dataloader.batch_size, num_features = data_iter_val.shape[1] , num_epoch=epoch) # early stopping
                    print("\n")

            #if (checkpoint_path is not None): # save model
            #       torch.save(model.state_dict(), checkpoint_path.format(epoch))
     
    if training_type  == "2_neorun_end_AMS" or training_type == "2_neorun_end_OCS":                
        train_norm_all_outputs = train_norm_all_outputs[1::] # all the train outputs per last epoch
        validation_norm_all_outputs = validation_norm_all_outputs[1::] # all the valdation outputs per last epoch

        #utils.sphere_plots.sphere_plot(train_norm_all_outputs,train_total_labels_epoch,scale= model.scale, margin = model.margin ,title = "Train Samples",figure_path=None) 

        #utils.sphere_plots.sphere_plot(validation_norm_all_outputs,validation_total_labels_epoch,scale= model.scale, margin = model.margin ,title = "Validation Samples",figure_path=None) 
    
    model.loss = criterion # loss function   
    model.optimizer = optimizer # optimizer  
    model.scheduler = scheduler # scheduler  
    # convert list to numpy array               
    train_losses_total       = np.array(train_losses_total).ravel()         # all the train loss
    validation_losses_total  = np.array(validation_losses_total).ravel()    # all the validation loss
    train_balanced_acc_total = np.array(train_balanced_acc_total).ravel()   # all the train balanced accuracy
    valid_balanced_acc_total = np.array(valid_balanced_acc_total).ravel()   # all the validation balanced accuracy
    f1_score_validation_total= np.array(f1_score_validation_total).ravel()  # all the validation f1 score
    train_total_eer          = np.array(train_total_eer).ravel()            #  all the train eer   
    validation_total_eer     = np.array(validation_total_eer).ravel()       # all the validation eer
    
    if checkpoint_path is not None:
        model = pickle.load(open(checkpoint_path, 'rb'))
        model = model.to(device)
    
    else:
        checkpoint_path = 'checkpoint.pkl'
        model = pickle.load(open(checkpoint_path, 'rb'))
        model = model.to(device)

    return model,train_losses_total,validation_losses_total,train_balanced_acc_total,valid_balanced_acc_total,f1_score_validation_total,validation_total_eer,train_total_eer

