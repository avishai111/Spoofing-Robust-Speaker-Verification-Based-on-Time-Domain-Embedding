    
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
# Train the model with 1 neuruon in the last layer with sigmoid activation function in the last layer
def train_model(model,train_dataloader,validation_dataloader ,num_epoch,learning_rate,weight_decay,checkpoint_path='/checks_point/Anti_Spoofing_Deeplearning-{}.pk',patience = None, optimizer=None):
    utils.DNN_functions.set_random_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    criterion = torch.nn.BCEWithLogitsLoss() #Loss function
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay);

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epoch, gamma=0.1);
    
    if patience is None:
        patience = num_epoch   
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    if checkpoint_path is not None:
        early_stopping = EarlyStopping(patience=patience, verbose=True,path=checkpoint_path)

    train_losses_total = []; train_balanced_acc_total = []; validation_losses_total = []; valid_balanced_acc_total = []; f1_score_validation_total = [];
    
    validation_total_eer = []; train_total_eer = [];

    with alive_bar(total=num_epoch) as pbar:
        model.train()
        for epoch in range(1, num_epoch+1 ):
            pbar(skipped=True)
            loss_epoch = [] # loss per epoch
            total_labels_epoch = [] # all the labels per epoch
            total_prob_epoch = []   # all the prob per epoch
            for batch_idx, (data_iter, labels_iter) in enumerate(train_dataloader):
                optimizer.zero_grad();# Initializes the weights
                data_iter = data_iter.to(device) # send data to device
                labels_iter = labels_iter.to(device) # send labels to device
                data_iter = data_iter.float()
                output = model(data_iter); # forward propagation
                loss = criterion(output.squeeze(), labels_iter) #Calculate the loss
                loss.backward() #Back_propagation
                optimizer.step() #Updating the optimizer
                prob = torch.sigmoid(output).squeeze()  #Apply sigmoid to the output
                
                labels_iter = labels_iter.squeeze() # squeeze labels
                
                loss_epoch.append(loss.item()) # loss per batch
                
                total_prob_epoch.append(prob.cpu().detach().numpy().ravel()) # all the prob per epoch
                total_labels_epoch.append(labels_iter.cpu().detach().numpy().ravel()) # all the labels per epoch
                
                #print(f"Train Epoch: {epoch} [{batch_idx * len(data_iter)}/{len(train_dataloader.dataset)}] ,Train Loss: {loss.item():.5f}") # print loss per batch
            
            total_prob_epoch = np.concatenate(total_prob_epoch) # concatenate all the prob per epoch
            total_labels_epoch = np.concatenate(total_labels_epoch) # concatenate all the labels per epoch
          
            eer, thresh = my_functions.compute_eer(total_labels_epoch,total_prob_epoch) # compute equal error rate
            
            train_total_eer.append(eer) # all the train eer
            
            prediction_epoch = np.array([1 if total_prob_element > thresh else 0 for total_prob_element in total_prob_epoch]) # prediction for specific threshold
 
            train_balanced_acc_total.append(sklearn.metrics.balanced_accuracy_score(total_labels_epoch,prediction_epoch)) ## total acurrecy 
                
                
            train_acc_percentage_epoch = (100. * train_balanced_acc_total[epoch-1]); # train balanced accuracy per epoch in percentage
            
            print(f"\nTrain Balanced Accuracy:  {train_acc_percentage_epoch:.2f}%")
            print(f"Train EER:  {100*eer:.2f}%\n")
            
        
            train_losses_total.append(np.average(loss_epoch)) # all the train loss 
            scheduler.step() # update learning rate

            if validation_dataloader is not None: # if validation dataloader is not None then do validation after each epoch
                model.eval()
                
                val_loss_epoch = []  # loss per epoch
                validation_losses_epoch = [] # average validation loss per epoch
                    
                valid_balanced_acc_epoch = [] # balanced accuracy per epoch
            
                valid_f1_score_epoch = [] # f1 score per epoch
                
                total_labels_epoch = [] # all the labels per epoch
                total_prob_epoch = []  # all the prob per epoch
                    
                with torch.no_grad():
                    for batch_idx, (data_iter_val, labels_iter_val) in enumerate(validation_dataloader):
                        data_iter_val = data_iter_val.to(device) # send data to device
                        labels_iter_val = labels_iter_val.to(device).float() # send labels to device
                        output = model((data_iter_val).float()) # forward propagation
                        loss = criterion(output.squeeze(),labels_iter_val)  #Calculate the loss
                        prob = torch.sigmoid(output)  #Apply sigmoid to the output
                        if torch.cuda.is_available():
                            loss = loss.cpu() # send loss to cpu
                            val_loss_epoch.append(loss.item()) # loss per batch
                            #prediction = torch.round(prob).squeeze()
                            
                            total_prob_epoch.append(prob.cpu().detach().numpy().ravel()) # all the prob per epoch
                            
                            total_labels_epoch.append(labels_iter_val.cpu().detach().numpy().ravel()) # all the labels per epoch
                        
                    total_prob_epoch = np.concatenate(total_prob_epoch) # concatenate all the prob per epoch
                    total_labels_epoch = np.concatenate(total_labels_epoch) # concatenate all the labels per epoch
                    
                    eer, thresh = my_functions.compute_eer(total_labels_epoch,total_prob_epoch) # compute equal error rate
                    
                    prediction_epoch = np.array([1 if total_prob_element > thresh else 0 for total_prob_element in total_prob_epoch]) # prediction for specific threshold

                        
                    valid_balanced_acc_epoch.append(sklearn.metrics.balanced_accuracy_score(total_labels_epoch,prediction_epoch)) ## acurrecy
                            
                    valid_f1_score_epoch.append(sklearn.metrics.f1_score(total_labels_epoch,prediction_epoch,average = 'micro')) ## f1 score
        

                    validation_losses_epoch = np.average(val_loss_epoch)     # average validation loss per epoch      
                    validation_losses_total.append(validation_losses_epoch)           # all the validation loss 
                    
                    valid_balanced_acc_total.append(valid_balanced_acc_epoch) # all the validation balanced accuracy
                    
                    f1_score_validation_total.append(valid_f1_score_epoch) # all the validation f1 score
                    
                    validation_total_eer.append(eer) # all the validation eer
                    
                    valid_balanced_acc_percentage_epoch = (100. * valid_balanced_acc_epoch[0])  # average validation balanced accuracy per epoch in percentage

                    print(f"\nValidation {epoch:.0f} ,Validation Loss {validation_losses_epoch:.5f} , Validation Balanced Accuracy: {valid_balanced_acc_percentage_epoch:.2f}%")
                    print(f"Validation EER:  {100*eer:.2f}%")
                    print(f"Validation f1 Score: {100*valid_f1_score_epoch[0]:.2f}% \n")
                    
                    early_stopping(validation_losses_epoch, model,batch_size = data_iter_val.shape[0], num_features = data_iter_val.shape[1::]) # early stopping
                    print("\n")

            if (checkpoint_path is not None): # save model
                    torch.save(model.state_dict(), checkpoint_path.format(epoch))
     
    # convert list to numpy array               
    train_losses_total       = np.array(train_losses_total).ravel()         # all the train loss
    validation_losses_total  = np.array(validation_losses_total).ravel()    # all the validation loss
    train_balanced_acc_total = np.array(train_balanced_acc_total).ravel()   # all the train balanced accuracy
    valid_balanced_acc_total = np.array(valid_balanced_acc_total).ravel()   # all the validation balanced accuracy
    f1_score_validation_total= np.array(f1_score_validation_total).ravel()  # all the validation f1 score
    train_total_eer          = np.array(train_total_eer).ravel()            #  all the train eer   
    validation_total_eer     = np.array(validation_total_eer).ravel()       # all the validation eer

    return train_losses_total,validation_losses_total,train_balanced_acc_total,valid_balanced_acc_total,f1_score_validation_total,validation_total_eer,train_total_eer

#################################################################################################################################################


#################################################################################################################################################

# Plot loss and balanced accuracy and eer per epoch 
def plot_loss_accuracy(num_epoch,train_losses,validation_losses,train_acc_percentage,validation_acc_percentage,f1_score_validation,validation_total_eer,train_total_err, title =""):
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    axs[0, 0].plot(range(1,train_losses.shape[0]+1),train_losses, "tab:blue") # plot train loss per batch
    axs[0, 0].plot(range(1,validation_losses.shape[0]+1),validation_losses, "tab:orange") # plot validation loss per batch
    axs[0, 0].legend(["train loss per epoch","validation loss per epoch"], loc ="upper right")
    axs[0, 0].set_title("loss per epoch")
    axs[0, 1].step(range(1,num_epoch+1),f1_score_validation, "tab:pink") # plot train balanced accuracy per epoch
    axs[0, 1].set_title("validation f1 score per epoch")
    axs[1, 0].step(range(1,num_epoch+1),train_acc_percentage, "tab:green") # plot train balanced accuracy per epoch
    axs[1, 0].set_title("train balanced accuracy per epoch")
    axs[1, 1].step(range(1,num_epoch+1),validation_acc_percentage, "tab:green") # plot validation balanced accuracy per epoch
    axs[1, 1].set_title("validation balanced accuracy per epoch")
    axs[0, 2].step(range(1,num_epoch+1),train_total_err, "tab:orange") # plot train eer per epoch
    axs[0, 2].set_title("train eer per epoch")
    axs[1, 2].step(range(1,num_epoch+1),validation_total_eer, "tab:orange") # plot validation eer per epoch
    axs[1, 2].set_title(f"validation eer per epoch")
    fig.suptitle(title)
    fig.tight_layout()
    fig.show()

#################################################################################################################################################


#################################################################################################################################################
# Train the model with 2 neuruons in the last layer with softmax activation function in the last layer
def train_model_2n(model,train_dataloader,validation_dataloader ,num_epoch,learning_rate,weight_decay,checkpoint_path='/checks_point/Anti_Spoofing_Deeplearning-{}.pk',patience = None, optimizer=None):
    utils.DNN_functions.set_random_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    criterion = torch.nn.CrossEntropyLoss() #Loss function
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay);

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epoch, gamma=0.1);
    
    if patience is None:
        patience = num_epoch   
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    if checkpoint_path is not None:
        early_stopping = EarlyStopping(patience=patience, verbose=True,path=checkpoint_path)

    train_losses_total = []; train_balanced_acc_total = []; validation_losses_total = []; valid_balanced_acc_total = []; f1_score_validation_total = [];
    
    validation_total_eer = []; train_total_eer = [];

    with alive_bar(total=num_epoch) as pbar:
        model.train()
        for epoch in range(1, num_epoch+1 ):
            pbar(skipped=True)
            loss_epoch = [] # loss per epoch
            total_labels_epoch = [] # all the labels per epoch
            total_prob_epoch = []   # all the prob per epoch
            for batch_idx, (data_iter, labels_iter) in enumerate(train_dataloader):
                optimizer.zero_grad();# Initializes the weights
                data_iter = data_iter.to(device) # send data to device
                labels_iter = labels_iter.to(device).type(torch.LongTensor) # send labels to device
                output = model((data_iter).float()); # forward propagation
                output = output.to(device).type(torch.Tensor) # send output to device
                loss = criterion(output, labels_iter) #Calculate the loss
                loss.backward() #Back_propagation
                optimizer.step() #Updating the optimizer
                prob = torch.softmax(output,dim = 1)  #Apply softmax to the output
               
                labels_iter = labels_iter.squeeze() # squeeze labels
                
                loss_epoch.append(loss.item()) # loss per batch
                
                total_prob_epoch.append(prob.cpu().detach().numpy()) # all the prob per epoch
                total_labels_epoch.append(labels_iter.cpu().detach().numpy()) # all the labels per epoch
                
                #print(f"Train Epoch: {epoch} [{batch_idx * len(data_iter)}/{len(train_dataloader.dataset)}] ,Train Loss: {loss.item():.5f}") # print loss per batch
            
            total_prob_epoch = np.concatenate(total_prob_epoch) # concatenate all the prob per epoch
            total_labels_epoch = np.concatenate(total_labels_epoch) # concatenate all the labels per epoch

            total_prob_epoch_pos = total_prob_epoch[:,1] # all the prob per epoch for positive class
            
            integers_labels = total_labels_epoch # convert labels to integers from one hot encoding
            
            eer, thresh = my_functions.compute_eer(integers_labels,total_prob_epoch_pos) # compute equal error rate
            
            train_total_eer.append(eer) # all the train eer
            
            prediction_epoch = np.array([1 if total_prob_element > thresh else 0 for total_prob_element in total_prob_epoch_pos]) # prediction for specific threshold
 
            train_balanced_acc_total.append(sklearn.metrics.balanced_accuracy_score(integers_labels,prediction_epoch)) ## total acurrecy 
                
                
            train_acc_percentage_epoch = (100. * train_balanced_acc_total[epoch-1]); # train balanced accuracy per epoch in percentage
            
            print(f"\nTrain Balanced Accuracy:  {train_acc_percentage_epoch:.2f}%")
            print(f"Train EER:  {100*eer:.2f}%\n")
            
        
            train_losses_total.append(np.average(loss_epoch)) # all the train loss 
            scheduler.step() # update learning rate

            if validation_dataloader is not None: # if validation dataloader is not None then do validation after each epoch
                model.eval()
                
                val_loss_epoch = []  # loss per epoch
                validation_losses_epoch = [] # average validation loss per epoch
                    
                valid_balanced_acc_epoch = [] # balanced accuracy per epoch
            
                valid_f1_score_epoch = [] # f1 score per epoch
                
                total_labels_epoch = [] # all the labels per epoch
                total_prob_epoch = []  # all the prob per epoch
                    
                with torch.no_grad():
                    for batch_idx, (data_iter_val, labels_iter_val) in enumerate(validation_dataloader):
                        data_iter_val = data_iter_val.to(device) # send data to device
                        labels_iter_val = labels_iter_val.to(device).type(torch.LongTensor) # send labels to device
                        output = model((data_iter_val).float()) # forward propagation
                        output = output.to(device).type(torch.Tensor) # send output to device
                        loss = criterion(output,labels_iter_val)  #Calculate the loss
                        prob = torch.softmax(output,dim = 1)  #Apply softmax to the output
                        if torch.cuda.is_available():
                            loss = loss.cpu() # send loss to cpu
                            val_loss_epoch.append(loss.item()) # loss per batch
                            
                            total_prob_epoch.append(prob.cpu().detach().numpy()) # all the prob per epoch
                            
                            total_labels_epoch.append(labels_iter_val.cpu().detach().numpy()) # all the labels per epoch
                        
                    total_prob_epoch = np.concatenate(total_prob_epoch) # concatenate all the prob per epoch
                    total_labels_epoch = np.concatenate(total_labels_epoch) # concatenate all the labels per epoch
                    
                    total_prob_epoch_pos = total_prob_epoch[:,1] # all the prob per epoch for positive class
                   
                    integers_labels = total_labels_epoch # convert labels to integers from one hot encoding
        
                    eer, thresh = my_functions.compute_eer(integers_labels,total_prob_epoch_pos) # compute equal error rate
                    
                    prediction_epoch = np.array([1 if total_prob_element > thresh else 0 for total_prob_element in total_prob_epoch_pos]) # prediction for specific threshold
                        
                    valid_balanced_acc_epoch.append(sklearn.metrics.balanced_accuracy_score(integers_labels,prediction_epoch)) ## acurrecy
                            
                    valid_f1_score_epoch.append(sklearn.metrics.f1_score(integers_labels,prediction_epoch,average = 'micro')) ## f1 score
        

                    validation_losses_epoch = np.average(val_loss_epoch)     # average validation loss per epoch      
                    validation_losses_total.append(validation_losses_epoch)           # all the validation loss 
                    
                    valid_balanced_acc_total.append(valid_balanced_acc_epoch) # all the validation balanced accuracy
                    
                    f1_score_validation_total.append(valid_f1_score_epoch) # all the validation f1 score
                    
                    validation_total_eer.append(eer) # all the validation eer
                    
                    valid_balanced_acc_percentage_epoch = (100. * valid_balanced_acc_epoch[0])  # average validation balanced accuracy per epoch in percentage

                    print(f"\nValidation {epoch:.0f} ,Validation Loss {validation_losses_epoch:.5f} , Validation Balanced Accuracy: {valid_balanced_acc_percentage_epoch:.2f}%")
                    print(f"Validation EER:  {100*eer:.2f}%")
                    print(f"Validation f1 Score: {100*valid_f1_score_epoch[0]:.2f}% \n")
                    
                    early_stopping(validation_losses_epoch, model,batch_size = data_iter_val.shape[0], num_features = data_iter_val.shape[1::]) # early stopping
                    print("\n")

            if (checkpoint_path is not None): # save model
                    torch.save(model.state_dict(), checkpoint_path.format(epoch))
     
    # convert list to numpy array               
    train_losses_total       = np.array(train_losses_total).ravel()         # all the train loss
    validation_losses_total  = np.array(validation_losses_total).ravel()    # all the validation loss
    train_balanced_acc_total = np.array(train_balanced_acc_total).ravel()   # all the train balanced accuracy
    valid_balanced_acc_total = np.array(valid_balanced_acc_total).ravel()   # all the validation balanced accuracy
    f1_score_validation_total= np.array(f1_score_validation_total).ravel()  # all the validation f1 score
    train_total_eer          = np.array(train_total_eer).ravel()            #  all the train eer   
    validation_total_eer     = np.array(validation_total_eer).ravel()       # all the validation eer

    return train_losses_total,validation_losses_total,train_balanced_acc_total,valid_balanced_acc_total,f1_score_validation_total,validation_total_eer,train_total_eer

#################################################################################################################################################



#################################################################################################################################################
# Train the model with 2 neuruons in the last layer with softmax margin loss 
from utils.pytorchtools import EarlyStopping
import torch.nn.functional as F 
import utils.sphere_plots

# Train the model with 2 neuruons in the last layer with softmax activation function in the last layer
def train_model_softmax_margim(model,train_dataloader,validation_dataloader ,num_epoch,learning_rate,weight_decay,checkpoint_path='/checks_point/Anti_Spoofing_Deeplearning-{}.pk',
                                patience = None, optimizer=None, scale = 30, margin = 0.4):
    utils.DNN_functions.set_random_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    
    # scale and margin for sphere plot

    criterion = model.loss.cpu() # loss function
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay);

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epoch, gamma=0.1);
    
    if patience is None:
        patience = num_epoch   
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    if checkpoint_path is not None:
        early_stopping = EarlyStopping(patience=patience, verbose=True,path=checkpoint_path)

    train_losses_total = []; train_balanced_acc_total = []; validation_losses_total = []; valid_balanced_acc_total = []; f1_score_validation_total = [];
    
    validation_total_eer = []; train_total_eer = [];
    
    

    with alive_bar(total=num_epoch) as pbar:
        model.train()
        for epoch in range(1, num_epoch+1 ):
            pbar(skipped=True)
            loss_epoch = [] # loss per epoch
            total_labels_epoch = [] # all the labels per epoch
            total_prob_epoch = []   # all the prob per epoch
            
            train_norm_all_outputs = np.zeros((1,model.loss.embedding_dim)) # all the outputs per epoch
            
            for batch_idx, (data_iter, labels_iter) in enumerate(train_dataloader):
                optimizer.zero_grad();# Initializes the weights
                data_iter = data_iter.to(device) # send data to device
                labels_iter = labels_iter.to(device).type(torch.LongTensor) # send labels to device
                output = model((data_iter).float()); # forward propagation
                output = output.to(device).type(torch.Tensor) # send output to device
                loss, _ ,logits = criterion(output, labels_iter) #Calculate the loss
                loss.backward() #Back_propagation
                
                optimizer.step() #Updating the optimizer
                train_norm_output = F.normalize(output)
                
                prob = torch.softmax(logits,dim = 1)  #Apply softmax to the logits to get the probability
                
                labels_iter = labels_iter.squeeze() # squeeze labels
                
                loss_epoch.append(loss.item()) # loss per batch
                
                total_prob_epoch.append(prob.cpu().detach().numpy()) # all the prob per epoch
                total_labels_epoch.append(labels_iter.cpu().detach().numpy()) # all the labels per epoch
                
                train_norm_all_outputs = np.vstack((train_norm_all_outputs, train_norm_output.detach().numpy()))
                
                #print(f"Train Epoch: {epoch} [{batch_idx * len(data_iter)}/{len(train_dataloader.dataset)}] ,Train Loss: {loss.item():.5f}") # print loss per batch
            
            total_prob_epoch = np.concatenate(total_prob_epoch) # concatenate all the prob per epoch
            total_labels_epoch = np.concatenate(total_labels_epoch) # concatenate all the labels per epoch

            total_prob_epoch_pos = total_prob_epoch[:,1] # all the prob per epoch for positive class
            
            integers_labels = total_labels_epoch # convert labels to integers from one hot encoding
            
            train_integers_labels = integers_labels
            
            eer, thresh = my_functions.compute_eer(integers_labels,total_prob_epoch_pos) # compute equal error rate
            
            train_total_eer.append(eer) # all the train eer
            
            prediction_epoch = np.array([1 if total_prob_element > thresh else 0 for total_prob_element in total_prob_epoch_pos]) # prediction for specific threshold
 
            train_balanced_acc_total.append(sklearn.metrics.balanced_accuracy_score(integers_labels,prediction_epoch)) ## total acurrecy 
                
                
            train_acc_percentage_epoch = (100. * train_balanced_acc_total[epoch-1]); # train balanced accuracy per epoch in percentage
            
            print(f"\nTrain Balanced Accuracy:  {train_acc_percentage_epoch:.2f}%")
            print(f"Train EER:  {100*eer:.2f}%\n")
            
            
        
            train_losses_total.append(np.average(loss_epoch)) # all the train loss 
            scheduler.step() # update learning rate

            if validation_dataloader is not None: # if validation dataloader is not None then do validation after each epoch
                model.eval()
                
                val_loss_epoch = []  # loss per epoch
                validation_losses_epoch = [] # average validation loss per epoch
                    
                valid_balanced_acc_epoch = [] # balanced accuracy per epoch
            
                valid_f1_score_epoch = [] # f1 score per epoch
                
                total_labels_epoch = [] # all the labels per epoch
                total_prob_epoch = []  # all the prob per epoch
                
                validation_norm_all_outputs = np.zeros((1,model.loss.embedding_dim)) # all the outputs per epoch]
                    
                with torch.no_grad():
                    for batch_idx, (data_iter_val, labels_iter_val) in enumerate(validation_dataloader):
                        data_iter_val = data_iter_val.to(device) # send data to device
                        labels_iter_val = labels_iter_val.to(device).type(torch.LongTensor) # send labels to device
                        output = model((data_iter_val).float()) # forward propagation
                        output = output.to(device).type(torch.Tensor) # send output to device
                      
                        loss, _ ,logits = criterion(output,labels_iter_val)  #Calculate the loss
                        
                        validation_norm_output = F.normalize(output) # normalize output
                        
                        prob = torch.softmax(logits,dim = 1)  #Apply softmax to the logits to get the probability
                        if torch.cuda.is_available():
                            loss = loss.cpu() # send loss to cpu
                            val_loss_epoch.append(loss.item()) # loss per batch
                            
                            total_prob_epoch.append(prob.cpu().detach().numpy()) # all the prob per epoch
                            
                            total_labels_epoch.append(labels_iter_val.cpu().detach().numpy()) # all the labels per epoch
                    
                    
                            validation_norm_all_outputs = np.vstack((validation_norm_all_outputs, validation_norm_output)) # all the outputs per epoch
                    
                    total_prob_epoch = np.concatenate(total_prob_epoch) # concatenate all the prob per epoch
                    total_labels_epoch = np.concatenate(total_labels_epoch) # concatenate all the labels per epoch
                    
                    total_prob_epoch_pos = total_prob_epoch[:,1] # all the prob per epoch for positive class
                   
                    integers_labels = total_labels_epoch # convert labels to integers from one hot encoding
        
                    eer, thresh = my_functions.compute_eer(integers_labels,total_prob_epoch_pos) # compute equal error rate
                    
                    prediction_epoch = np.array([1 if total_prob_element > thresh else 0 for total_prob_element in total_prob_epoch_pos]) # prediction for specific threshold
                        
                    valid_balanced_acc_epoch.append(sklearn.metrics.balanced_accuracy_score(integers_labels,prediction_epoch)) ## acurrecy
                            
                    valid_f1_score_epoch.append(sklearn.metrics.f1_score(integers_labels,prediction_epoch,average = 'micro')) ## f1 score
        

                    validation_losses_epoch = np.average(val_loss_epoch)     # average validation loss per epoch      
                    validation_losses_total.append(validation_losses_epoch)           # all the validation loss 
                    
                    valid_balanced_acc_total.append(valid_balanced_acc_epoch) # all the validation balanced accuracy
                    
                    f1_score_validation_total.append(valid_f1_score_epoch) # all the validation f1 score
                    
                    validation_total_eer.append(eer) # all the validation eer
                    
                    valid_balanced_acc_percentage_epoch = (100. * valid_balanced_acc_epoch[0])  # average validation balanced accuracy per epoch in percentage

                    print(f"\nValidation {epoch:.0f} ,Validation Loss {validation_losses_epoch:.5f} , Validation Balanced Accuracy: {valid_balanced_acc_percentage_epoch:.2f}%")
                    print(f"Validation EER:  {100*eer:.2f}%")
                    print(f"Validation f1 Score: {100*valid_f1_score_epoch[0]:.2f}% \n")
                    
                    early_stopping(validation_losses_epoch, model,batch_size = data_iter_val.shape[0], num_features = data_iter_val.shape[1::]) # early stopping
                    print("\n")

           
            
            
            if (checkpoint_path is not None): # save model
                    torch.save(model.state_dict(), checkpoint_path.format(epoch))
    
    
    train_norm_all_outputs = train_norm_all_outputs[1::] # all the train outputs per last epoch
    validation_norm_all_outputs = validation_norm_all_outputs[1::] # all the valdation outputs per last epoch
    
    
    utils.sphere_plots.sphere_plot(train_norm_all_outputs,train_integers_labels,scale= scale, margin = margin ,title = "Train Samples",figure_path=None) 

    utils.sphere_plots.sphere_plot(validation_norm_all_outputs,integers_labels,scale= scale, margin = margin ,title = "Valdation Samples",figure_path=None) 
    
    # convert list to numpy array               
    train_losses_total       = np.array(train_losses_total).ravel()         # all the train loss
    validation_losses_total  = np.array(validation_losses_total).ravel()    # all the validation loss
    train_balanced_acc_total = np.array(train_balanced_acc_total).ravel()   # all the train balanced accuracy
    valid_balanced_acc_total = np.array(valid_balanced_acc_total).ravel()   # all the validation balanced accuracy
    f1_score_validation_total= np.array(f1_score_validation_total).ravel()  # all the validation f1 score
    train_total_eer          = np.array(train_total_eer).ravel()            #  all the train eer   
    validation_total_eer     = np.array(validation_total_eer).ravel()       # all the validation eer

    return model,train_losses_total,validation_losses_total,train_balanced_acc_total,valid_balanced_acc_total,f1_score_validation_total,validation_total_eer,train_total_eer
#################################################################################################################################################