import numpy as np
import torch
import pickle

# This class is used to create EarlyStopping object that can be used in the training loop
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.onnx', trace_func=print, onnx = False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.onnx = onnx
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        if self.onnx == False and self.path == 'checkpoint.onnx':
            self.path = 'checkpoint.pkl'
        
    def __call__(self, val_loss, model,batch_size = 128, num_features = 160,num_epoch = None):

        score = -val_loss
        

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,batch_size,num_features,num_epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,batch_size,num_features,num_epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,batch_size = 128, num_features = 160,num_epoch = None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if num_epoch is not None:
                self.trace_func(f'num epoch: {num_epoch} ,Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)
        # Export the model to ONNX
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(batch_size, num_features, requires_grad=True).to(device)
        if self.onnx == True:
            if self.path == 'checkpoint.pkl':
                self.path = 'checkpoint.onnx'
            torch.onnx.export(model,               # model being run
                                x,                         # model input (or a tuple for multiple inputs)
                                self.path,   # where to save the model (can be a file or file-like object)
                                export_params=True,        # store the trained parameter weights inside the model file
                                opset_version=10,          # the ONNX version to export the model to
                                do_constant_folding=True,  # whether to execute constant folding for optimization
                                input_names = ['input'],   # the model's input names
                                output_names = ['output'], # the model's output names
                                verbose = self.verbose
            )
        else:
            if self.path == 'checkpoint.onnx':
                self.path = 'checkpoint.pkl'
            with open(self.path, 'wb') as f:
                pickle.dump(model, f)
        self.val_loss_min = val_loss
