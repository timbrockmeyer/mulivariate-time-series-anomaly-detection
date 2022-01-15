import torch
from time import time
from copy import deepcopy

from .utils import format_time


class Trainer:
    '''
    Class for model training, validation and testing.
    
    Args:
        model (callable): Pytorch nn.module class object that defines the neural network model.
        optimizer (callable): Pytorch optim object, e.g. Adam.
        criterion (func): Loss function that takes two arguments.
    '''
    
    def __init__(self, model, optimizer, criterion):
            
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def _train_iteration(self, loader):
        '''
        Returns the average training loss of one iteration over the training dataloader.
        '''
        self.model.train()

        avg_pred_loss = 0
        for i, window in enumerate(loader):
            self.optimizer.zero_grad()

            x = window.x
            y = window.y

            # forward step
            pred = self.model(x)

            assert pred.shape == y.shape

            loss = self.criterion(pred, y)

            # backward step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
            self.optimizer.step()

            avg_pred_loss += loss.item()
        avg_pred_loss /= i+1

        return avg_pred_loss

    def test(self, loader, return_errors=True):
        '''
        Returns the average loss over the test data.
        Optionally returns a list of predictions and corresponding groundtruth values
        and anomaly labels.
        '''
        self.model.eval()

        avg_pred_loss = 0
        pred_errors = []
        y_labels = []
        with torch.no_grad():
            for i, window in enumerate(loader):
                x = window.x
                y = window.y
                batch_size = len(window.ptr) - 1

                pred = self.model(x)

                assert pred.shape == y.shape

                pred_loss = self.criterion(pred, y)

                if return_errors:
                    y_label = window.y_label     
                    if y_label is not None:                   
                        y_labels.append(y_label[::pred.size(1)])
                    else:
                        y_labels.append(y_label) # NoneType labels for validation data

                    pred_error = ((pred[:, -1] - y[:, -1]) ** 2).detach() 
                    pred_errors.append(pred_error.T.view(batch_size, -1).T) 

                avg_pred_loss += pred_loss.item()

            avg_pred_loss /= i+1

        # results to be returned
        re = []
        if return_errors:
            pred_errors = torch.cat(pred_errors, dim=1)

            if isinstance(y_labels[0], torch.Tensor):
                anomaly_labels = torch.cat(y_labels)
            else: # during validation
                anomaly_labels = None 
            
            re.append([pred_errors, anomaly_labels])

        re.append(avg_pred_loss)
        
        if len(re) == 1:
            return re.pop()
        else:
            return tuple(re)

    def train(self, train_loader, val_loader=None, epochs=10, early_stopping=10, return_model_state=False, return_val_results=False, verbose=True):
        '''
        Main function of the Trainer class. Handles the training procedure,
        including the training and validation steps for each epoch testing
        the resulting model on the test data.

        Args:
             train_loader (iterable): Dataloader holding the (batched) training samples.
             val_loader (iterable, optional): Dataloader holding the (batched) validation samples.
             epochs (int, optional): Number of epochs for training.
             early_stopping (int, optional): Number of epochs without improvement on the validation data until training is stopped.
             return_model_state (bool, optional): If true, returns the model state dict.
             return_val_results (bool, optional): If true, returns predictions and groundtruth values for validation.
             verbose (bool, optional): If true, prints updates on training and validation loss each epoch.
        '''

        train_loss_history = []
        val_loss_history = []
        early_stopping_counter = 0
        early_stopping_point = early_stopping
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        val_results = None # dummy variable for optional return values
        indicator = ''
        for i in range(epochs):
            start = time()
            # train
            train_loss = self._train_iteration(train_loader)
            train_loss_history.append(train_loss)

            if val_loader is not None:
                # validate if validation loader is provided
                if return_val_results:
                    val_results, val_loss = self.test(val_loader, return_errors=return_val_results)
                else:
                    val_loss = self.test(val_loader, return_errors=return_val_results)
                val_loss_history.append(val_loss)
            else:
                # use training loss for early stopping if no validation data
                val_loss = train_loss
        
            # check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_results = val_results
                best_train_loss = train_loss
                best_model_state = deepcopy(self.model.state_dict())
                early_stopping_counter = 0
                indicator = '*' 
            else:
                early_stopping_counter += 1
                indicator = ''
                
            if verbose:
                # print loss of epoch
                time_elapsed = format_time(time() - start)
                train_print_string = f'Train Loss: {train_loss:>9.5f}'
                val_print_string = f' || Validation Loss: {val_loss:>9.5f}' if val_loader is not None else ''
                print(f'   Epoch {i+1:>2}/{epochs} ({time_elapsed}/it) -- ({train_print_string}{val_print_string}) {indicator}')

            # stop training if early stopping criterion is fulfilled
            if early_stopping_counter == early_stopping_point and not epochs == i+1:
                if verbose:
                    print(f'   ...Stopping early after {i+1} epochs...')
                break
            # end of epoch
        # end of training loop

        if verbose:
            # print loss after training
            print('   Training Results:')
            print(f'     Train MSE: {best_train_loss:.5f}')
            if val_loader is not None:
                print(f'     Validation MSE: {best_val_loss:.5f}\n') 

        # return values: loss for each epoch, validation results (optional), model_state_dict (optional)
        if val_loader is None:
            re = [train_loss_history, None]
            best_val_results = None
        else:
            re = [train_loss_history, val_loss_history]
        self.model.load_state_dict(best_model_state)
        if return_model_state:
            re.append(best_model_state)
        if return_val_results:
            re.append(best_val_results)

        if len(re) == 1:
            return re.pop()
        else:
            return tuple(re)
        

