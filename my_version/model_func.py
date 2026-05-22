# --------
# Imports
import sys
import time
import subprocess
import glob
import itertools

import torchvision
import torch.nn as nn
import torch
import pandas as pd
import tqdm
import numpy as np

from plotting_func import plot_loss
from plotting_func import plot_acc
from plotting_func import plot_conf_matrix
from plotting_func import make_classification_report
from plotting_func import plot_correlation_scatter_hist

# ---------
# model creation

# # make network model
def make_model(use_pretrained, num_features, transfer, model):

    print("use_pretrained: " + str(use_pretrained)) # hard-set to True

    if model == "VGG16": # CGG16 model
        # load VGG16 model
        #net = torchvision.models.vgg16(pretrained=use_pretrained) # pretrained=syntax dropped in torchvision 0.13
        net = torchvision.models.vgg16(weights = 'IMAGENET1K_V1')

        # change the last unit of VGG16
        net.classifier[6] = nn.Linear(in_features=4096, out_features=num_features) # last classifier layer changed

    elif model == "DenseNet121":
        # load DenseNet121 model
        #net = torchvision.models.densenet121(pretrained=use_pretrained) # pretrained syntax dropped in torchvision 0.13
        net = torchvision.models.densenet121(weights = 'IMAGENET1K_V1')

        # change the last unit of DenseNet121
        net.classifier = nn.Linear(in_features=1024, out_features=num_features) # classifier layer added

    # set to training mode
    net.train() # set model to training mode - affects behaviour of DropOut and BatchNorm layers

    params_to_update = []

    if transfer == False: # train full graph (fine-tuning) or linear-probing. Should work for both models - check why not work for VGG16.
        print("### Full learning ###")
        for name, param in net.named_parameters():
            param.requires_grad = True
            params_to_update.append(param)

    else:
        print("### Transfer learning ###")
        if model != "VGG16":
            print("Transfer leraning is available only for VGG16")
            sys.exit()
        
        
        # parameters for training
        update_param_names = ["classifier.6.weight", "classifier.6.bias"]

        for name, param in net.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
                params_to_update.append(param)
                print(name)
            else:
                param.requires_grad = False

    print("-----------")
    print(params_to_update)
    print("-----------")

    return net, params_to_update


# ---------------
# Model training/validation - could be better if split into if phase == train, do this, if phase == valid do this, or as two diff funciton calls.

# Loss function for gene prediction

def loss_function(outputs, labels):
    criterion = nn.SmoothL1Loss() # Smooth L1 Loss: squared error loss if diff between true and pred is < threshold, typically 1; mean absolute error otherwise.
    num_gene = outputs.shape[1]

    loss = 0
    
    for i in range(num_gene):
        loss += criterion(outputs[:,i], labels[:,i]) / num_gene # divide calculated loss by number of genes

    return loss

# Calculate Pearson Correlation
def calc_cor(outputs, labels):

    num_gene = outputs.shape[1]

    corR = []

    for i in range(num_gene):
        corR.append(np.corrcoef(outputs[:,i].to('cpu').detach().numpy(), labels[:,i].to('cpu').detach().numpy())[0,1])

    corR = np.array(corR)

    corR[np.isnan(corR)] = 0.0
        
    print("corR: "+str(corR))

    return np.mean(corR)





# Train model

def run_train(outDir, net, dataloaders_dict, optimizer, num_epochs, device, early_stop_max, name, ClusterPredictionMode):
    # Set to use multiple GPU:
    if str(device) != 'cpu':
        net.to(device) # move model to device
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net) # Parallelize by splitting input acorss devices by chunking given batch.
    
    # Create results df
    if ClusterPredictionMode:
        res_df = pd.DataFrame(columns=['train_loss','valid_loss','train_acc','valid_acc'])
    else:
        res_df = pd.DataFrame(columns=['train_loss','valid_loss','train_cor','valid_cor'])

    # Track time of each epoch/batch??
    time_df = pd.DataFrame(columns=['time'])

    # Initialise validation parameters across all of training
    valid_loss_prev = 1e+100; valid_loss_best = 1e+100; valid_cor_best = -1e+100;
    early_stop_count = 0

    # Epoch loop
    for epoch in range(num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')
        start = time.time()

        # Initialise loss parameters for current epoch
        train_loss = 0; valid_loss = 0;
        train_acc_or_cor = 0; valid_acc_or_cor = 0;

        for phase in ['train', 'valid']:
            net.train() if phase == 'train' else net.eval() # train or eval mode based on training or validation mode

            epoch_loss = 0.0  # sum of loss
            epoch_corrects_or_cor = 0  # sum of corrects or correlation

            # skip training if epoch == 0
            if (epoch == 0) and (phase == 'train'): continue

            # extract minibatch from dataloader
            for inputs, labels in tqdm(dataloaders_dict[phase]): # extract training or validation data
                # if GPU is avalable
                if str(device) != 'cpu': # load data/labels into device.
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                # initialize optimizer
                optimizer.zero_grad() # reset gradients.

                # forward calculation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs) # forward prop
                    
                    # calculate loss
                    if ClusterPredictionMode: # If clustering: CrossEntropyLoss (classification)
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(outputs, labels[:,0])
                        _, preds = torch.max(outputs, 1)  # predict cluster as cluster for which value is highest.
                    else:
                        loss = loss_function(outputs, labels) # output loss averaged across no of genes


                    # backpropagation if training
                    if phase == 'train': 
                        loss.backward()
                        optimizer.step()

                    # update sum of loss
                    epoch_loss += loss.item() * inputs.size(0) # multiply by number of inputs; as loss is divided by dataset size after epoch

                    # update sum of corrects, or calculate pearson correlation
                    if ClusterPredictionMode:
                        epoch_corrects_or_cor += torch.sum(preds == labels[:,0])
                    # update sum of correlation
                    else:
                        epoch_corrects_or_cor += calc_cor(outputs, labels)

            # Once finished with all training per epoch - output/calculate training and validation loss per epoch
            if ClusterPredictionMode:
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset) # loss = loss/length of dataset
                epoch_corrects_or_cor = epoch_corrects_or_cor.double() / len(dataloaders_dict[phase].dataset) # correct = num correct/length of dataset
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_corrects_or_cor))
            else:
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset) # Loss = total loss/length of dataset
                print('{} Loss: {:.4f}'.format(phase, epoch_loss)) # Print loss

            # train loss or valid loss
            if phase == 'train':
                train_acc_or_cor = float(epoch_corrects_or_cor)
                train_loss = epoch_loss # set as training loss
            else:
                valid_acc_or_cor = float(epoch_corrects_or_cor)
                valid_loss = epoch_loss # set as validation loss

        
        ### append loss to DataFrame
        res_df = res_df.append([pd.Series([train_loss,valid_loss,train_acc_or_cor,valid_acc_or_cor],index=res_df.columns)], ignore_index=True)
        
        ### save training_loss.txt (happens at end of each epoch)
        res_df.to_csv(outDir+"/training_loss_"+name+".txt", sep='\t', float_format='%.6f')

        # Save best model
        save_best = False
        if ClusterPredictionMode:
            if valid_loss_best > valid_loss: # if validation loss is lower, or validation accuracy/correlation is higher; save model.
                valid_loss_best = valid_loss
                save_best = True
        else:
            if valid_cor_best < valid_acc_or_cor:
                valid_cor_best = valid_acc_or_cor
                save_best = True

        if save_best:
            if str(device) != 'cpu' and torch.cuda.device_count() > 1: # if running on single GPU vs multiple GPUs
                subprocess.call(['rm','-r',outDir+'/model_'+name+'/']) # rm previous models
                subprocess.call(['mkdir',outDir+'/model_'+name+'/'])
                torch.save(net.module.state_dict(), outDir+"/model_"+name+"/model_"+str(epoch)+".pth") # save model
            else:
                subprocess.call(['rm','-r',outDir+'/model_'+name+'/'])
                subprocess.call(['mkdir',outDir+'/model_'+name+'/'])
                torch.save(net.state_dict(), outDir+"/model_"+name+"/model_"+str(epoch)+".pth")

        # Stop early if validation loss does not increase for early_stop_count epochs
        if valid_loss_prev > valid_loss:
            early_stop_count = 0
        else:
            early_stop_count += 1

        print("early_stop_count: "+str(early_stop_count))

        if early_stop_count == early_stop_max: break

        # Save validation loss of current as previous, to set up for new epoch
        valid_loss_prev = valid_loss

        # Calculated elapsed time and output to df
        ## elapsed time
        elapsed_time = time.time() - start
        print ("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")

        ### append loss to DataFrame (per epoch)
        time_df = time_df.append([pd.Series([elapsed_time],index=time_df.columns)], ignore_index=True)

        # save training loss to txt file (per epoch)
        ### save training_loss.txt
        time_df.to_csv(outDir+"/time_"+name+".txt", sep='\t', float_format='%.6f')

# Test model
def run_test(outDir, data_list_df, dataloaders_dict, model, device, geneSymbols, num_features, ClusterPredictionMode, name):
    print("### load loss_acc_df ###")
    loss_acc_df = pd.read_csv(outDir+"/training_loss_"+name+".txt", sep='\t') # load in training loss df??

    loss_acc_df = loss_acc_df.rename(columns={'Unnamed: 0':'no'})

    print("loss_acc_df: "+str(loss_acc_df.shape))
    loss_acc_df.head()

    # write name of best model to txt file
    print("### Best model ###")
    best_files = glob.glob(outDir+"/model_"+name+"/model_*")

    print(best_files)
    best_model = best_files[0]
        
    print("best_model: "+str(best_model))

    with open(outDir+"/best_model_"+name+".txt", mode='w') as f:
        f.write(str(best_model))

    loss_acc_df.loc[0,'train_loss'] = np.nan # set training loss of first epoch to 0

    # Plot loss diagram
    print("### Plot Loss ###")
    plot_loss(outDir, loss_acc_df, name)

    # If predicting clusters, plot training accuracy
    if ClusterPredictionMode:
        loss_acc_df.loc[0,'train_acc'] = np.nan
        
        print("### Plot Acc ###")
        plot_acc(outDir, loss_acc_df, name)

    if ClusterPredictionMode: # Load new version of model with transfer set to false (fine-tuning mode vs linear probing)
        net, params_to_update = make_model(use_pretrained=False,
                                           num_features=num_features,
                                           transfer=False,
                                           model=model)
    else:
        net, params_to_update = make_model(use_pretrained=False,
                                           num_features=len(geneSymbols),
                                           transfer=False,
                                           model=model)
        
    # Load outputted state_dict of best model (parameters):
    print("### load the best model ###")
    if str(device) != 'cpu':
        net.load_state_dict(torch.load(best_model))
    else:
        net.load_state_dict(torch.load(best_model, map_location={'cuda:0': 'cpu'}))

    
    print("### Predict test set ###")
    # set to eval mode (no parameter adjustment/training)
    net.eval()   # eval mode

    valid_preds = np.array([[]])
    valid_labels = np.array([[]])

    phase = 'valid'
    check_first = True # first iteration through

    # extract minibatch from dataloader
    for inputs, labels in tqdm(dataloaders_dict[phase]): # from test dataloader (why set up as dictionary, only one phase???????)

        # forward calculation
        with torch.set_grad_enabled(phase == 'train'):
            outputs = net(inputs)
            
            if ClusterPredictionMode:
                _, preds = torch.max(outputs, 1)  # predict labels
                valid_preds = np.append(valid_preds, preds.clone().numpy())
                valid_labels = np.append(valid_labels, labels[:,0].data.clone().numpy())

            else:
                if check_first:
                    valid_preds = outputs.clone().numpy() # if first iteration, raw values, if not, concatenate to prev results.
                    valid_labels = labels.clone().numpy()
                    check_first = False
                else:
                    valid_preds = np.concatenate([valid_preds, outputs.clone().numpy()])
                    valid_labels = np.concatenate([valid_labels, labels.clone().numpy()])

    valid_preds_df = pd.DataFrame(valid_preds)
    
    print(valid_preds_df)

    # name columns in output df
    if ClusterPredictionMode:
        valid_preds_df.columns = ["Cluster_pred"]
    else:
        valid_preds_df.columns = [s+"_pred" for s in geneSymbols]

    print("valid_preds_df: "+str(valid_preds_df.shape))
    valid_preds_df.head()

    data_list_df = data_list_df.reset_index(drop=True)
    data_list_df = pd.concat([data_list_df, valid_preds_df], axis=1) # concatenate actual values with predicted values

    if ClusterPredictionMode:
        data_list_df['Cluster_pred'] = [int(i)+1 for i in valid_preds.tolist()]        
  
        print("### Plot confusion matrix ###")
        idx = [int(i+1) != -1 for i in valid_labels]
        valid_labels = list(itertools.compress(valid_labels, idx))
        valid_preds = list(itertools.compress(valid_preds, idx))
        
        print("valid_labels: "+str(valid_labels))
        print("valid_preds: "+str(valid_preds))
        print(["Cluster"+str(i+1) for i in range(num_features)])
        
        plot_conf_matrix(outDir, valid_labels, valid_preds, ["Cluster"+str(i+1) for i in range(num_features)], name)

        print("### make classification_report ###")
        make_classification_report(outDir, valid_labels, valid_preds, ["Cluster"+str(i+1) for i in range(num_features)], name)

    else: # for gene expression, plot histogram of gene pearson corr
        print("### plot_correlation_scatter_hist ###")
        plot_correlation_scatter_hist(outDir, valid_labels, valid_preds, geneSymbols, scatter=False, name=name)

    return data_list_df, net # return df with predictions and actual labels, and model.
