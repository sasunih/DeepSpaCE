#--------------
#  Imports
import argparse
import os
import pickle

import torch
import numpy as np
import random
import torch.optim as optim

from DataLoader_func import makeDataList
from DataLoader_func import makeTrainDataloader
from DataLoader_func import makeTestDataloader
from model_func import make_model
from model_func import run_train
from model_func import run_test

#---------------
# Setup

# Hard-set values
size = 224
print("size: "+str(size))

mean = (0.485, 0.456, 0.406)
print("mean: "+str(mean))

std = (0.229, 0.224, 0.225)
print("std: "+str(std))

#----------------
# Parse Inputs
argrequired = False

parser = argparse.ArgumentParser(description='DeepSpaCE')

parser.add_argument('--dataDir', type=str, default='/home/'+os.environ['USER']+'/DeepSpaCE/data', required=argrequired,
                    help='Data directory (default: '+'/home/'+os.environ['USER']+'/DeepSpaCE/data'+')')

parser.add_argument('--outDir', type=str, default='/home/'+os.environ['USER']+'/DeepSpaCE/',
                    help='Root directory (default: '+'/home/'+os.environ['USER']+'/DeepSpaCE/'+')')

parser.add_argument('--sampleNames_train', type=str, default='Human_Breast_Cancer_Block_A_Section_1',
                    help='Sample names to train (default: Human_Breast_Cancer_Block_A_Section_1)')

parser.add_argument('--sampleNames_test', type=str, default='Human_Breast_Cancer_Block_A_Section_1',
                    help='Sample names to test (default: Human_Breast_Cancer_Block_A_Section_1)')

parser.add_argument('--sampleNames_semi', type=str, default='None',
                    help='Sample names to semi-supervised learning (default: None)')

parser.add_argument('--semi_option', type=str, choices=['normal', 'random', 'permutation'], default='normal',
                    help='Option of semi-supervised learning (default: normal)') ## Check what this does

parser.add_argument('--seed', type=int, default=0,
                    help='Random seed (default: 0)') # seed for reproducibility

parser.add_argument('--threads', type=int, default=8,
                    help='Number of CPU threads (default: 8)')

parser.add_argument('--GPUs', type=int, default=1,
                    help='Number of GPUs (default: 1)') 

parser.add_argument('--cuda', action='store_true',
                    help='Enables CUDA training')

parser.add_argument('--transfer', action='store_true',
                    help='Enables transfer training') # Transfer training: Must be TRUE for VGG16, must be FALSE for DenseNet121

parser.add_argument('--model', type=str, choices=['VGG16','DenseNet121'], default='DenseNet121',
                    help='Deep learning model') # Model options = DenseNet121 or VGG16

parser.add_argument('--batch_size', type=int, default=128,
                    help='Input batch size for training (default: 128)')

parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4)') # Step size

parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (default: 1e-4)') # What does weight decay do?

parser.add_argument('--clusteringMethod', type=str, choices=['graphclust', 'kmeans_2_clusters', 'kmeans_3_clusters', 'kmeans_4_clusters', 'kmeans_5_clusters', 'kmeans_6_clusters', 'kmeans_7_clusters','kmeans_8_clusters', 'kmeans_9_clusters', 'kmeans_10_clusters'], default='graphclust',
                    help='Clustering method (default: graphclust)') # Clustering method to extract from data

parser.add_argument('--extraSize', type=int, default=150,
                    help='Extra image size (default: 150)') # Extra percentage to get around spots (set in CropImage.py)

parser.add_argument('--quantileRGB', type=int, default=80,
                    help='Threshold of quantile RGB (default: 80)') # RGB Threshold (set in CropImage.py)

parser.add_argument('--augmentation', type=str, default='flip,crop,color,random',
                    help='Image augmentation methods (default: flip,crop,color,random)') # Don't know what this does

parser.add_argument('--early_stop_max', type=int, default=5,
                    help='How many epochs to wait for loss improvement (default: 5)') # Stop after how many epochs with no improvement

parser.add_argument('--rm_cluster', type=str, default='-1',
                    help='Remove cluster name (default: None)') # Which clusters to remove if you want

parser.add_argument('--ClusterPredictionMode', action='store_true',
                    help='Enables ClusterPredictionMode') # Predict genes or clusters

parser.add_argument('--cross_index', type=int, default=0,
                    help='Index of 5-fold cross-validation (default: 0)') # Cross-validation (default set to 0)

parser.add_argument('--geneSymbols', type=str, default='ESR1,ERBB2,MKI67',
                    help='Gene symbols (default: ESR1,ERBB2,MKI67)')

args = parser.parse_args()

# Print arguments
print(args)

# Put args into variables
dataDir = args.dataDir
print("dataDir: "+str(dataDir))

outDir = args.outDir
print("outDir: "+str(outDir))

batch_size = args.batch_size * args.GPUs
print("batch_size: "+str(batch_size))

num_epochs = args.num_epochs
print("num_epochs: "+str(num_epochs))

lr = args.lr
print("lr: "+str(lr))

weight_decay = args.weight_decay
print("weight_decay: "+str(weight_decay))

model = args.model
print("model: "+str(model))

clusteringMethod = args.clusteringMethod
print("clusteringMethod: "+str(clusteringMethod))

cuda = args.cuda and torch.cuda.is_available()
print("cuda: "+str(cuda))

transfer = args.transfer
print("transfer: "+str(transfer))

quantileRGB = args.quantileRGB
print("quantileRGB: "+str(quantileRGB))

seed = args.seed
print("seed: "+str(seed))

threads = args.threads
print("threads: "+str(threads))

early_stop_max = args.early_stop_max
print("early_stop_max: "+str(early_stop_max))

extraSize = args.extraSize
print("extraSize: "+str(extraSize))

augmentation = args.augmentation
print("augmentation: "+augmentation)

semi_option = args.semi_option
print("semi_option: "+str(semi_option))

cross_index = args.cross_index
print("cross_index: "+str(cross_index))

ClusterPredictionMode = args.ClusterPredictionMode
print("ClusterPredictionMode: "+str(ClusterPredictionMode))



if args.rm_cluster == 'None':
    rm_cluster = -1
else:
    rm_cluster = int(args.rm_cluster)

print("rm_cluster: "+str(rm_cluster))



sampleNames_train = args.sampleNames_train.split(',')
print("sampleNames_train: "+str(sampleNames_train))



sampleNames_test = args.sampleNames_test.split(',')
print("sampleNames_test: "+str(sampleNames_test))

if sampleNames_train == sampleNames_test: # Is same sample provided for training and testing?
    train_equals_test = True
else:
    train_equals_test = False

print("train_equals_test: "+str(train_equals_test))



sampleNames_semi = args.sampleNames_semi.split(',')
print("sampleNames_semi: "+str(sampleNames_semi))



geneSymbols = args.geneSymbols.split(',')
print(geneSymbols)

# Make output directory
print("### Set seeds ###")
os.makedirs(outDir, exist_ok=True)

# Set seed for reproducibility (0 by default)
print("### Set seeds ###")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Set CPU threads
torch.set_num_threads(threads)

# Torch backends
# cuDNN = GPU acceleration primitives library by NVIDIA for deep neural networks
torch.backends.cudnn.deterministic = True #(Use deterministic algorithms for reproducibility)
torch.use_deterministic_algorithms(True, warn_only = True) # Replaced with modern alternative (my code)
torch.backends.cudnn.benchmark = False # Benchmark convolutional algorithms and use fastest (NOT deterministic) - need to check ifit works with Rocm

# Check GPU availability  
print("### Check GPU availability ###")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# ---------------------------------
## Data Importation

# Make Data Lists
# # make data_list (teacher)


data_list_teacher = makeDataList(rootDir=dataDir,
                                 sampleNames=sampleNames_train,
                                 clusteringMethod=clusteringMethod,
                                 extraSize=extraSize,
                                 geneSymbols=geneSymbols,
                                 quantileRGB=quantileRGB,
                                 seed=seed,
                                 cross_index=cross_index,
                                 train_equals_test=train_equals_test,
                                 is_test=False,
                                 rm_cluster=rm_cluster) # Creates data frame of all training/validation data, or all training/validation/testing data (if training and testing sample is the same), returns pd.DataFrame

# if training and testing sample are same, filter out data set aside for testing
if train_equals_test:
    data_list_teacher_tmp = data_list_teacher.copy()
    data_list_teacher = data_list_teacher_tmp.query('phase != "test"').copy()

# Export to csv
data_list_teacher.to_csv(outDir+"/data_list_teacher.txt", index=False, sep='\t', float_format='%.6f')

print("data_list_teacher: "+str(data_list_teacher.shape))
data_list_teacher.head()

# Load training data into dictionary with training dataloader and validation dataloader
dataloaders_dict_teacher = makeTrainDataloader(rootDir=dataDir,
                                               data_list_df=data_list_teacher,
                                               geneSymbols=geneSymbols,
                                               size=size,
                                               mean=mean,
                                               std=std,
                                               augmentation=augmentation,
                                               batch_size=batch_size,
                                               ClusterPredictionMode=ClusterPredictionMode)

# save dataloader_dist to pkl file
print("### save dataloader ###")
with open(outDir+"/dataloaders_dict_teacher.pickle", mode='wb') as f:
    pickle.dump(dataloaders_dict_teacher, f)

#-------------------------
# Model Creation
#  Create actual model - VGG16 or DenseNet121
print("### make model ###")
if ClusterPredictionMode:
    net, params_to_update = make_model(use_pretrained=True,
                                       num_features=len(data_list_teacher['Cluster'].unique()),
                                       transfer=transfer,
                                       model=model)
else:
    net, params_to_update = make_model(use_pretrained=True,
                                       num_features=len(geneSymbols),
                                       transfer=transfer,
                                       model=model)
    

# ------------------------
# Set optimization function 
optimizer = optim.Adam(params=params_to_update, lr=lr, weight_decay=weight_decay) # Adam optimisation algorithm, lr and weight decay set from DeepSpaCE.py inputs

# -------------------------
# Begin training and validation
print("### run train ###")
run_train(outDir=outDir,
          net=net,
          dataloaders_dict=dataloaders_dict_teacher,
          optimizer=optimizer,
          num_epochs=num_epochs,
          device=device,
          early_stop_max=early_stop_max,
          ClusterPredictionMode=ClusterPredictionMode,
          name='teacher')

# ------------------------------------
# Testing

# Load data for testing
data_list_test = makeDataList(rootDir=dataDir,
                              sampleNames=sampleNames_test,
                              clusteringMethod=clusteringMethod,
                              extraSize=extraSize,
                              geneSymbols=geneSymbols,
                              quantileRGB=quantileRGB,
                              seed=seed,
                              cross_index=cross_index,
                              train_equals_test=True,
                              is_test=True,
                              rm_cluster=rm_cluster) # Load in data from sample provided as test; unnecessary if train = test

if train_equals_test: # if training sample is same as testing sample
    data_list_test = data_list_teacher_tmp.query('phase == "test"').copy() # use output from previous training data call, and filter for test samples

data_list_test['phase'] = 'valid' # set phase = valid
data_list_test.to_csv(outDir+"/data_list_test.txt", index=False, sep='\t', float_format='%.6f') # export to csv

print("data_list_test: "+str(data_list_test.shape))
data_list_test.head()


dataloaders_dict_test = makeTestDataloader(rootDir=dataDir,
                                           data_list_df=data_list_test,
                                           model=model,
                                           geneSymbols=geneSymbols,
                                           size=size,
                                           mean=mean,
                                           std=std,
                                           augmentation=augmentation,
                                           batch_size=batch_size,
                                           ClusterPredictionMode=ClusterPredictionMode)

# Begin testing
### Test ###
data_list_test_teacher, net_best = run_test(outDir=outDir,
                                            data_list_df=data_list_test,
                                            dataloaders_dict=dataloaders_dict_test,
                                            model=model,
                                            device=device,
                                            geneSymbols=geneSymbols,
                                            num_features=len(data_list_teacher['Cluster'].unique()),
                                            ClusterPredictionMode=ClusterPredictionMode,
                                            name="teacher")

# Output test labels: actual and predicted to csv
data_list_test_teacher.to_csv(outDir+"/data_list_test_teacher.txt", index=False, sep='\t', float_format='%.6f')

print("data_list_test_teacher: "+str(data_list_test_teacher.shape))
data_list_test_teacher.head()


# # Semi-supervised

