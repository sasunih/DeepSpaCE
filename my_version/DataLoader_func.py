#---------
# Imports
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold
import torch.utils.data as data
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch

#------------
# makeDataList
def makeDataList(rootDir, sampleNames, clusteringMethod, extraSize, geneSymbols, quantileRGB, seed, cross_index, train_equals_test, is_test, rm_cluster):
    # make cluster df from spaceranger clustering
    print("### load cluster list ###")
    cluster_list = pd.DataFrame(columns=['Sample','Barcode','Cluster'] )

    for sample in sampleNames:
        tmp = pd.read_csv(rootDir+"/"+sample+"/SpaceRanger/analysis/clustering/"+clusteringMethod+"/clusters.csv")
        tmp['Sample'] = sample
        cluster_list = pd.concat([cluster_list, tmp], ignore_index=True)

    # Print head of cluster dataframe
    print("cluster_list: "+str(cluster_list.shape))
    print(cluster_list.head())

    # If remove_cluster set to cluster, remove it, otherwise leave if set to -1 (filtering system could be improved - only allows for removal of one cluster)
    print("### remove cluster ###")
    cluster_list['have_cluster'] = [True if i != rm_cluster else False for i in cluster_list['Cluster']]

    cluster_list['Cluster'] = [i if i != rm_cluster else -1 for i in cluster_list['Cluster']]
    cluster_list['Cluster'] = [i if i < rm_cluster else i - 1 for i in cluster_list['Cluster']]

    print("cluster_list: "+str(cluster_list.shape))
    print(cluster_list.head())

    print(cluster_list['Cluster'].unique())

    # Load tissue_position_list.csv (where spots are) into pd dataframe
    print("### load tissue_position_list.csv ###")
    tissue_pos = pd.DataFrame(columns=['Sample','Barcode','in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres','imageID'] )

    for sample in sampleNames:
        tmp = pd.read_csv(rootDir+"/"+sample+"/SpaceRanger/spatial/tissue_positions_list.csv", header=None)

        tmp.columns = ['Barcode','in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres']
        tmp['imageID'] = tmp.index

        tmp['Sample'] = sample
        tissue_pos = pd.concat([tissue_pos, tmp], ignore_index = True)

    print("tissue_pos: "+str(tissue_pos.shape))
    print(tissue_pos.head())

    # Merge clustering and tissue positions file into one df.
    print("### merge cluster file and tissue position file ###")
    cluster_pos_df = pd.merge(cluster_list, tissue_pos, how='inner', on=['Sample','Barcode'])

    cluster_pos_df['image_path'] = [rootDir+"/"+sample+"/CropImage/size_"+str(extraSize)+"/spot_images/spot_image_"+str(s_id).zfill(4)+".tif" for sample,s_id in zip(cluster_pos_df['Sample'].tolist(), cluster_pos_df['imageID'].tolist())]

    cluster_pos_df = cluster_pos_df.sort_values('imageID')

    cluster_pos_df.index = cluster_pos_df['imageID'].tolist()

    print("cluster_pos_df: "+str(cluster_pos_df.shape))
    print(cluster_pos_df.head())

    
    # Load expression matrix output from NormUMI (filtered SCT-transformed, log-transformed matrix)
    print("### load expression metrix ###")
    exp_mat = pd.DataFrame(columns=['Sample','Barcode']+geneSymbols)

    for sample in sampleNames:
        tmp = pd.read_csv(rootDir+"/"+sample+"/NormUMI/exp_mat_fil_SCT_log10.txt", sep='\t')

        tmp = tmp.T
        tmp.columns = tmp.iloc[0,:].tolist()
        tmp = tmp.drop("symbol", axis=0)
        tmp = tmp.loc[:,geneSymbols]

        tmp['Barcode'] = tmp.index
        tmp['Barcode'] = tmp['Barcode'].str.replace('.','-')
        tmp['Sample'] = sample

        exp_mat = pd.concat([exp_mat, tmp], ignore_index = True)
       

    print("exp_mat: "+str(exp_mat.shape))
    print(exp_mat.head())

    # Convert exp_mat df to numpy array
    print("### Min-Max scaling ###")
    exp_mat_np = exp_mat.iloc[:,range(2,exp_mat.shape[1])].to_numpy()

    # min_max scale (sklearn.preprocessing.minmax_scale) the gene expression values of each gene to between 0 and 1 (so now the different genes are at different scales compared to each other)
    for i in range(exp_mat.shape[1]-2):
        exp_mat_np[:,i] = minmax_scale(exp_mat_np[:,i].tolist())
    
    # Convert back to df
    exp_mat_np = pd.DataFrame(exp_mat_np)
    exp_mat_np = exp_mat_np.astype('float64')
    exp_mat_np.columns = exp_mat.iloc[:,range(2,exp_mat.shape[1])].columns

    exp_mat = pd.concat([exp_mat.iloc[:,0:2],exp_mat_np], axis=1)

    print(exp_mat)

    # Set all barcodes in exp_mat has have_exp = True
    exp_mat['have_exp'] = True
        
    print("exp_mat: "+str(exp_mat.shape))
    print(exp_mat.head())

    # Merge cluster file and expression matrix
    print("### merge cluster file and expression metrix ###")
    cluster_pos_df = pd.merge(cluster_pos_df, exp_mat, how='left', on=['Sample','Barcode'])

    print("cluster_pos_df: "+str(cluster_pos_df.shape))
    print(cluster_pos_df.head())

    # Read in image quality information outputted from CropImage.py, based on RGB value.
    print("### Filter image ###")
    cluster_pos_filter_df = pd.DataFrame(columns=['Sample','Barcode','ImageFilter'] )

    for sample in sampleNames:
        tmp = pd.read_csv(rootDir+"/"+sample+"/CropImage/size_"+str(extraSize)+"/RGB_"+str(quantileRGB)+"/cluster_position_filter.txt", sep='\t')
        tmp.index = tmp['imageID'].tolist()

        tmp = tmp.query('ImageFilter == "OK"').copy() # filter to only rows where RGB values were ok.

        tmp = tmp.loc[:,['Barcode','ImageFilter']]

        tmp['Sample'] = sample
        cluster_pos_filter_df = pd.concat([cluster_pos_filter_df, tmp], ignore_index = True)
        
    print("cluster_pos_filter_df: "+str(cluster_pos_filter_df.shape))
    print(cluster_pos_filter_df.head())

    # Merge image info with cluster + pos + gene info df, use inner join to remove values where RGB value ('ImageFilter') was not 'OK'.
    print("### Use only OK images ###")
    data_list_df = pd.merge(cluster_pos_df, cluster_pos_filter_df, how='inner', on=['Sample','Barcode'])

    print("data_list_df: "+str(data_list_df.shape))
    print(data_list_df.head())

    # Split data into training and validation (can be used for cross-validation, but really clunky; they have written so can be used for initial model training and cross-validation, not sure why they didn't use kfold.split)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Can use kfold.split instead???
    count = 0
    for train_index, test_index in kf.split(data_list_df.index, data_list_df.index):
        if count == cross_index: # cross_index provided by arguments into deepspace.py
            data_list_df.loc[data_list_df.index[train_index], 'phase'] = 'train'
            data_list_df.loc[data_list_df.index[test_index], 'phase'] = 'valid'

        count += 1

    if is_test: # set all data as validation?????
        data_list_df['phase'] = 'valid'

    else:
        data_list_df_train = data_list_df.query('phase == "train"').copy() # training data
        data_list_df_test = data_list_df.query('phase == "valid"').copy() # validation data

        data_list_df_train_index = data_list_df_train.query('have_exp != True').index # Drop training data without gene exp
        data_list_df_train = data_list_df_train.drop(data_list_df_train_index)

        data_list_df_train_index = data_list_df_train.query('have_cluster != True').index # Drop training data without cluster expression
        data_list_df_train = data_list_df_train.drop(data_list_df_train_index)

        if train_equals_test: # If training data is same as testing_data, change validation data to testing data
            data_list_df_test['phase'] = "test"
            # Split training data again using KFold-cross validator for training and new validation:
            count = 0
            for train_index, test_index in kf.split(data_list_df_train.index, data_list_df_train.index):
                if count == cross_index:
                    data_list_df_train.loc[data_list_df_train.index[train_index], 'phase'] = 'train'
                    data_list_df_train.loc[data_list_df_train.index[test_index], 'phase'] = 'valid'

                count += 1

        else: # If different sample provided for testing data, filter validation data by expression and cluster data
            data_list_df_test_index = data_list_df_test.query('have_exp != True').index
            data_list_df_test = data_list_df_test.drop(data_list_df_test_index)

            # Supposed to be test: changed from original code
            data_list_df_test_index = data_list_df_test.query('have_cluster != True').index
            data_list_df_test = data_list_df_test.drop(data_list_df_train_index)
            
        data_list_df = pd.concat([data_list_df_train, data_list_df_test]) # Concat training/testing or training/validation data back together.

    data_list_df = data_list_df.sort_values(['Sample','imageID'])
    data_list_df = data_list_df.reset_index(drop=True)

    print("data_list_df: "+str(data_list_df.shape))
    print(data_list_df.head())
    
    return data_list_df

# Image Transform: custom Transform class for transforming images; requires __call__ and __init__ functions (see pytorch dataloading tutorials)
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            # albumentations: image transformation library for deep learning
            # alternative: torchvision.transforms.v2
            'init': albu.Compose([ # Compose used to string multiple transformations together, use v2.Compose instead
                albu.Resize(resize, resize) # Resize image to 224 (hard set) (use v2.Resize instead
            ]),
            'end': albu.Compose([
                albu.Normalize(mean,std), # normalize image values to mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
                # could use v2.Normalize instead and then use a combinations of v2.ToImage() and v2.ToDType('float32', scale = False)
                ToTensorV2() # updated for new albumentations module
            ]),
            'flip': albu.Compose([
                albu.RandomRotate90(p=0.5), # Randomly rotate by 90° (0, 90, 180, or 270), can use torchvision.transforms.RandomRotation, but less control
                # albu.Flip(p=0.5), #deprecated
                albu.OneOf([
                    albu.VerticalFlip(p=1.0),
                    albu.HorizontalFlip(p=1.0),
                    albu.Compose([albu.VerticalFlip(p=1.0), albu.HorizontalFlip(p=1.0)])
                ], p = 0.5), # recreated behaviour; could use v2.randomapply(v2.randomchoice(v2.RandomVerticalFlip, v2.RandomHorizontalFlip, or both))
                albu.Transpose(p=0.5) # Reflect image across diagonal, can use torch.transpose instead
            ], p=1.0),
            'noise': albu.Compose([
                #albu.OneOf([
                #    albu.IAAAdditiveGaussianNoise(p=1.0), #
                #    albu.GaussNoise(p=1.0)
                #], p=1.0),  #### albu.IAAAdditiveGaussianNoise deprecated
                albu.GaussNoise(p = 1.0),
            ], p=1.0),
            'blur': albu.Compose([
                albu.OneOf([
                    albu.MotionBlur(p=1.0), # Simulate motion blur in random direction
                    albu.MedianBlur(p=1.0), # replace pixel with median in square window.
                    albu.Blur(p=1.0) # average over square kernel
                ], p=1.0),
            ], p=1.0),
            'dist': albu.Compose([
                albu.OneOf([
                    albu.OpticalDistortion(p=1.0), # Apply optical distortion (lens/camera or fisheye model)
                    albu.GridDistortion(p=1.0), # Apply grid distortion by dividing the image into cells and warping each
                    albu.PiecewiseAffine(p=1.0), # image broken up into patches, which have unique affine transformation applied to each (scaling, rotation, translation, shearing - keeps straight lines straight)
                    albu.ShiftScaleRotate(p=1.0) # randomly apply affine transformation to whole image
                ], p=1.0),
            ], p=1.0),
            'contrast': albu.Compose([
                # albu.RandomContrast deprecated
                albu.RandomBrightnessContrast(brightness_limit=0, p=0.5), # Randomly changes the brightness and contrast of the input image (set brightness to 0)
                albu.RandomGamma(p=0.5), # apply random gamma correction (luminance)
                # albu.RandomBrightness(p=0.5) deprecated
                albu.RandomBrightnessContrast(contrast_limit=0, p = 0.5) # see above
            ], p=1.0),
            'color': albu.Compose([
                albu.HueSaturationValue(p=0.5), # randomly change hue, saturation and value of image.
                albu.ChannelShuffle(p=0.5), # randomly rearrange channels of image
                albu.RGBShift(p=0.5) # apply constant uniform shift to each channel of input RGB image.
            ], p=1.0),
            'crop': albu.Compose([
                albu.RandomResizedCrop(height=resize, width=resize, scale=(0.5, 1.0), p=0.5), # Crop a random part of the input and rescale it to a specified size. (scale = size of crop relate to orig image)
            ], p=1.0),
            'random': albu.Compose([ # random selection of previous.
                albu.OneOf([
                    #albu.OneOf([
                        #albu.IAAAdditiveGaussianNoise(p=1.0), # deprecated
                    #    albu.GaussNoise(p=1.0)
                #    ], p=1.0),
                albu.GaussianNoise(p = 1.0),
                    albu.OneOf([
                        albu.MotionBlur(p=1.0),
                        albu.MedianBlur(p=1.0),
                        albu.Blur(p=1.0)
                    ], p=1.0),
                    albu.OneOf([
                        albu.OpticalDistortion(p=1.0),
                        albu.GridDistortion(p=1.0),
                        albu.PiecewiseAffine(p=1.0),
                        albu.ShiftScaleRotate(p=1.0)
                    ], p=1.0),
                ], p=1.0),            
            ], p=1.0),
            'valid': albu.Compose([
                albu.Resize(resize, resize),
                albu.CenterCrop(resize, resize),
                albu.Normalize(mean, std),
                ToTensorV2()
            ])
        }

    def __call__(self, img, phase='train', param=''):
        if phase == 'train':
            img = self.data_transform['init'](image=img) # apply init transform (resize to 224)

            if param != 'none': # apply given parameters (flip, crop, color, random by default: flip vertically horizontally, both or neither; crop image to 0.5 - whole image, and then resize to original; randomly change hue/saturation/value of image, rearrange the channels of image, or shift RGB values or random combination; randomly apply Gaussian noise, blur, or image distortion)
                param = param.split(',')
                
                for para in param:
                    img = self.data_transform[para](image=img['image'])

            img = self.data_transform['end'](image=img['image']) # ormalize image values to mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), and convert to tensor.
        elif phase == 'valid':
            img = self.data_transform['valid'](image=img) # do not apply any noise; instead resize to 224, then crop 224 x 224 using CenterCrop, normalize to given mean and std, then convert to Tensor. 
        
        return img['image']

# SpotImageDataset: custom Dataset class (store of samples and corresponding labels)
##### Requires __init___, __len__, and __getitem__ functions. See pytorch tutorials basics.
class SpotImageDataset(data.Dataset): # Input torch.utils.data.Dataset
    def __init__(self, file_list, label_df, transform=None, phase='train', param=''):
        self.file_list = file_list # list of spot image locations
        self.label_df = label_df # labels as pd dataframe
        self.transform = transform # transformation to apply to features
        self.phase = phase # training/testing phase
        self.param = param # augmentation to apply to images (from input to deepspace.py)

    def __len__(self): # Return number of samples
        return len(self.file_list)

    def __getitem__(self, index): # Return sample from the dataset at given index
        # load image of index
        img = cv2.imread(self.file_list[index]) # read using open-cv (open-source computer vision) (loads image from file path as openCV matrix, torchvision.io.decode_image unavailable in their pytorch version)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert from bgr to rgb image; can use numpy slicing instead with pytorch.
        
        # Image transform
        img_transformed = self.transform(img, self.phase, self.param) # Apply ImageTransform (see above) if train introduce noise, is valid just resize, crop and normalise.

        # label
        label = self.label_df.iloc[index,:].tolist() # label to tensor (either number for cluster, or 1d gene expression tensor.)
        
        return img_transformed, torch.tensor(label) # return img tensor and label tensor of given index.

# Load Training Data into dict???
# ## makeTrainDataloader
def makeTrainDataloader(rootDir, data_list_df, geneSymbols, size, mean, std, augmentation, batch_size, ClusterPredictionMode):
    
    # data output from makeDatalist (only training/validation data)
    data_list_df = data_list_df.reset_index(drop=True)

    print("### make dataset ###")
    if ClusterPredictionMode: #if clusterpredictionmode label = clusters; list of 0 - n else gene values for each chosen gene,
        train_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'train', 'image_path'].tolist(),
                                         label_df=data_list_df.loc[data_list_df['phase'] == 'train', ['Cluster']] - 1,
                                         transform=ImageTransform(size, mean, std),
                                         phase='train',
                                         param=augmentation)

        valid_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'valid', 'image_path'].tolist(),
                                         label_df=data_list_df.loc[data_list_df['phase'] == 'valid', ['Cluster']] - 1,
                                         transform=ImageTransform(size, mean, std),
                                         phase='valid',
                                         param=augmentation)
    else:
        train_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'train', 'image_path'].tolist(),
                                         label_df=data_list_df.loc[data_list_df['phase'] == 'train', geneSymbols],
                                         transform=ImageTransform(size, mean, std),
                                         phase='train',
                                         param=augmentation)

        valid_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'valid', 'image_path'].tolist(),
                                         label_df=data_list_df.loc[data_list_df['phase'] == 'valid', geneSymbols],
                                         transform=ImageTransform(size, mean, std),
                                         phase='valid',
                                         param=augmentation)
        
    print("### check ###")
    index = 1
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])

    # make DataLoader using pytorch funcion
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=1, pin_memory=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False)

    # Add dataloaders to dictionary
    dataloaders_dict = {"train": train_dataloader, "valid": valid_dataloader}
    
    print("### check2 ###")
    batch_iterator = print(dataloaders_dict)
    batch_iterator = iter(dataloaders_dict["train"])
    inputs, labels = next(batch_iterator)
    print(inputs.size())
    print(labels.size())
    
    return dataloaders_dict # return dictionary with dataloaders

def makeTestDataloader(rootDir, data_list_df, model, geneSymbols, size, mean, std, augmentation, batch_size, ClusterPredictionMode):
    
    # data output from makeDatalist (test data only)
    data_list_df = data_list_df.reset_index(drop=True)

    print("### make dataset ###")
    if ClusterPredictionMode: #if clusterpredictionmode label = clusters; list of 0 - n else gene values for each chosen gene,
        test_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'valid', 'image_path'].tolist(),
                                        label_df=data_list_df.loc[data_list_df['phase'] == 'valid', ['Cluster']] - 1,
                                        transform=ImageTransform(size, mean, std),
                                        phase='valid',
                                        param=augmentation)
    else:
        test_dataset = SpotImageDataset(file_list=data_list_df.loc[data_list_df['phase'] == 'valid', 'image_path'].tolist(),
                                        label_df=data_list_df.loc[data_list_df['phase'] == 'valid', geneSymbols],
                                        transform=ImageTransform(size, mean, std),
                                        phase='valid',
                                        param=augmentation)
        
    print("### check ###")
    index = 1
    print(test_dataset.__getitem__(index)[0].size())
    print(test_dataset.__getitem__(index)[1])

    # make Dataloader using pytorch function, and put into dictionary?
    print("### make DataLoader ###")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False)

    print("### make dictionary ###")
    dataloaders_dict_test = {"valid": test_dataloader}

    print("### check ###")
    batch_iterator = iter(dataloaders_dict_test["valid"])
    inputs, labels = next(batch_iterator)
    print(inputs.shape)
    print(labels.shape)
    
    return dataloaders_dict_test # return test dataloader in dictionary
