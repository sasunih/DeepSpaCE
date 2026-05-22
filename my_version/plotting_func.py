# ---------
# Imports
import math

import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale

# ------------
# Plot loss diagram - training and validation
def plot_loss(outDir, loss_acc_df, name):
    fig = plt.figure()

    plt.scatter(loss_acc_df.index, loss_acc_df['train_loss'], label="train")
    plt.scatter(loss_acc_df.index, loss_acc_df['valid_loss'], label="valid")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.tick_params(labelsize=12)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.3, fontsize=15)

    plt.yscale('log')

    plt.xticks(np.arange(0, math.ceil((loss_acc_df.shape[0]-1)/50) * 50 + 1, math.ceil((loss_acc_df.shape[0]-1)/50) * 10))

    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.16, top=0.9)

    fig.savefig(outDir+"/loss_plot_"+name+".png")

    plt.close()

# ----------------
# Plot training accuracy score for cluster prediction
def plot_acc(outDir, loss_acc_df, name):
    fig = plt.figure()

    plt.scatter(loss_acc_df.index, loss_acc_df['train_acc'], label="train")
    plt.scatter(loss_acc_df.index, loss_acc_df['valid_acc'], label="valid")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.tick_params(labelsize=12)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.3, fontsize=15)

    plt.xticks(np.arange(0, math.ceil((loss_acc_df.shape[0]-1)/50) * 50 + 1, math.ceil((loss_acc_df.shape[0]-1)/50) * 10))

    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.16, top=0.9)

    fig.savefig(outDir+"/acc_plot_"+name+".png")

    plt.close()

# ------------------
# Plot confusion matrix for cluster prediction
### Plot confusion matrix ###
def plot_conf_matrix(outDir, valid_labels, valid_preds, class_names, name):
    
    conf_mat = confusion_matrix(y_target=valid_labels, y_predicted=valid_preds)

    plt.figure()

    plt.rcParams["font.size"] = 15

    fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
                                    colorbar=False,
                                    show_absolute=True,
                                    show_normed=True,
                                    class_names=class_names,
                                    cmap=plt.cm.Blues,
                                    figsize=(10,10))

    plt.xlabel("Predicted label", fontsize=25)
    plt.ylabel("True label", fontsize=25)
    plt.tick_params(labelsize=12)

    plt.ylim(conf_mat.shape[0]-0.5,-0.5)

    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.16, top=0.9)
    plt.savefig(outDir+"/confusion_matrix_plot_"+name+".png")

    #plt.show()
    plt.close()

# -----------------------
# make classification report for cluster prediction
### make classification_report ###
def make_classification_report(outDir, valid_labels, valid_preds, class_names, name):
    
    report_df = classification_report(valid_labels, valid_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_df)
    report_df = report_df.T
    report_df['Cluster'] = report_df.index
    report_df = report_df.loc[:,['Cluster','precision','recall','f1-score','support']]
    report_df['support'] = report_df['support'].astype(np.int64)

    report_df.to_csv(outDir+"/classification_report_"+name+".txt", index=False, sep='\t', float_format='%.6f')

    report_df

# -----------------------
# plot histogram of pearson correlation for gene prediction
def plot_correlation_scatter_hist(outDir, valid_labels, valid_preds, geneSymbols, scatter, name):
    corR = []
    
    corR_df = pd.DataFrame(columns=['geneSymbols','corR','RMSE','RMSE_MinMax'] )

    
    for i in range(len(geneSymbols)):
        
        idx = np.isnan(valid_labels[:,i])
        lab = valid_labels[~idx,i]
        pred = valid_preds[~idx,i]

        corR.append(np.corrcoef(lab, pred)[0,1])
        corR_tmp = np.corrcoef(lab, pred)[0,1]
        rmse = np.sqrt(mean_squared_error(lab, pred))
        rmse_minmax = np.sqrt(mean_squared_error(lab, minmax_scale(pred)))
                              
        corR_df = corR_df.append(pd.Series([geneSymbols[i],corR_tmp,rmse,rmse_minmax], index=['geneSymbols','corR','RMSE','RMSE_MinMax']), ignore_index=True)

                              
    corR_df.to_csv(outDir+"/corR_"+name+".txt", index=False, sep='\t', float_format='%.6f')


    with open(outDir+"/correlation_"+name+".txt", mode='w') as f:
        for i in range(len(geneSymbols)):
            f.write(geneSymbols[i]+"\t"+'{:.3f}'.format(corR[i])+"\n")

    ### scatter plot
    if scatter == True:
        for i in range(len(geneSymbols)):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            idx = np.isnan(valid_labels[:,i])
            lab = valid_labels[~idx,i]
            pred = valid_preds[~idx,i]

            
#            plt.scatter(valid_labels[:,i], valid_preds[:,i], label="test")
            plt.scatter(lab, pred, label="test")
            plt.xlabel("log10(UMIcount+1)", fontsize=20)
            plt.ylabel("Predicted expression", fontsize=20)
            plt.tick_params(labelsize=12)

            plt.subplots_adjust(left=0.16, right=0.9, bottom=0.16, top=0.9)

            plt.text(0.025, 0.9, "r="+'{:.3f}'.format(corR[i]), size = 20, color = "black", transform=ax.transAxes)

            fig.savefig(outDir+"/plot/scatter_plot_"+geneSymbols[i]+"_"+name+".png")

            plt.close()

    ### Hist
    fig = plt.figure()

    plt.hist(corR, bins=20)
    plt.xlim([0,1])

    plt.xlabel("Pearson correlation coefficient", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)

    fig.savefig(outDir+"/correlation_"+name+".png")
    plt.close()

    ### Hist2
    fig = plt.figure()

    plt.hist(corR, bins=20)
    plt.xlim([-1,1])

    plt.xlabel("Pearson correlation coefficient", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)

    fig.savefig(outDir+"/correlation2_"+name+".png")
    plt.close()