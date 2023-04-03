import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Utility for frequently used operation

# reset weights for model
def weights_reset(model) -> None:
    reset_params = getattr(model, 'reset_parameters', None)
    if callable(reset_params):
        model.reset_parameters()


# plot loss curve for a training progress
def plot_loss_curve(loss_list, model_name, fold) -> None:
    # plot the trend of loss
    plt.plot(loss_list)
    plt.title('Loss versus Epochs for ' + model_name + '_fold' + str(fold))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    # save the figures
    plt.savefig('./figures/loss/{}_fold{}.png'.format(model_name, str(fold)), dpi=600)
    plt.show()

# get classification report for model
def get_classification_report(true, pred, model_name, fold):
    report = classification_report(
        true, pred, target_names=['Normal', 'AF',  'Non-AF', 'noisy recording'], output_dict=True)
    print(report)
    tmp = pd.DataFrame(report).transpose()
    tmp.to_csv('./report/{}_fold{}_result.csv'.format(model_name, str(fold)), index=True)

    return report


# get confusion matrix and plot the heatmap
def get_confusion_matrix(true, pred, model_name, fold):
    # use confusion matrix to show the prediction result directly
    conf_matrix = confusion_matrix(true, pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.0f')
    plt.xlabel('predictions')
    plt.ylabel('test labels')
    plt.title('Confusion Matrix for ' + model_name + '_fold' + str(fold))
    # automatically save the figures
    plt.savefig('./figures/confusion_matrix/{}_fold{}.png'.format(model_name, str(fold)), dpi=600)
    plt.show()
    return conf_matrix
