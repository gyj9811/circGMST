import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  #计算roc和auc

plt.rc('font',family='Times New Roman')

def calculate_AUC(file_name, color, linewidth):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(5):
        files = f'glu_results_hebing/{file_name}/{file_name}_result_{i+1}.csv'
        dataset=pd.read_csv(files, skiprows=1)
        y_test = dataset.iloc[:, 1]
        y_score = dataset.iloc[:, 0]
        fpr, tpr, thre = roc_curve(y_test,y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs = auc(fpr, tpr)
        j = str(i)
        
        #print(file_name + j + '_AUC:'+'%0.4f'%aucs)
        #return aucs
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  #计算auc的值，就是roc曲线下的面积

    print(file_name+'_AUC:'+'%0.4f'%mean_auc)
    # plt.plot(mean_fpr,mean_tpr,color=color,linewidth=linewidth, label=file_name + ' (AUC = ' + '%0.4f'%mean_auc + ')')
    return mean_auc

if __name__ == '__main__':
    #plt.figure(figsize=(10, 8))
    calculate_AUC('AGO1', 'sienna',  1)
    calculate_AUC('AGO2', 'darkblue',1)
    calculate_AUC('AGO3', 'purple', 1)
    calculate_AUC('ALKBH5', 'red', 1)
    calculate_AUC('AUF1', 'aqua',  1)
    calculate_AUC('C17ORF85', 'blanchedalmond', 1)
    calculate_AUC('C22ORF28', 'palegreen', 1)
    calculate_AUC('CAPRIN1', 'aquamarine', 1)
    calculate_AUC('DGCR8', 'indigo',1)
    calculate_AUC('EIF4A3', 'fuchsia',1)
    calculate_AUC('EWSR1', 'orange',  1)
    calculate_AUC('FMRP', 'burlywood',  1)
    calculate_AUC('FOX2', 'magenta', 1)
    calculate_AUC('FUS', 'grey',  1)
    calculate_AUC('FXR1', 'steelblue',  1)
    calculate_AUC('FXR2', 'lightgreen',  1)
    calculate_AUC('HNRNPC', 'tan',  1)
    calculate_AUC('HUR', 'lavenderblush',1)
    calculate_AUC('IGF2BP1', 'black',1)
    calculate_AUC('IGF2BP2', 'blueviolet',1)
    calculate_AUC('IGF2BP3', 'cyan',1)
    calculate_AUC('LIN28A', 'cadetblue',  1)
    calculate_AUC('LIN28B', 'forestgreen',  1)
    calculate_AUC('METTL3', 'chartreuse',  1)
    calculate_AUC('MOV10', 'chocolate',  1)
    calculate_AUC('PTB', 'coral',  1)
    calculate_AUC('PUM2', 'cornflowerblue',  1)
    calculate_AUC('QKI', 'blue',  1)
    calculate_AUC('SFRS1', 'darkred',  1)
    calculate_AUC('TAF15', 'mediumblue',1)
    calculate_AUC('TDP43', 'hotpink',  1)
    calculate_AUC('TIA1', 'seagreen',  1)
    calculate_AUC('TIAL1', 'olive',  1)
    calculate_AUC('TNRC6', 'dimgray',  1)
    calculate_AUC('U2AF65', 'darksalmon',  1)
    calculate_AUC('WTAP', 'green',1)
    calculate_AUC('ZC3H7B', 'peru',1)

    #normal | bold | bolder | lighter
    # plt.plot([0, 1], [0, 1], linestyle='--')
    # plt.tick_params(axis='both', direction='out')
    # plt.xlim([-0.01, 1.01])
    # plt.ylim([-0.01, 1.01])
    # plt.xticks(fontsize=12, fontweight=500)  # 默认字体大小为10
    # plt.yticks(fontsize=12, fontweight=500)
    # plt.xlabel('FPR', fontsize=12, fontweight='normal')
    # plt.ylabel('TPR', fontsize=12, fontweight='normal')
    # plt.title('ROC Curve', fontsize=12, fontweight='normal')
    # plt.legend(loc='lower right', ncol = 2, columnspacing=0.3, labelspacing=0.1,  handletextpad=0.2)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext,fontsize=5, fontweight=1000)  # 设置图例字体的大小和粗细
    # #plt.figure(figsize=(10,8))
    # plt.savefig("roc.jpg", dpi = 600)
