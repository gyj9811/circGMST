import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

plt.rc('font', family='Times New Roman')

def calculate_AUPR(file_name, color, linewidth):
    y_real = []
    y_pred = []
    auprs = []
    for i in range(5):
        files = f'glu_results_hebing/{file_name}/{file_name}_result_{i+1}.csv'
        dataset = pd.read_csv(files, skiprows=1)
        y_test = dataset.iloc[:, 1]
        y_score = dataset.iloc[:, 0]
        y_real.append(y_test)
        y_pred.append(y_score)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        ap = auc(recall, precision)
        j = str(i)
        
        # print(file_name + j + 'AP:'+'%0.4f'%ap)
        # return ap
        auprs.append(auc(recall, precision))
    y_real = np.concatenate(y_real)
    y_pred = np.concatenate(y_pred)
    precision, recall, thresholds = precision_recall_curve(y_real, y_pred)
    mean_aupr = np.mean(auprs)
    # plt.plot(recall, precision, color=color,linewidth=linewidth, label=file_name + ' (AUPR = '+ '%0.4f'%mean_aupr+')')
    print(file_name+ '_AUPR:' +'%0.4f' %mean_aupr)
    return mean_aupr

if __name__ == '__main__':
    #plt.figure(figsize=(10, 8))
    calculate_AUPR('AGO1', 'sienna',  1)
    calculate_AUPR('AGO2', 'darkblue',1)
    calculate_AUPR('AGO3', 'purple', 1)
    calculate_AUPR('ALKBH5', 'red', 1)
    calculate_AUPR('AUF1', 'aqua',  1)
    calculate_AUPR('C17ORF85', 'blanchedalmond', 1)
    calculate_AUPR('C22ORF28', 'palegreen', 1)
    calculate_AUPR('CAPRIN1', 'aquamarine', 1)
    calculate_AUPR('DGCR8', 'indigo',1)
    calculate_AUPR('EIF4A3', 'fuchsia',1)
    calculate_AUPR('EWSR1', 'orange',  1)
    calculate_AUPR('FMRP', 'burlywood',  1)
    calculate_AUPR('FOX2', 'magenta', 1)
    calculate_AUPR('FUS', 'grey',  1)
    calculate_AUPR('FXR1', 'steelblue',  1)
    calculate_AUPR('FXR2', 'lightgreen',  1)
    calculate_AUPR('HNRNPC', 'tan',  1)
    calculate_AUPR('HUR', 'lavenderblush',1)
    calculate_AUPR('IGF2BP1', 'black',1)
    calculate_AUPR('IGF2BP2', 'blueviolet',1)
    calculate_AUPR('IGF2BP3', 'cyan',1)
    calculate_AUPR('LIN28A', 'cadetblue',  1)
    calculate_AUPR('LIN28B', 'forestgreen',  1)
    calculate_AUPR('METTL3', 'chartreuse',  1)
    calculate_AUPR('MOV10', 'chocolate',  1)
    calculate_AUPR('PTB', 'coral',  1)
    calculate_AUPR('PUM2', 'cornflowerblue',  1)
    calculate_AUPR('QKI', 'blue',  1)
    calculate_AUPR('SFRS1', 'darkred',  1)
    calculate_AUPR('TAF15', 'mediumblue',1)
    calculate_AUPR('TDP43', 'hotpink',  1)
    calculate_AUPR('TIA1', 'seagreen',  1)
    calculate_AUPR('TIAL1', 'olive',  1)
    calculate_AUPR('TNRC6', 'dimgray',  1)
    calculate_AUPR('U2AF65', 'darksalmon',  1)
    calculate_AUPR('WTAP', 'green',1)
    calculate_AUPR('ZC3H7B', 'peru',1)

#     plt.plot([0, 1], [0, 1], linestyle='--')
#     plt.tick_params(axis='both', direction='out')
#     plt.xlim([-0.01, 1.01])
#     plt.ylim([-0.01, 1.01])

#     plt.xticks(fontsize=12, fontweight=500)  # 默认字体大小为10
#     plt.yticks(fontsize=12, fontweight=500)
#     plt.xlabel('Recall', fontsize=12, fontweight='normal')
#     plt.ylabel('Precision', fontsize=12, fontweight='normal')
#     plt.title('Precision/Recall Curve', fontsize=12, fontweight='normal')
#     #plt.legend(loc = 'lower left',  ncol = 2)
#     plt.legend(loc='lower left', ncol = 2, columnspacing=0.3, labelspacing=0.1,  handletextpad=0.2)
#     leg = plt.gca().get_legend()
#     ltext = leg.get_texts()
#     plt.setp(ltext, fontsize=5, fontweight=1000)  # 设置图例字体的大小和粗细
#     plt.savefig("PR.jpg", dpi = 600)

