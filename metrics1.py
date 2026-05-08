from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, precision_score, recall_score,accuracy_score,f1_score,matthews_corrcoef


precision = []
recall = []
def calculate_metric(file_name):
    ACC = []
    Precision = []
    Recall = []
    Fscore1 = []
    MCC = []  # 新增MCC列表
    for i in range(5):
        files = f'glu_results_hebing/{file_name}/{file_name}_result_{i+1}.csv'
        dataset = pd.read_csv(files, skiprows=1)
        y_test = dataset.iloc[:, 1]
        y_score = dataset.iloc[:, 0]
        y_score = np.around(y_score, 0).astype(int)
        acc = accuracy_score(y_test, y_score)
        ACC.append(acc)
        precision = precision_score(y_test, y_score)
        Precision.append(precision)
        recall = recall_score(y_test, y_score)
        Recall.append(recall)
        fscore = f1_score(y_test, y_score)
        Fscore1.append(fscore)
        mcc = matthews_corrcoef(y_test, y_score)  # 计算MCC
        MCC.append(mcc)
    
    print("ACC: %.4f " %np.mean(ACC))
    print("Precision: %.4f " %np.mean(Precision))
    print("Recall: %.4f " %np.mean(Recall))
    print("fscore1: %.4f " %np.mean(Fscore1))
    print("MCC: %.4f " %np.mean(MCC))  # 打印MCC
    
    return np.mean(ACC), np.mean(Precision), np.mean(Recall), np.mean(Fscore1), np.mean(MCC)

#if __name__ == '__main__':
    # calculate_metric('AGO1')
    # calculate_metric('AGO2')
    # calculate_metric('AGO3')
    # calculate_metric('ALKBH5')
    # calculate_metric('AUF1')
    # calculate_metric('C17ORF85')
    # calculate_metric('C22ORF28')
    # calculate_metric('CAPRIN1')
    # calculate_metric('DGCR8')
    # calculate_metric('EIF4A3')
    # calculate_metric('EWSR1')
    # calculate_metric('FMRP')
    #calculate_metric('FOX2')
    # calculate_metric('FUS')
    # calculate_metric('FXR1')
    # calculate_metric('FXR2')
    # calculate_metric('HNRNPC')
    # calculate_metric('HUR')
    # calculate_metric('IGF2BP1')
    # calculate_metric('IGF2BP2')
    # calculate_metric('IGF2BP3')
    # calculate_metric('LIN28A')
    # calculate_metric('LIN28B')
    # calculate_metric('METTL3')
    # calculate_metric('MOV10')
    # calculate_metric('PTB')
    # calculate_metric('PUM2')
    # calculate_metric('QKI')
    # calculate_metric('SFRS1')
    # calculate_metric('TAF15')
    # calculate_metric('TDP43')
    # calculate_metric('TIA1')
    # calculate_metric('TIAL1')
    # calculate_metric('TNRC6')
    # calculate_metric('U2AF65')
    #calculate_metric('WTAP')
    #calculate_metric('ZC3H7B')
