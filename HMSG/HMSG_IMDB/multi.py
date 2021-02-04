import os
from test import main
import numpy as np
import subprocess

mac_f1_list = []
mic_f1_list = []
NMI_list = []
ARI_list = []
os.environ['MKL_THREADING_LAYER'] = 'GNU'
seeds = [228, 7243, 3295, 4020, 8149, 4832, 741, 6538, 7850, 6391]

for i in seeds:
    p = subprocess.Popen("python main.py --seed {}".format(i), shell=True)
    p.wait()
    p.kill()
    macro_f1_mean , micro_f1_mean, NMI, ARI = main()
    mac_f1_list.append(macro_f1_mean)
    mic_f1_list.append(micro_f1_mean)
    NMI_list.append(NMI)
    ARI_list.append(ARI)
print('\n************************** Average results *******************************')
print('Macro-F1: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(*np.mean(np.array(mac_f1_list), axis=0).tolist()))
print('Micro-F1: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(*np.mean(np.array(mic_f1_list), axis=0).tolist()))
print('NMI: {:.4f}, ARI: {:.4f}'.format(np.mean(NMI_list), np.mean(ARI_list)))
