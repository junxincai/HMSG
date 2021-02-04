import os
import subprocess
import numpy as np

AUC = []
AP = []

os.environ['MKL_THREADING_LAYER'] = 'GNU'
seeds = [228, 724, 3295, 4020, 8149, 483, 741, 6538, 7850, 6391]
for i in seeds:
    p = subprocess.Popen("python main.py --seed {}".format(i), shell=True)
    p.wait()
    p.kill()
    res = np.load('./out/res.npy')
    AUC.append(res[0])
    AP.append(res[1])
print('\n************************** Average results *******************************')
print('AUC: {:.4f}'.format(np.mean(AUC)))
print('AP: {:.4f}'.format(np.mean(AP)))


