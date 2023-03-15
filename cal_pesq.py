import librosa as lr
import numpy as np

from pesq import pesq
from os.path import dirname, join as pjoin
from scipy.io import wavfile



"""
calculate the mean value of PESQ improvement accross all results

"""


parser = argparse.ArgumentParser()
parser.add_argument('--wave_dst', default='', type=str, help='the directory to store the result files')
parser.add_argument('--num_eles', default= 50, type=int, help='number of test elements')
args = parser.parse_args()

pesq_list_z = []
pesq_list_y = []

for index in range(0, args.num_eles):

    print(index)
    wav_name = '{0:>08d}'.format(index)+'.wav'
    file_name = pjoin(args.wave_dst, wav_name)

    sc, data = wavfile.read(file_name)

    mix = data[:,1]
    target = data[:, 0]
    preds = data[:, 2]
    nb_pesq_z = pesq(sc, target, preds, 'nb')
    nb_pesq_y = pesq(sc, target, mix, 'nb')

    pesq_list_z.append(nb_pesq_z)

    pesq_list_y.append(nb_pesq_y)

mean_pesq_delta = sum(pesq_list_z)/len(pesq_list_z)-sum(pesq_list_y)/len(pesq_list_y)


