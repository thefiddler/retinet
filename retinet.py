# This file is part of the supplementary material for the manuscript:
#
# RetiNet: Automated AMD identification in OCT volumetric data
#
# Copyright (C) 2016 Stefanos Apostolopoulos
#                    Carlos Ciller
#                    Sandro De Zanet
#                    Raphael Sznitman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import random
random.seed(42)

import h5py
import keras
import numpy as np
import os
import yaml
import zipfile

from keras.models import *
from keras.utils.data_utils import get_file

from metrics import *


def normalize(volume):
    stdev = np.std(volume, dtype=np.float64)
    mean = np.mean(volume, dtype=np.float64)
    return (volume - np.float32(mean)) / np.float32(stdev)

#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

archive = get_file('dataset.zip',
                   'https://dl.dropbox.com/s/10fg73028s8xdld/dataset.zip?dl=1',
                   cache_subdir='retinet',
                   md5_hash='04c6ad5a6d3e855f7a639ddcd3d8564a')

print('Extracting dataset')
with zipfile.ZipFile(os.path.expanduser('~/.keras/retinet/dataset.zip')) as dataset:
    dataset.extractall('.')
    dataset.close()

print('Loading dataset')
dataset = h5py.File('dataset/oct.h5', mode='r')
labels = np.array(dataset['labels'][0], dtype=np.float32)

print('Compiling model')
model_yaml = yaml.load(open('dataset/retinet.yml').read())
model = model_from_yaml(model_yaml)
model.summary()

weights = [
    'dataset/retinet_weights0.h5', 'dataset/retinet_weights1.h5',
    'dataset/retinet_weights2.h5', 'dataset/retinet_weights3.h5',
    'dataset/retinet_weights4.h5'
]

shuffle = [42, 44, 186, 132, 99, 119, 138, 103, 85, 105, 104, 148, 197, 142,
           29, 124, 221, 106, 2, 201, 52, 203, 180, 77, 78, 101, 190, 21, 79,
           9, 84, 12, 81, 27, 216, 130, 61, 220, 30, 191, 199, 60, 4, 152, 45,
           5, 36, 194, 3, 15, 66, 182, 210, 126, 153, 168, 184, 73, 72, 149,
           100, 10, 89, 47, 140, 156, 198, 38, 13, 195, 214, 64, 112, 0, 120,
           111, 185, 37, 121, 145, 34, 123, 98, 192, 223, 32, 176, 157, 165,
           48, 227, 134, 76, 109, 219, 200, 128, 19, 144, 204, 171, 161, 125,
           65, 122, 226, 46, 51, 110, 127, 159, 115, 169, 131, 177, 95, 172,
           162, 33, 146, 135, 82, 174, 113, 202, 63, 217, 206, 133, 54, 16,
           164, 102, 80, 211, 167, 14, 83, 196, 69, 155, 118, 213, 222, 136,
           43, 18, 68, 53, 90, 94, 41, 93, 116, 187, 181, 25, 207, 170, 74, 58,
           175, 17, 49, 147, 92, 158, 160, 75, 141, 20, 96, 31, 137, 117, 11,
           67, 205, 88, 91, 24, 97, 209, 218, 86, 208, 39, 193, 87, 212, 178,
           40, 1, 71, 150, 114, 56, 107, 215, 179, 166, 183, 50, 143, 225, 154,
           129, 59, 55, 23, 7, 8, 108, 151, 22, 139, 224, 173, 26, 188, 35, 57,
           62, 70, 189, 6, 28, 163]
cscan_count = len(shuffle)

fold_labels = []
fold_results = []
for fold in range(5):
    print("Fold {0}:".format(fold))
    print("[Cscan]\t\t[Result]\t[Label]")

    model.load_weights(weights[fold])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    test_idx = list( \
        range(int(round(float(fold) / 5. * cscan_count)),
              int(round((fold + 1.) / 5. * cscan_count))))

    for idx in test_idx:
        cscan = normalize(np.expand_dims(
            np.array(dataset['cscans'][0][shuffle[idx]], dtype=np.float32),
            0))
        result = model.predict(cscan, batch_size=1)[0]
        label = labels[shuffle[idx]]

        mistake = abs(result[1] - label[1]) > 0.5
        print("{0}({1})\t{2:1.3f}\t\t{3:1.3f}{4}".format(
            "AMD\t" if label[1] == 1 else "Control ", shuffle[idx],
            result[1], label[1], ' *' if mistake else ''))

        fold_results.append(result)
        fold_labels.append(label)

    print()


fold_labels = np.array(fold_labels, dtype=np.float)
fold_results = np.array(fold_results, dtype=np.float)
thresholds, all_metrics = metrics_curve(fold_labels[:, 1], fold_results[:, 1],
                                        [fnr, fpr, tpr, recall, precision])
all_metrics = np.array(all_metrics, dtype=np.float)

fnr_vs_fpr = np.array(all_metrics[:, [0, 1]])
roc = all_metrics[:, [1, 2]]
roc_auc = area_under_curve(roc)

thresh_for = lambda x, y: np.argmin(([abs(element[y] - x) for element in all_metrics]))
thresh95 = thresh_for(0.05, 0)
thresh99 = thresh_for(0.01, 0)

fpr_threshold_95 = all_metrics[thresh95, 1]
fpr_threshold_99 = all_metrics[thresh99, 1]

print("Results:")
print("ROC-AUC: {0:1.4f}".format(roc_auc))
print("FPR 95%: {0:1.4f}".format(fpr_threshold_95))
print("FPR 99%: {0:1.4f}".format(fpr_threshold_99))
