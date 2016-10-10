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

import numpy as np


def fnr(tp, tn, fp, fn):
    return np.divide(fn, fn + tp)


def fpr(tp, tn, fp, fn):
    return np.divide(fp, fp + tn)


def tpr(tp, tn, fp, fn):
    return np.divide(tp, tp + fn)


def tnr(tp, tn, fp, fn):
    return np.divide(tn, tn + fp)


def precision(tp, tn, fp, fn):
    return np.divide(tp, tp + fp)


def recall(tp, tn, fp, fn):
    return np.divide(tp, tp + fn)


def metrics_curve(labels, prediction, metrics):
    """
        Computes metrics for all thresholds in prediction compared to labels.

        labels: a list of 0s and 1s (0: negative, 1: positive)
        prediction: a list of values
        metrics: list of functions with the four parameters: tp, tn, fp, fn in this order. Use the metrics in this file
                 as a guide
    """

    # aggregate ground truth and estimated into one matrix
    data = np.vstack((np.atleast_2d(labels), np.atleast_2d(prediction))).T.astype(float)

    # sort data by estimated (these are our thresholds)
    data = data[data[:, 1].argsort()]

    # rows: thresholds, columns: tp, tn, fp, fn
    stats = np.zeros((data.shape[0], 4))

    # iterate over thresholds (list is sorted, threshold is implicit) and track changes per threshold
    for idx in range(len(data)):
        stats[idx, 0] = stats[idx - 1, 0] - data[idx - 1, 0] if idx > 0 else sum(data[:, 0])  # tp
        stats[idx, 1] = stats[idx - 1, 1] + (1 - data[idx - 1, 0]) if idx > 0 else 0  # tn
        stats[idx, 2] = stats[idx - 1, 2] - (1 - data[idx - 1, 0]) if idx > 0 else sum(1 - data[:, 0])  # fp
        stats[idx, 3] = data.shape[0] - stats[idx, :3].sum()  # fn

    all_metrics = [metric(stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]) for metric in metrics]

    return data[:, 1], np.array(all_metrics).T


def area_under_curve(values):
    area = 0
    for index, right in enumerate(values[1:, :]):
        left = values[index]
        area += (max(left[1], right[1]) - abs(right[1] - left[1]) * 0.5) * (right[0] - left[0])

    return abs(area)  # the list can be the wrong way around, which turns around the sign

