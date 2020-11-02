import json
import numpy as np
from mean_average_precisioin import AveragePrecisionMeter

object_categories = ['battery', 'bottle', 'firecracker', 'grenade', 'gun', 'hammer', 'merge_knife']

result_save_path = 'test_merge.json'


def main():

    ap_meter = AveragePrecisionMeter(False)
    ap_meter.reset()

    TP, FP, FN, TN = np.zeros(len(object_categories), dtype='float32'), np.zeros(len(object_categories), dtype='float32'), np.zeros(len(object_categories), dtype='float32'), np.zeros(len(object_categories), dtype='float32')
    two_cls_TP, two_cls_FP, two_cls_FN, two_cls_TN = 0.0, 0.0, 0.0, 0.0

    with open(result_save_path, 'r') as f:
        result = json.load(f)

    wrong_num = 0

    for i in range(len(result)):
        pred = result[i]['pred']
        label = result[i]['label']

        ap_meter.add(np.array(pred, dtype=float).reshape(1, len(object_categories)), np.array(label, dtype=float).reshape(1, len(object_categories)))

        pred = np.array(pred, dtype=float) > 0.5

        # thre = np.array([0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.7, 0.5])
        # pred = np.array(pred, dtype=float) > thre


        for col in range(len(object_categories)):
            if label[col] > 0.9 and pred[col]:
                TP[col] += 1
            elif label[col] > 0.9 and (not pred[col]):
                FN[col] += 1
            elif label[col] < 0.2 and pred[col]:
                FP[col] += 1
            elif label[col] < 0.2 and (not pred[col]):
                TN[col] += 1

        if np.array(pred == [i>0.5 for i in label]).sum() != len(object_categories):
            wrong_num += 1

        # two class
        if np.array(label).sum() > 1.5 and pred.sum() > 0.95:
            two_cls_TP += 1
        elif np.array(label).sum() > 1.5 and pred.sum() < 0.95:
            two_cls_FN += 1
        elif np.array(label).sum() < 1.0 and pred.sum() > 0.95:
            two_cls_FP += 1
        elif np.array(label).sum() < 1.0 and pred.sum() < 0.95:
            two_cls_TN += 1

    print('wrong_num:', wrong_num)
    print('TP:', TP)
    print('FP:', FP)
    print('FN:', FN)
    print('TN:', TN)

    ap = 100 * ap_meter.value()
    map = ap.mean()

    print("mAP: {:.3f}  ".format(map.item()))
    for col in range(len(object_categories)):
        precision = TP[col] / (TP[col] + FP[col])
        recall = TP[col] / (TP[col] + FN[col])
        print(object_categories[col])
        repr_str = '\t' + "ap: {:.3f}  ".format(ap[col]) + "precsion: {:.3f}  ".format(precision) + "recall: {:.3f}  ".format(recall)
        # repr_str = '\t' + "precsion: {:.3f}  ".format(precision) + "recall: {:.3f}  ".format(recall)
        print(repr_str)

    print('---------Two cls results--------')
    print('two_cls_TP:\t', two_cls_TP)
    print('two_cls_FP:\t', two_cls_FP)
    print('two_cls_FN:\t', two_cls_FN)
    print('two_cls_TN:\t', two_cls_TN)
    two_cls_precision = two_cls_TP / (two_cls_TP + two_cls_FP)
    two_cls_recall = two_cls_TP / (two_cls_TP + two_cls_FN)
    print("two_cls_precsion:  {:.3f}".format(two_cls_precision))
    print("two_cls_recall:    {:.3f}  ".format(two_cls_recall))
    print('---------Two cls results--------')


if __name__ == "__main__":
    main()

