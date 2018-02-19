# coding=utf-8

import os
import numpy as np
from gci import assign_est_gci, classify_gci, get_err_list, trans_ref_2_gci_list


class ResultEvaluator(object):
    def __init__(self, ref_path, est_path, rate=20000, def_dist=5):
        self.ref_path = ref_path
        self.est_path = est_path
        self.def_dist = def_dist
        self.rate = rate
        self.mark_count = 0
        self.cor_count = 0
        self.mis_count = 0
        self.fls_count = 0
        self.acc_count = 0
        self.err_list = list()

    def get_mark_count(self):
        return self.mark_count

    def get_cor_count(self):
        return self.cor_count

    def get_mis_count(self):
        return self.mis_count

    def get_fls_count(self):
        return self.fls_count

    def get_acc_count(self):
        return self.acc_count

    def get_err_list(self):
        return self.err_list

    def _result_eval(self, key):
        #  List the location of  real GCIs.
        # ref = self.ref_file[key]
        ref = self.get_ref_mark_list(key)
        #  List the location of  estimated GCIs.
        # est = self.est_file[key]
        est = self.get_est_mark_list(key)
        # Transform a list of reference gci location to a list of  GCI class.
        ref_mark_list = trans_ref_2_gci_list(ref)
        self.mark_count += ref_mark_list.__len__()
        # Assign estimated GCIs to real GCIs.
        assign_est_gci(ref_mark_list, est)
        # Classify assigned  GCIs into three classes: correct, missed, falseAlarmed.
        [correct, missed, false_alarmed, accepted] = classify_gci(ref_mark_list)
        self.cor_count += correct.__len__()
        self.mis_count += missed.__len__()
        self.fls_count += false_alarmed.__len__()
        self.acc_count += accepted.__len__()
        # Get a list of error  between real GCI & estimated GCI
        err_list = get_err_list(correct)
        self.err_list.append(err_list)

    def results_eval(self):
        key_list = self.get_key_list()
        for key in key_list:
            self._result_eval(key)

    def print_eval_metrics(self):
        print("GCI Detection Metrics: ")
        cor_rate = self.cal_cor_rate()
        mis_rate = self.cal_mis_rate()
        fls_rate = self.cal_fls_rate()
        acc_rate = self.cal_acc_rate()
        id_accuracy = self.cal_id_accuracy()
        print("\tcorrect rate:{}".format(cor_rate))
        print("\tmissed rate:{}".format(mis_rate))
        print("\tfalse alarmed rate:{}".format(fls_rate))
        print("\taccuracy:{}".format(acc_rate))
        print("\tid accuracy:{}".format(id_accuracy))
        pass

    def cal_cor_rate(self):
        return self.cor_count / self.mark_count

    def cal_mis_rate(self):
        return self.mis_count / self.mark_count

    def cal_fls_rate(self):
        return self.fls_count / self.mark_count

    def cal_acc_rate(self):
        return self.acc_count / self.cor_count

    def cal_id_accuracy(self):
        return np.std(self.err_list, ddof=1)

    def get_key_list(self):
        # Get all mark file names as keys.
        items = os.listdir(self.ref_path)
        key_list = []
        for names in items:
            if names.endswith(".marks"):
                key_list.append(names.split(".")[0])
                pass
            pass
        return key_list

    def get_ref_mark_list(self, key):
        return self._get_mark_list(self.ref_path + key + ".marks")

    def get_est_mark_list(self, key):
        return self._get_mark_list(self.est_path + key + ".marks")

    def _get_mark_list(self, path):
        mark_list = list()
        with open(path) as mark_file:
            while 1:
                lines = mark_file.readlines(10000)
                if not lines:
                    break
                mark_list.extend(map(lambda l: round(float(l) * self.rate), lines))
        return mark_list


def main():
    ref_path = "data/result/model/ref/"
    est_path = "data/result/model/est/"
    evaluator = ResultEvaluator(ref_path, est_path, rate=20000, def_dist=5)
    evaluator.results_eval()
    evaluator.print_eval_metrics()
    pass


if __name__ == "__main__":
    main()
