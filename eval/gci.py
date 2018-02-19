# coding=utf-8


class GCI(object):
    def __init__(self, location, left_border, right_border):
        self.location = location
        self.left_border = left_border
        self.right_border = right_border
        self.estimated_list = list()

    def get_location(self):
        return self.location

    def get_left_border(self):
        return self.left_border

    def get_right_border(self):
        return self.right_border

    def get_estimated_list(self):
        return self.estimated_list

    def add_estimated_gci(self, location):
        if self.is_in_larynx_cycle(location):
            self.estimated_list.append(location)
            return True
        return False

    def is_correct(self):
        if self.estimated_list.__len__() == 1:
            return True
        return False

    def is_missed(self):
        if self.estimated_list.__len__() == 0:
            return True
        return False

    def is_in_larynx_cycle(self, location):
        if self.left_border <= location < self.right_border:
            return True
        return False

    def cal_error(self):
        if self.is_correct():
            return abs(self.location - self.estimated_list[0])
        return 0

    def accept_error(self, admissible_error=0.25):
        if self.is_correct():
            if admissible_error >= self.cal_error():
                return True
        return False


# Transform a list of reference gci location to a list of  GCI class.
def trans_ref_2_gci_list(ref, def_radius=5):
    real_gci_list = list()
    for i in range(0, ref.__len__()):
        left_radius = def_radius \
            if i == 0 or def_radius <= (ref[i] - ref[i - 1]) / 2 \
            else (ref[i] - ref[i - 1]) / 2
        right_radius = def_radius \
            if i == ref.__len__() - 1 or def_radius <= (ref[i + 1] - ref[i]) / 2 \
            else (ref[i + 1] - ref[i]) / 2
        real_gci_list.append(GCI(ref[i], ref[i] - left_radius, ref[i] + right_radius))
    return real_gci_list


# Assign estimated GCIs to real GCIs.
def assign_est_gci(real_gci_list, estimate):
    last_gci_idx = 0
    for i in range(0, estimate.__len__()):
        for j in range(last_gci_idx, real_gci_list.__len__()):
            if real_gci_list[j].is_in_larynx_cycle(estimate[i]):
                real_gci_list[j].add_estimated_gci(estimate[i])
                last_gci_idx = j


# Classify assigned  GCIs into three classes: correct, missed, falseAlarmed.
def classify_gci(real_gci_list):
    correct = list()
    missed = list()
    false_alarmed = list()
    accepted = list()
    for gci in real_gci_list:
        if gci.is_correct():
            correct.append(gci)
            if gci.accept_error():
                accepted.append(gci)
        elif gci.is_missed():
            missed.append(gci)
        else:
            false_alarmed.append(gci)
    return [correct, missed, false_alarmed, accepted]


# Get a list of error between real GCI & estimated GCI.
def get_err_list(cor_list):
    err_list = list()
    for gci in cor_list:
        err_list.append(gci.cal_error())
    return err_list
