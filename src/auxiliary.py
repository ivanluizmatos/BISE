import torch

class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def accuracy(output, target, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def update_meters_MulticolorMNIST(output, target, private_label, average_meters):

    with torch.no_grad():

        maxk = 1
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).squeeze()


        private_label_left, private_label_right = private_label[0], private_label[1]

        aligned_left  = (target == private_label_left)
        aligned_right = (target == private_label_right)

        aligned_aligned         = (  aligned_left  &   aligned_right).bool()
        aligned_conflicting     = (  aligned_left  & (~aligned_right)).bool()
        conflicting_aligned     = ((~aligned_left) &   aligned_right).bool()
        conflicting_conflicting = ((~aligned_left) & (~aligned_right)).bool()


        corrects_per_group = {
            'aligned_aligned':         correct[aligned_aligned],
            'aligned_conflicting':     correct[aligned_conflicting],
            'conflicting_aligned':     correct[conflicting_aligned],
            'conflicting_conflicting': correct[conflicting_conflicting]
            }


        for g in ['aligned_aligned', 'aligned_conflicting', 'conflicting_aligned', 'conflicting_conflicting']:
            if len(corrects_per_group[g]) > 0:
                 average_meters[g].update(100 * corrects_per_group[g].float().mean().item(),
                                          corrects_per_group[g].size(0))
                 


def update_meters_CivilComments(output, target, private_label, average_meters):

    with torch.no_grad():

        maxk = 1
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).squeeze()

        corrects_per_group = {
            '0_0': correct[((target == 0) & (private_label == 0)).bool()],
            '0_1': correct[((target == 0) & (private_label == 1)).bool()],
            '1_0': correct[((target == 1) & (private_label == 0)).bool()],
            '1_1': correct[((target == 1) & (private_label == 1)).bool()]
            }


        for g in ['0_0', '0_1', '1_0', '1_1']:
            if len(corrects_per_group[g]) > 0:
                 average_meters[g].update(100 * corrects_per_group[g].float().mean().item(),
                                          corrects_per_group[g].size(0))


# ===============================================================================================