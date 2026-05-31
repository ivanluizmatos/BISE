import numpy as np
import torch

# ====================================================================

# Adapted from:
# https://github.com/renahon/mining_bias_target_alignment_from_voronoi_cells/blob/main/utils/vcba/information_removal.py

class MI(torch.nn.Module):
    def __init__(self, privates: int = 10, device: str = "cpu"):
        super(MI, self).__init__()
        self.device = device
        self.privates = privates
        self.scaling = 1 / np.log(privates)

    def _get_joint_and_marginals(self, GT_private_onehot, prob_private, nb_samples):
        joint = (
            torch.clamp(
                torch.mm(torch.transpose(GT_private_onehot, 0, 1), prob_private),
                min=1e-15,
            )
            / nb_samples
        )
        marginals_out_private = torch.sum(joint, dim=0, keepdim=True)
        marginals_GT_private = torch.sum(joint, dim=1, keepdim=True)
        marginals = torch.clamp(
            torch.mm(marginals_GT_private, marginals_out_private), min=1e-15
        )
        return joint, marginals

    def forward(self, private_head, private_label, conflict_mask=None):
        out_private = private_head.forward_attached()
        GT_private_onehot = 1.0 * torch.nn.functional.one_hot(private_label, num_classes=self.privates)
        prob_private = torch.nn.functional.softmax(out_private, dim=1)

        # ------------
        if conflict_mask is None:
            GT_private_onehot_masked = GT_private_onehot
            prob_private_masked = prob_private
        else:
            GT_private_onehot_masked = GT_private_onehot[conflict_mask, :]
            prob_private_masked = prob_private[conflict_mask, :]
        # ------------
        nb_samples_masked = torch.sum(GT_private_onehot_masked)

        (joint_masked, marginals_masked) = self._get_joint_and_marginals(
            GT_private_onehot_masked, prob_private_masked, nb_samples_masked
        )
        
        return torch.sum(
            joint_masked * torch.log(joint_masked / marginals_masked) * self.scaling
        )


# ====================================================================


def computeCEloss(args, output, target, private_label):

    # NOTE: automatizing for different datasets
    if args.dataset in ['Bmnist', 'Cifar10C', 'MulticolorMNIST']:
        nb_classes = 10
    elif args.dataset in ['CelebA', 'CivilComments']:
        nb_classes = 2


    ############ For unsupervised ############
    if args.balanced_loss == 'EMPIRICAL' and args.dataset == 'Bmnist':
        balancing_weights = (1. / (nb_classes*nb_classes * args.correlation))[target, private_label]

        target_criterion = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)
        loss_task_per_sample = target_criterion(output, target)
        loss_task = (loss_task_per_sample * balancing_weights).mean()
    ##########################################



    elif args.balanced_loss == 1:             # reweighting v1 (consider number of bias-aligned and bias-conflicting samples in the whole training set)

        if args.dataset != 'MulticolorMNIST':
        
            weight_aligned    = 1/(nb_classes*args.correlation)
            weight_conficting = (nb_classes-1)/(nb_classes*(1-args.correlation))
            balancing_weights = torch.where(target == private_label, weight_aligned, weight_conficting)


        else:   # ---- SPECIFIC IMPLEMENTATION FOR MulticolorMNIST ----

            private_label_left  = private_label[0]
            private_label_right = private_label[1]

            (rho_left, rho_right) = args.correlation 

            weight_aligned_aligned         = (1/(nb_classes**2)) * 1 / (rho_left * rho_right)
            weight_aligned_conflicting     = ((nb_classes-1)/(nb_classes**2)) * 1 / (rho_left * (1 - rho_right))
            weight_conflicting_aligned     = ((nb_classes-1)/(nb_classes**2)) * 1 / ((1 - rho_left) * rho_right)
            weight_conflicting_conflicting = (((nb_classes-1)**2)/(nb_classes**2)) * 1 / ((1 - rho_left) * (1 - rho_right))

            aligned_left  = (target == private_label_left)
            aligned_right = (target == private_label_right)

            aligned_aligned         = (  aligned_left  &   aligned_right)
            aligned_conflicting     = (  aligned_left  & (~aligned_right))
            conflicting_aligned     = ((~aligned_left) &   aligned_right)
            conflicting_conflicting = ((~aligned_left) & (~aligned_right))
            
            balancing_weights =  torch.where(aligned_aligned,         weight_aligned_aligned,         0.)
            balancing_weights += torch.where(aligned_conflicting,     weight_aligned_conflicting,     0.)
            balancing_weights += torch.where(conflicting_aligned,     weight_conflicting_aligned,     0.)
            balancing_weights += torch.where(conflicting_conflicting, weight_conflicting_conflicting, 0.)




        target_criterion = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)
        loss_task_per_sample = target_criterion(output, target)
        loss_task = (loss_task_per_sample * balancing_weights).mean()  # BALANCED LOSS v1

    elif args.balanced_loss == 2:           # reweighting v2 (consider number of bias-aligned and bias-conflicting samples in the CURRENT BATCH)

        B = len(output)   # current batch size

        num_aligned = (target == private_label).sum()
        num_conflicting = B - num_aligned

        k_aligned = B / (args.correlation*(nb_classes-1)*num_conflicting + (1-args.correlation)*num_aligned)
        k_conflicting = (nb_classes-1)*k_aligned

        weight_aligned    = k_aligned * (1 - args.correlation)
        weight_conficting = k_conflicting * args.correlation

        balancing_weights = torch.where(target == private_label, weight_aligned, weight_conficting)

        target_criterion = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)
        loss_task_per_sample = target_criterion(output, target)
        loss_task = (loss_task_per_sample * balancing_weights).mean()  # BALANCED LOSS v2

    elif args.balanced_loss == 3:           # Reweighting from "Voronoi cells" paper (Nahon et al.)

        if args.dataset != 'MulticolorMNIST':

            weight_aligned    = 1/(args.correlation)
            weight_conficting = 1/(1-args.correlation)
            balancing_weights = torch.where(target == private_label, weight_aligned, weight_conficting)


        else:   # ---- SPECIFIC IMPLEMENTATION FOR MulticolorMNIST ----

            private_label_left  = private_label[0]
            private_label_right = private_label[1]

            (rho_left, rho_right) = args.correlation 

            weight_aligned_aligned         = 1 / (rho_left * rho_right)
            weight_aligned_conflicting     = 1 / (rho_left * (1 - rho_right))
            weight_conflicting_aligned     = 1 / ((1 - rho_left) * rho_right)
            weight_conflicting_conflicting = 1 / ((1 - rho_left) * (1 - rho_right))

            aligned_left  = (target == private_label_left)
            aligned_right = (target == private_label_right)

            aligned_aligned         = (  aligned_left  &   aligned_right)
            aligned_conflicting     = (  aligned_left  & (~aligned_right))
            conflicting_aligned     = ((~aligned_left) &   aligned_right)
            conflicting_conflicting = ((~aligned_left) & (~aligned_right))
            
            balancing_weights =  torch.where(aligned_aligned,         weight_aligned_aligned,         0.)
            balancing_weights += torch.where(aligned_conflicting,     weight_aligned_conflicting,     0.)
            balancing_weights += torch.where(conflicting_aligned,     weight_conflicting_aligned,     0.)
            balancing_weights += torch.where(conflicting_conflicting, weight_conflicting_conflicting, 0.)



        target_criterion = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)
        loss_task_per_sample = target_criterion(output, target)
        loss_task = (loss_task_per_sample * balancing_weights).mean()  # BALANCED LOSS v3 (as used in paper -- Voronoi)

    elif args.balanced_loss == 4:           # Only computing the loss for bias conflicting samples. IGNORE IT.

        weight_aligned    = 0.
        weight_conficting = 1.
        balancing_weights = torch.where(target == private_label, weight_aligned, weight_conficting)

        target_criterion = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)
        loss_task_per_sample = target_criterion(output, target)
        loss_task = (loss_task_per_sample * balancing_weights).mean()  # BALANCED LOSS v4 (only computing C.E. loss for the bias conflicting samples...)
    
    elif args.balanced_loss == 5:           # IGNORE IT. First attempt of loss reweighting for CelebA.

        weight_aligned_dark = 1 / args.correlation['correlation_dark']
        weight_conflicting_dark = 1 / (1 - args.correlation['correlation_dark'])

        weight_aligned_blond = 1 / args.correlation['correlation_blond']
        weight_conflicting_blond = 1 / (1 - args.correlation['correlation_blond'])

        aligned_dark     = ((target==0) & (private_label==0))
        conflicting_dark = ((target==0) & (private_label==1))
        
        aligned_blond     = ((target==1) & (private_label==1))
        conflicting_blond = ((target==1) & (private_label==0))

        balancing_weights =  torch.where(aligned_dark, weight_aligned_dark, 0.)
        balancing_weights += torch.where(conflicting_dark, weight_conflicting_dark, 0.)
        balancing_weights += torch.where(aligned_blond, weight_aligned_blond, 0.)
        balancing_weights += torch.where(conflicting_blond, weight_conflicting_blond, 0.)

        target_criterion = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)
        loss_task_per_sample = target_criterion(output, target)
        loss_task = (loss_task_per_sample * balancing_weights).mean()  # BALANCED LOSS v5

    elif args.balanced_loss == 6:           # Reweighting to account ONLY for target class imbalance

        if args.dataset == 'CelebA':        # Accounting only for class imbalance "dark vs. blond hair"

            balancing_weights =  torch.where(target==1, 1/args.correlation['proportion_blond'], 1/(1 - args.correlation['proportion_blond']))

            target_criterion = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)
            loss_task_per_sample = target_criterion(output, target)
            loss_task = (loss_task_per_sample * balancing_weights).mean()  # BALANCED LOSS v6


    elif args.balanced_loss == 7:           # Reweighting to account for target class imbalance and for spurious correlations

        if args.dataset == 'CelebA':

            c_b = args.correlation['proportion_blond']
            rho_b = args.correlation['correlation_blond']
            rho_d = args.correlation['correlation_dark']

            weight_women_blond = 1 / (rho_b * c_b)
            weight_men_blond = 1 / ((1-rho_b) * c_b)

            weight_men_dark = 1 / ((1-rho_d) * (1-c_b))
            weight_women_dark = 1 / (rho_d * (1-c_b))

            men_dark  = ((target==0) & (private_label==0))
            men_blond = ((target==1) & (private_label==0))

            women_dark  = ((target==0) & (private_label==1))
            women_blond = ((target==1) & (private_label==1))

            balancing_weights =  torch.where(men_dark, weight_men_dark, 0.)
            balancing_weights += torch.where(men_blond, weight_men_blond, 0.)
            balancing_weights += torch.where(women_dark, weight_women_dark, 0.)
            balancing_weights += torch.where(women_blond, weight_women_blond, 0.)

            target_criterion = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)
            loss_task_per_sample = target_criterion(output, target)
            loss_task = (loss_task_per_sample * balancing_weights).mean()  # BALANCED LOSS v7
        


    elif args.balanced_loss == 8:           # similar to balanced_loss 7, but with factor 1/4 for CelebA

        if args.dataset == 'CelebA':

            c_b = args.correlation['proportion_blond']
            rho_b = args.correlation['correlation_blond']
            rho_d = args.correlation['correlation_dark']

            weight_women_blond = 1 / (4 * rho_b * c_b)
            weight_men_blond = 1 / (4 * (1-rho_b) * c_b)

            weight_men_dark = 1 / (4 * (1-rho_d) * (1-c_b))
            weight_women_dark = 1 / (4 * rho_d * (1-c_b))

            men_dark  = ((target==0) & (private_label==0))
            men_blond = ((target==1) & (private_label==0))

            women_dark  = ((target==0) & (private_label==1))
            women_blond = ((target==1) & (private_label==1))

            balancing_weights =  torch.where(men_dark, weight_men_dark, 0.)
            balancing_weights += torch.where(men_blond, weight_men_blond, 0.)
            balancing_weights += torch.where(women_dark, weight_women_dark, 0.)
            balancing_weights += torch.where(women_blond, weight_women_blond, 0.)


        elif args.dataset in ['CivilComments']:

            balancing_weights =  torch.where((target==0) & (private_label==0), args.correlation[0][0], 0.)
            balancing_weights += torch.where((target==0) & (private_label==1), args.correlation[0][1], 0.)
            balancing_weights += torch.where((target==1) & (private_label==0), args.correlation[1][0], 0.)
            balancing_weights += torch.where((target==1) & (private_label==1), args.correlation[1][1], 0.)


        target_criterion = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)
        loss_task_per_sample = target_criterion(output, target)
        loss_task = (loss_task_per_sample * balancing_weights).mean()  # BALANCED LOSS v8



    else:   # balanced_loss == 0:

        loss_task = args.criterion(output, target)  # NOT balanced loss

    return loss_task
