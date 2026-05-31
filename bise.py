# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import torch
from torch import nn
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torch.utils import data
import argparse
import torch
import time
import random
import numpy as np
import torch
import copy
import pickle               
from pathlib import Path    
from datetime import datetime   


import pandas as pd
import torchvision
import torch.nn as nn


from src.auxiliary import *
from src.datasets_and_loaders import *


# For CivilComments
from src.aux_and_reimplementations import modified_softmax, ModifiedLayerNorm, replace_layernorm_with_modified, update_layernorm_masks



BMNIST_NAMES            = ['bmnist', 'biasedmnist', 'biased-mnist']
CELEBA_NAMES            = ['celeba']
CIFAR10_CORRUPTED_NAMES = ['cifar10c', 'cifar10-c', 'cifar10-corrupted', 'corrupted-cifar10', 'corruptedcifar10']
MULTICOLORMNIST_NAMES   = ['multicolormnist', 'multicolor-mnist', 'multi-colormnist']
CIVILCOMMENTS_NAMES     = ['civilcomments', 'civil comments', 'civil-comments']





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# from components import MaskLayer

from src.masked_models import SimpleConvNetPure, MaskedMLP

# ======================================================================================================
# ======================================================================================================


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from src.components import Privacy_head

# ------------------------- Implementation of the MI -------------------------
from src.losses import MI
# -------------------------------------------------------------------------

from src.components import mask_optimizer


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from src.train_eval_functions import train_vanilla, test_vanilla

##############################################################################

# -------------------- C.E. loss selection --------------------
# from losses import computeCEloss
##############################################################################


from src.train_eval_functions import pretrain_PH 


from src.train_mask_and_aux_classifier import train_PH_prune


from src.train_eval_functions import test


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from src.components import calculate_global_sparsity, update_tau_periodic


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def main_process():
    
    parser = argparse.ArgumentParser(description='Implementation of BISE debiasing algorithm')


    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dev', default="cuda:0")

    parser.add_argument('--datapath', default='data/')
    parser.add_argument('--dataset', default='Bmnist')

    parser.add_argument('--correlation', type=float, default=0.99)      # For BiasedMNIST and Corrupted-CIFAR10


    # ---- Specifically for BiasedMNIST ----
    parser.add_argument('--epochs', type=int, default=80)               # for the vanilla-training
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--test-batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)                # for the vanilla-training
    # --------------------------------------

    
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum-sgd', type=float, default=0.9)

    # For the debiasing process
    parser.add_argument('--lr_p', type=float, default=0.1)         # Learning rate for the aux. classifier
    parser.add_argument('--lr_m_i', type=float, default=1e-2)      # Learning rate for the mask

    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)       # Keep at 1

    parser.add_argument('--aux_classif_epochs', type=int, default=50)       # Epochs to train/refine aux. classifier
    parser.add_argument('--tau_update_period', type=int, default=10)        # Period for updating the temperature
    parser.add_argument('--tau_update_factor', type=float, default=0.5)     # Factor to update the temperature
    parser.add_argument('--tau_min', type=float, default=1e-3)              # tau_min to stop algorithm
    parser.add_argument('--initial_tau', type=float, default=1)              # tau_min to stop algorithm


    parser.add_argument('--max_debiasing_epochs', type=int, default=200)


    parser.add_argument('--str_add_info', type=str, default='')     # Additional info to be appended to the filename of the dictionary saved



    parser.add_argument('--resume_exec', type=int, default=1)     # 0 or 1
    parser.add_argument('--dict_to_resume', type=str, default='')

    parser.add_argument('--use_identification_model', type=int, default=0)     # 0 or 1


    # args = parser.parse_args([])
    args = parser.parse_args()




    # ----- Standardize dataset name -----
    if args.dataset.lower() in BMNIST_NAMES:
        args.dataset = 'Bmnist'
    elif args.dataset.lower() in CELEBA_NAMES:
        args.dataset = 'CelebA'
    elif args.dataset.lower() in CIFAR10_CORRUPTED_NAMES:
        args.dataset = 'Cifar10C'
    elif args.dataset.lower() in MULTICOLORMNIST_NAMES:
        args.dataset = 'MulticolorMNIST'
    elif args.dataset.lower() in CIVILCOMMENTS_NAMES:
        args.dataset = 'CivilComments'
    # ------------------------------------------------------



    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    args.device = torch.device(args.dev)



    # ---------------------------------------------------------
    args.weight_decay_m_i = args.weight_decay    

    args.annealing_mask_lr = 0           #  -->  0 (no annealing of lr_m_i), or 1 (annealing from epoch 1), or 2 (annealing when tau < threshold)
    start_anneal_epoch = -1              

    args.balanced_loss = {'Bmnist': 1,
                          'Cifar10C': 1, 
                          'CelebA': 8, 
                          'MulticolorMNIST': 3, 
                          'CivilComments': 8
                          }[args.dataset]
    
    if args.use_identification_model and args.dataset == 'Bmnist':
        args.balanced_loss = 'EMPIRICAL'
    
    args.modif_MI = 1           
    args.annealing_temperature = 'periodic'
    args.refine_PH = None

    seed = args.seed
    dataset = args.dataset
    correlation = args.correlation
    balanced_loss = args.balanced_loss
    gamma = args.gamma
    modif_MI = args.modif_MI
    annealing_temperature = args.annealing_temperature
    max_debiasing_epochs = args.max_debiasing_epochs
    refine_PH = args.refine_PH
    lr_m_i = args.lr_m_i
    refine_ep = args.aux_classif_epochs

    if args.resume_exec:
        args.str_add_info = args.str_add_info + '_RESUMED'

    str_add_info = args.str_add_info
    # ---------------------------------------------------------



    if dataset == 'Bmnist':
        print("\n==== EXPERIMENT: Bmnist ====")
    elif dataset == 'CelebA':
        print("\n==== EXPERIMENT: CelebA (loader from https://github.com/EIDOSLAB/bridging-debiasing-privacy-deep-learning/blob/master/src/datasets/celeba.py) ====")
    elif dataset == 'Cifar10C':
        print("\n==== EXPERIMENT: Cifar10C (loader from https://github.com/EIDOSLAB/unbiased-contrastive-learning/blob/master/debiasing/data/corrupted_cifar.py) ====")
    elif dataset == 'MulticolorMNIST':
        print("\n==== EXPERIMENT: MulticolorMNIST (loader from https://github.com/zhihengli-UR/DebiAN/blob/main/datasets/multi_color_mnist.py) ====")
    elif dataset == 'CivilComments':
        print("\n==== EXPERIMENT: CivilComments ====")

    # LOGGING_BATCHES = False
    
    # ======== FOR CIVILCOMMENTS ========
    BEST_VANILLA_CIVILCOMMENTS = False
    LAYERS_TO_MASK = ['ALL_LINEAR_LAYERS', 'ONLY_MLP', 'ALL_LINEAR_EXCEPT_QKV'][1]
    WHERE_TO_MASK = ['beforeActivLN', 'afterActivLN', 'afterActivBeforeLN'][0]
    USE_MODIFIED_LAYERNORM_SOFTMAX = True           # <-------- IMPORTANT
    # ---------------------------------------------------------




    # ======================================================================================================
    # ============================== For BiasedMNIST: model, loader, optim, rho ============================
    if args.dataset == 'Bmnist':
        tau = args.initial_tau 
        # model = SimpleConvNet(tau=tau, num_classes=10).to(args.device)

        model = SimpleConvNetPure(num_classes=10).to(args.device)
        # insert_masks_SimpleConvNet(model, tau=tau)

        model.avgpool = nn.Sequential(model.avgpool, torch.nn.Identity().to(args.device))

        args.MI = MI(privates=10, device=args.device)
        args.PH = Privacy_head(model.avgpool, nn.Sequential(torch.nn.Linear(128, 10))).to(args.device)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        #==========================================Biased 60k to train the vanilla model==================================================
        biased_train_vdataset = ColourBiasedMNIST(
            args.datapath + "MNIST/",
            train=True,
            download=True,
            data_label_correlation=args.correlation,
            n_confusing_labels=9,
            transform=transform,
        )

        biased_train_vloader = torch.utils.data.DataLoader(biased_train_vdataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True)
        

        #===================================================Biased 50k to train ==========================================================
        biased_dataset = ColourBiasedMNIST(args.datapath + "MNIST/", train=True, download=True,
                                          data_label_correlation=args.correlation, n_confusing_labels=9, transform=transform)
        biased_train_dataset, _ = torch.utils.data.random_split(biased_dataset, [50000, 10000])
        biased_train_loader = torch.utils.data.DataLoader(biased_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True)

        
        #=======================================Unbiased 10k for Validation ( from the 60k folder) =======================================
        unbiased_dataset = ColourBiasedMNIST(args.datapath + "MNIST/", train=True, download=True,
                                          data_label_correlation=0.1, n_confusing_labels=9, transform=transform)
        
        _, unbiased_val_dataset = torch.utils.data.random_split(unbiased_dataset, [50000, 10000])   

        unbiased_val_loader = torch.utils.data.DataLoader(unbiased_val_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=8, pin_memory=True)
        
        #=======================================Unbiased 10k for testing ( from the 10k folder) ============================================
        test_dataset = ColourBiasedMNIST(args.datapath + "MNIST/", train=False, data_label_correlation=0.1,
                                         n_confusing_labels=9, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=4, pin_memory=True)
        

        # Optimizer used to train the vanilla model
        args.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum_sgd, weight_decay=args.weight_decay)

    
    # ======================================================================================================
    # ================================ For CelebA: model, loader, optim, rho ===============================
    elif args.dataset == 'CelebA':

        tau = args.initial_tau 

        num_classes = 2
        num_features = 512
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1').to(args.device)
        model.fc = nn.Linear(
            in_features=num_features, out_features=num_classes, bias=True
        ).to(args.device)

        model.mask_layers = []
        model.names_masked_layers = []



        # Attach Privacy Head (Aux. classifier)
        model.avgpool = nn.Sequential(model.avgpool, torch.nn.Identity().to(args.device))
        args.MI = MI(privates=num_classes, device=args.device)
        args.PH = Privacy_head(model.avgpool, nn.Sequential(torch.nn.Linear(num_features, num_classes))).to(args.device)


        # Dataloaders                                             
        dataloaders_celebA = load_celeba(args)
        biased_train_loader     = dataloaders_celebA['train']
        unbiased_val_loader     = dataloaders_celebA['valid']
        test_loader             = dataloaders_celebA['unbiased_test']
        test_loader_conflicting = dataloaders_celebA['bias_conflicting_test']

        # Optimizer used to train the vanilla model
        args.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) 

        # Number of epochs to train vanilla model
        args.epochs = 50

        # Computing correlation for CelebA
        df = biased_train_loader.dataset.attr_df

        if balanced_loss in [1,2,3, 4]:
            count = (df['Blond_Hair'] == df['Male']).sum()
            args.correlation = count/len(biased_train_loader.dataset)

        elif balanced_loss == 5:
            men_dark    = ((df['Blond_Hair'] == 0) & (df['Male']==0)).sum()
            men_blond   = ((df['Blond_Hair'] == 1) & (df['Male']==0)).sum()
            women_dark  = ((df['Blond_Hair'] == 0) & (df['Male']==1)).sum()
            women_blond = ((df['Blond_Hair'] == 1) & (df['Male']==1)).sum()

            correlation_blond = women_blond / (women_blond + men_blond)
            correlation_dark  = men_dark / (men_dark + women_dark)
            args.correlation = {
                'correlation_blond': correlation_blond,
                'correlation_dark': correlation_dark
                }

        elif balanced_loss in [6, 7,8]:
            men_dark    = ((df['Blond_Hair'] == 0) & (df['Male']==0)).sum()
            men_blond   = ((df['Blond_Hair'] == 1) & (df['Male']==0)).sum()
            women_dark  = ((df['Blond_Hair'] == 0) & (df['Male']==1)).sum()
            women_blond = ((df['Blond_Hair'] == 1) & (df['Male']==1)).sum()

            correlation_blond = women_blond / (women_blond + men_blond)
            correlation_dark  = women_dark / (men_dark + women_dark)

            proportion_blond = (women_blond + men_blond) / (women_blond + men_blond + men_dark + women_dark)

            args.correlation = {
                'correlation_blond': correlation_blond,
                'correlation_dark': correlation_dark,
                'proportion_blond': proportion_blond
                }

    # ======================================================================================================
    # ========================== For Cifar10-Corrupted: model, loader, optim, rho ==========================
    elif args.dataset == 'Cifar10C':

        tau = args.initial_tau  

        num_classes = 10
        num_features = 512
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1').to(args.device)
        model.fc = nn.Linear(
            in_features=num_features, out_features=num_classes, bias=True
        ).to(args.device)

        model.mask_layers = []
        model.names_masked_layers = []
        



        # Attach Privacy Head (Aux. classifier)
        model.avgpool = nn.Sequential(model.avgpool, torch.nn.Identity().to(args.device))
        args.MI = MI(privates=num_classes, device=args.device)
        args.PH = Privacy_head(model.avgpool, nn.Sequential(torch.nn.Linear(num_features, num_classes))).to(args.device)

        # Dataloaders
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        T_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
        T_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        corruption = str(round(100-100*args.correlation, 1)).replace('.0','')+'pct'    
        data_dir = 'data/'

        train_dataset = CorruptedCIFAR10(root=data_dir, split="train", percent=corruption, transform=T_train)
        test_dataset = CorruptedCIFAR10(root=data_dir, split="test", percent=corruption, transform=T_test)

        biased_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,
                                                num_workers=8, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, 
                                                persistent_workers=True)


        # Optimizer used to train the vanilla model
        args.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Number of epochs to train vanilla model
        args.epochs = 200

    # ======================================================================================================
    # =========================== For Multicolor-MNIST: model, loader, optim, rho ==========================
    elif args.dataset == 'MulticolorMNIST':

        tau = args.initial_tau   

        num_classes = 10
        num_features = 100

        model = MaskedMLP(num_class=num_classes, tau=tau).to(args.device)


        # Attach Privacy Heads (Aux. classifiers) (here, we need to define TWO heads)
        model.feature = nn.Sequential(model.feature, torch.nn.Identity().to(args.device))
        args.MI = MI(privates=num_classes, device=args.device)
        args.PH_left = Privacy_head(model.feature, nn.Sequential(torch.nn.Linear(num_features, num_classes))).to(args.device)
        args.PH_right = Privacy_head(model.feature, nn.Sequential(torch.nn.Linear(num_features, num_classes))).to(args.device)


        # Dataloaders
        severity = 4
        correlation_left  = 0.99
        correlation_right = 0.95

        left_color_skew = round(1 - correlation_left, 2)    # left_color_skew  = 0.01
        right_color_skew = round(1 - correlation_right, 2)  # right_color_skew = 0.05

        train_dataset = MultiColorMNIST(
            'data', 'train', left_color_skew, right_color_skew, severity)
        test_dataset = MultiColorMNIST(
            'data', 'valid', left_color_skew, right_color_skew, severity)

        biased_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,
                                                num_workers=8, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, 
                                                persistent_workers=True)


        # Optimizer used to train the vanilla model
        args.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Number of epochs to train vanilla model
        args.epochs = 100

        # Correlation_left and Correlation_right
        args.correlation = (0.99, 0.95)


        assert balanced_loss in [0, 1, 3], "balanced_loss selected is NOT implemented for this dataset"



    # ======================================================================================
    # ===================== For CivilComments: model, loader, optim ========================

    elif args.dataset == 'CivilComments':

        import types
        import transformers
        import src.data_utils_civilcomments
        from torch.utils.data import DataLoader

        # from aux_and_reimplementations import modified_softmax, ModifiedLayerNorm, replace_layernorm_with_modified, update_layernorm_masks

        # -----------------------
        if USE_MODIFIED_LAYERNORM_SOFTMAX and LAYERS_TO_MASK=='ALL_LINEAR_LAYERS':
            nn.functional.softmax = modified_softmax
            F.softmax = modified_softmax
        # -----------------------


        def _bert_replace_fc(model):    # FROM: https://github.com/izmailovpavel/spurious_feature_learning/blob/main/models/text_models.py
            model.fc = model.classifier
            delattr(model, "classifier")

            def classifier(self, x):
                return self.fc(x)
            
            model.classifier = types.MethodType(classifier, model)

            model.base_forward = model.forward

            def forward(self, x):
                return self.base_forward(
                    input_ids=x[:, :, 0],
                    attention_mask=x[:, :, 1],
                    token_type_ids=x[:, :, 2]).logits

            model.forward = types.MethodType(forward, model)
            return model


        tau = args.initial_tau

        num_classes = 2
        num_features = 512
        model = _bert_replace_fc(
            transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            ).to(args.device)
        

        if USE_MODIFIED_LAYERNORM_SOFTMAX:
            replace_layernorm_with_modified(model)


        model.mask_layers = []
        model.names_masked_layers = []

        # print(model)


        # ---- Dataloaders (from https://github.com/izmailovpavel/spurious_feature_learning/blob/main/utils/common_utils.py#L234) ----

        

        train_transform = src.data_utils_civilcomments.BertTokenizeTransform(train=True)
        test_transform = src.data_utils_civilcomments.BertTokenizeTransform(train=False)

        basedir_civilcomments = 'data/civilcomments/new'

        trainset = src.data_utils_civilcomments.WildsCivilCommentsCoarse(basedir=basedir_civilcomments,
                               split="train",
                               transform=train_transform)

        testset_dict = {split: src.data_utils_civilcomments.WildsCivilCommentsCoarse(basedir=basedir_civilcomments,
                                           split=split,
                                           transform=test_transform)
                            for split in ["test", "val"]}


        

        collate_fn=src.data_utils_civilcomments.get_collate_fn(mixup=False, num_classes=trainset.n_classes)


        loader_kwargs = {'batch_size': 16,                         
                         'num_workers': 16, 'pin_memory': True}
        

        biased_train_loader = DataLoader(trainset,
                                         shuffle=True,
                                         sampler=None,
                                         collate_fn=collate_fn,
                                         **loader_kwargs)
        
        test_loader = DataLoader(testset_dict['test'],
                                 shuffle=False,
                                 **loader_kwargs)

        val_loader =  DataLoader(testset_dict['val'],
                                 shuffle=False,
                                 **loader_kwargs)



        # ---------------- Optimizer used to train the vanilla model ----------------
        # args.optimizer = ...   # NOTE THE VANILLA IS TRAINED USING CODE FROM https://github.com/izmailovpavel/spurious_feature_learning


        # ---------------- Number of epochs to train vanilla model ----------------
        # args.epochs = ...      # NOTE: THE VANILLA IS TRAINED USING CODE FROM https://github.com/izmailovpavel/spurious_feature_learning


        # ---------------- Computing correlation ----------------

        counts_groups_train = np.zeros((2,2))
        for y,s in zip(biased_train_loader.dataset.y_array, biased_train_loader.dataset.spurious_array):
            counts_groups_train[y][s] += 1

        args.proportions = counts_groups_train / len(biased_train_loader.dataset)


        if balanced_loss in [3]:
            count_aligned = counts_groups_train[0][0] + counts_groups_train[1][1]
            args.correlation = count_aligned/len(biased_train_loader.dataset)

        elif balanced_loss in [8]:

            args.correlation = 1 / (4 * args.proportions)    # 2x2 matrix



    # ======================================================================================================
    

    else:
        print('ERROR: Dataset not recognized!')
        return
    


    args.criterion = torch.nn.CrossEntropyLoss().to(args.device)




    print("\n\nEXPERIMENT LAUNCHED ON:  ", datetime.now().strftime("%d %B %Y, %H:%M:%S"))
    print('========================= SETTINGS =========================')
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    str_balanced_loss = ['', '_balancedLoss', '_balancedLoss2', '_balancedLoss3', '_balancedLoss4', '_balancedLoss5', '_balancedLoss6', '_balancedLoss7', '_balancedLoss8', '_balancedLoss9'][balanced_loss] if balanced_loss != 'EMPIRICAL' else '_EMPIRICAL'  # balanced_loss is 0 or 1 or 2
    str_modifMI = ['', '_modifMI', '_modifMI2'][args.modif_MI]
    str_annealing_temperature = '_cosine' if (args.annealing_temperature == 'cosine') else ''

    if args.dataset == 'Bmnist':
        dict_filename = ('dicts/Bmnist/rho_' + str(args.correlation) +
                        '/dict_BISE_' + args.dataset +
                        '_rho_' + str(args.correlation) +
                        '_gamma_' + str(args.gamma) +
                        '_extraction_' + str(args.epochs) +
                        '_seed_' + str(args.seed) +
                        str_balanced_loss +
                        str_modifMI +
                        str_annealing_temperature +
                        ('_maxDebEpochs_' + str(max_debiasing_epochs) if max_debiasing_epochs is not None else '') +
                        ('_refinePH' if args.refine_PH else '') +
                        str_add_info +
                        '.pkl')
    
    elif args.dataset == 'CelebA':
        dict_filename = ('dicts/CelebA' + 
                        '/dict_BISE_' + args.dataset +
                        '_gamma_' + str(args.gamma) +
                        '_seed_' + str(args.seed) +
                        str_balanced_loss +
                        str_modifMI +
                        str_annealing_temperature +
                        ('_maxDebEpochs_' + str(max_debiasing_epochs) if max_debiasing_epochs is not None else '') +
                        ('_refinePH' if args.refine_PH else '') +
                        str_add_info +
                        '.pkl')
    
    elif args.dataset == 'Cifar10C':
        dict_filename = ('dicts/Cifar10C/rho_' + str(args.correlation) +
                        '/dict_BISE_' + args.dataset +
                        '_rho_' + str(args.correlation) +
                        '_gamma_' + str(args.gamma) +
                        '_seed_' + str(args.seed) +
                        str_balanced_loss +
                        str_modifMI +
                        str_annealing_temperature +
                        ('_maxDebEpochs_' + str(max_debiasing_epochs) if max_debiasing_epochs is not None else '') +
                        ('_refinePH' if args.refine_PH else '') +
                        str_add_info +
                        '.pkl')

    elif args.dataset == 'MulticolorMNIST':
        dict_filename = ('dicts/MulticolorMNIST' +
                         
                         '/rhos_0.99_0.95' +
                         # '/rhos_' + str(args.correlation[0]) + '_' + str(args.correlation[1]) +

                         '/dict_BISE_' + args.dataset +

                         '/rhos_0.99_0.95' +
                         # '/rhos_' + str(args.correlation[0]) + '_' + str(args.correlation[1]) +

                         '_gamma_' + str(args.gamma) +
                         '_seed_' + str(args.seed) +
                         str_balanced_loss +
                         str_modifMI +
                         str_annealing_temperature +
                         ('_maxDebEpochs_' + str(max_debiasing_epochs) if max_debiasing_epochs is not None else '') +
                         ('_refinePH' if args.refine_PH else '') +
                         str_add_info +
                         '.pkl')


    elif args.dataset == 'CivilComments':
        dict_filename = ('dicts/CivilComments' +
                         '/dict_BISE_' + args.dataset +
                         '_gamma_' + str(args.gamma) +
                         '_seed_' + str(args.seed) +
                         str_balanced_loss +
                         str_modifMI +
                         str_annealing_temperature +
                         ('_maxDebEpochs_' + str(max_debiasing_epochs) if max_debiasing_epochs is not None else '') +
                         ('_refinePH' if args.refine_PH else '') +
                         str_add_info +
                         '.pkl')






    print("\nDictionary to be saved: " + dict_filename + "\n")
    print('============================================================\n')

    os.makedirs(os.path.dirname(dict_filename), exist_ok=True) 
    # ------------------------------


    # ---------------------------
    dict_train_val_test = {partition: {metric: [] for metric in ['loss_task', 'acc_task', 'MI_private', 'acc_private',
                                                                 'loss_task_pruned', 'acc_task_pruned',
                                                                 'vanilla_loss_task', 'vanilla_acc_task']}
                           for partition in ['train', 'val', 'test']}


    if args.dataset in ['MulticolorMNIST', 'CivilComments']:
        for partition in ['train', 'val', 'test']:
            for metric in ['vanilla_avg_acc', 'vanilla_weighted_avg_acc', 'vanilla_worst_group_avg_acc']:
                dict_train_val_test[partition][metric] = []


    
    dict_train_val_test['final_mask_layers'] = None
    dict_train_val_test['best_mask_layers'] = None


    dict_train_val_test['sparsity'] = []
    dict_train_val_test['sparsity_pruned'] = []
    # --------------------------



    # ====================================================================================================================================
    #===========================================================vanilla training==========================================================


    if args.dataset == 'Bmnist':
        # Scheduler for L.R. of vanilla model
        sched = torch.optim.lr_scheduler.MultiStepLR(args.optimizer, 
                                                    milestones=([40,60] if correlation in [0.99, 0.995] else [80,90]),
                                                    gamma=0.1, verbose=True)
        
        vanilla_model_path = f"checkpoints/Bmnist/corr_{args.correlation}/vanilla_model_corr_{args.correlation}_seed_{seed}.pth"     

    elif args.dataset == 'CelebA':
        vanilla_model_path = f"checkpoints/CelebA/vanilla_model_CelebA_seed_{seed}.pth"

    elif args.dataset == 'Cifar10C':
        # Scheduler for L.R. of vanilla model
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(args.optimizer, T_max=args.epochs)

        vanilla_model_path = f"checkpoints/Cifar10C/vanilla_model_Cifar10C_corr_{args.correlation}_seed_{seed}.pth"
    
    elif args.dataset == 'MulticolorMNIST':
        vanilla_model_path = f"checkpoints/MulticolorMNIST/vanilla_model_MulticolorMNIST_corrs_0.99_0.95_seed_{seed}.pth"

    elif args.dataset == 'CivilComments':   # NOTE: FOR CIVILCOMMENTS, THE VANILLA SHOULD BE TRAINED USING THE CODE FROM: https://github.com/izmailovpavel/spurious_feature_learning/tree/main
        if BEST_VANILLA_CIVILCOMMENTS:
            vanilla_model_path = f'checkpoints/civilcomments/erm_seed{seed}/best_checkpoint.pt'
        else:
            vanilla_model_path = f'checkpoints/civilcomments/erm_seed{seed}/final_checkpoint.pt'



    if args.dataset in ['Bmnist', 'CelebA', 'Cifar10C']:
        vanilla_model_path = vanilla_model_path.replace('.pth', '-without_masks.pth')




    if not Path(vanilla_model_path).exists():
        os.makedirs("checkpoints", exist_ok=True)
        print(f"\n========== Starting Training for Correlation {correlation} with Seed {seed} ==========")
        print(f"\n Vanilla model Training for correlation {correlation}:")

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}")

            # ========== TRAINING ==========

            train_loss_vanilla, train_acc_vanilla = train_vanilla(model, args,
                                                                  biased_train_vloader if args.dataset == 'Bmnist' else biased_train_loader,
                                                                  return_values = True)

            if args.dataset == 'MulticolorMNIST':
                (train_acc_vanilla,
                 train_alig_alig_acc_vanilla, train_alig_conf_acc_vanilla, train_conf_alig_acc_vanilla, train_conf_conf_acc_vanilla,
                 train_avg_acc_vanilla
                 ) = train_acc_vanilla
                


            elif args.dataset == 'CivilComments':   # NOTE: NOT USED FOR NOW, SINCE WE USE ANOTHER CODE TO TRAIN THE VANILLA FOR CIVILCOMMENTS
                (train_acc_vanilla,
                 train_0_0_acc_vanilla, train_0_1_acc_vanilla, train_1_0_acc_vanilla, train_1_1_acc_vanilla,
                 train_avg_acc_vanilla, train_weighted_avg_acc_vanilla, train_worst_group_acc_vanilla
                 ) = train_acc_vanilla
                # assert False, "Not implemented here. See README to check how to train the vanilla for CivilComments."


            dict_train_val_test['train']['vanilla_loss_task'].append(train_loss_vanilla)
            dict_train_val_test['train']['vanilla_acc_task'].append(train_acc_vanilla)


            if args.dataset in ['MulticolorMNIST', 'CivilComments']:
                dict_train_val_test['train']['vanilla_avg_acc'].append(train_avg_acc_vanilla)
                # dict_train_val_test['train']['vanilla_weighted_avg_acc'].append(train_weighted_avg_acc_vanilla)
                # dict_train_val_test['train']['vanilla_worst_group_avg_acc'].append(train_worst_group_acc_vanilla)



            with torch.no_grad():

                # ========== VALIDATING ==========            

                if args.dataset in ['Bmnist', 'CelebA', 'CivilComments']:

                    print('---------')

                    val_loss_vanilla, val_acc_vanilla = test_vanilla(model, args,
                                                                    val_loader if args.dataset in ['CivilComments'] else unbiased_val_loader,
                                                                    desc='On val. set',
                                                                    return_values = True)



                    if args.dataset == 'CivilComments':   # NOTE: NOT USED FOR NOW, SINCE WE USE ANOTHER CODE TO TRAIN THE VANILLA FOR CIVILCOMMENTS
                        (val_acc_vanilla,
                        val_0_0_acc_vanilla, val_0_1_acc_vanilla, val_1_0_acc_vanilla, val_1_1_acc_vanilla,
                        val_avg_acc_vanilla, val_weighted_avg_acc_vanilla, val_worst_group_acc_vanilla
                        ) = val_acc_vanilla



                    dict_train_val_test['val']['vanilla_loss_task'].append(val_loss_vanilla)
                    dict_train_val_test['val']['vanilla_acc_task'].append(val_acc_vanilla)


                    if args.dataset in ['CivilComments']:
                        dict_train_val_test['val']['vanilla_avg_acc'].append(val_avg_acc_vanilla)
                        dict_train_val_test['val']['vanilla_weighted_avg_acc'].append(val_weighted_avg_acc_vanilla)
                        dict_train_val_test['val']['vanilla_worst_group_avg_acc'].append(val_worst_group_acc_vanilla)


                # ========== TESTING ==========

                test_loss_vanilla, test_acc_vanilla = test_vanilla(model, args, test_loader, return_values = True)

                if args.dataset == 'MulticolorMNIST':
                    (test_acc_vanilla,
                    test_alig_alig_acc_vanilla, test_alig_conf_acc_vanilla, test_conf_alig_acc_vanilla, test_conf_conf_acc_vanilla,
                    test_avg_acc_vanilla
                    ) = test_acc_vanilla

                elif args.dataset == 'CivilComments':   # NOTE: NOT USED FOR NOW, SINCE WE USE ANOTHER CODE TO TRAIN THE VANILLA FOR CIVILCOMMENTS
                    (test_acc_vanilla,
                    test_0_0_acc_vanilla, test_0_1_acc_vanilla, test_1_0_acc_vanilla, test_1_1_acc_vanilla,
                    test_avg_acc_vanilla, test_weighted_avg_acc_vanilla, test_worst_group_acc_vanilla
                    ) = test_acc_vanilla



                dict_train_val_test['test']['vanilla_loss_task'].append(test_loss_vanilla)
                dict_train_val_test['test']['vanilla_acc_task'].append(test_acc_vanilla)


                if args.dataset in ['MulticolorMNIST', 'CivilComments']:
                    dict_train_val_test['test']['vanilla_avg_acc'].append(test_avg_acc_vanilla)
                    # dict_train_val_test['test']['vanilla_weighted_avg_acc'].append(test_weighted_avg_acc_vanilla)
                    # dict_train_val_test['test']['vanilla_worst_group_avg_acc'].append(test_worst_group_acc_vanilla)


                if args.dataset == 'CelebA':
                    test_vanilla(model, args, test_loader_conflicting, desc='Test vanilla (bias-conflicting set)')


                print('---------')


            if args.dataset in ['Bmnist', 'Cifar10C']:
                sched.step()



            torch.save(copy.deepcopy(model.state_dict()), vanilla_model_path.replace('.pth', f'_epoch_{epoch}.pth'))

        os.makedirs(os.path.dirname(vanilla_model_path), exist_ok=True)
        torch.save(model.state_dict(), vanilla_model_path)

    else:

        USE_BEST_VANILLA = False

        if USE_BEST_VANILLA:
            vanilla_model_path = vanilla_model_path.replace('_last.pth', '_best.pth')
            

        print(f"\nLoading Vanilla model from {vanilla_model_path} to continue training (PH pretraining)...")
        checkpoint = torch.load(vanilla_model_path)
        model.load_state_dict(checkpoint)
        model.to(args.device)
        test_vanilla(model, args, test_loader)
        if args.dataset == 'CelebA':
            test_vanilla(model, args, test_loader_conflicting, desc='Test vanilla (bias-conflicting set)')
    # ---------------------------------------


    # **************************************************************************************************************
    # ************************************************ UNSUPERVISED ************************************************

    if args.use_identification_model:

        IDENTIFICATION_MODEL_EPOCHS = 1

        # Save RNG states
        rng_state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state['cuda'] = torch.cuda.get_rng_state_all()


        if args.dataset == 'Bmnist':

            # from src.masked_models import SimpleConvNet
            from src.components import overwrite_bias_labels_with_predictions
        

            # Loading the IDENTIFICATION MODEL  # NOTE: NEED TO TRAIN THE IDENTIFICATION MODEL, FIRST
            identification_model_path = f"checkpoints/Bmnist/corr_{args.correlation}/vanilla_model_corr_{args.correlation}_seed_{args.seed}-IDENTIFICATION_MODEL_{IDENTIFICATION_MODEL_EPOCHS}_epochs-without_masks.pth"
            print(f'-------> Identification model: {identification_model_path}\n')
            identification_model = SimpleConvNetPure(num_classes=10).to(args.device)
            identification_model_checkpoint = torch.load(identification_model_path)
            identification_model.load_state_dict(identification_model_checkpoint)
            identification_model.to(args.device)

            for mask_layer_identif in identification_model.mask_layers:
                    mask_layer_identif.mode = 'vanilla'

            # Run identification model and replace bias labels
            all_preds, all_labels, all_private_labels = overwrite_bias_labels_with_predictions(identification_model, biased_train_loader, device=args.device)
            _ = overwrite_bias_labels_with_predictions(identification_model, unbiased_val_loader, device=args.device)

            # Updating correlation for Bmnist
            print('Updating correlation for Bmnist:')
            print('Original correlation for Bmnist:', args.correlation)

            proportion_correctly_classified = (all_preds == all_labels).float().mean().item()
            args.correlation = proportion_correctly_classified


            if args.balanced_loss == 'EMPIRICAL':
                args.correlation = torch.zeros((10,10)).to(args.device)
                for y_, b_ in zip(all_labels, all_preds):
                    args.correlation[y_][b_] += 1
                args.correlation = args.correlation.clamp(min=1)
                args.correlation = args.correlation / args.correlation.sum()

            print('Updated (WITH CLAMP!): args.correlation = ', args.correlation)   # empirical rho


            # Restore RNG states
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if torch.cuda.is_available() and 'cuda' in rng_state:
                torch.cuda.set_rng_state_all(rng_state['cuda'])

        # --------

        elif args.dataset == 'CelebA':

            from src.components import overwrite_attr_column_with_predictions


            # ======== Loading the IDENTIFICATION MODEL ========    # NOTE: NEED TO TRAIN THE IDENTIFICATION MODEL, FIRST
            identification_model_path = f"checkpoints/CelebA/vanilla_model_celebA_seed_{args.seed}-without_masks-IDENTIFICATION_MODEL_{IDENTIFICATION_MODEL_EPOCHS}_epochs.pth"
            print(f'-------> Identification model: {identification_model_path}\n')

            identification_model = torchvision.models.resnet18(weights='IMAGENET1K_V1').to(args.device)
            identification_model.fc = nn.Linear(
                in_features=num_features, out_features=num_classes, bias=True
            ).to(args.device)

            identification_model_checkpoint = torch.load(identification_model_path)
            identification_model.load_state_dict(identification_model_checkpoint)
            identification_model.to(args.device)
            identification_model.mask_layers = []
            identification_model.eval()


            for mask_layer_identif in identification_model.mask_layers:
                    mask_layer_identif.mode = 'vanilla'
    

            # Run identification model and replace bias labels
            all_preds, all_labels, all_private_labels = overwrite_attr_column_with_predictions(identification_model, biased_train_loader, device=args.device)
            _ = overwrite_attr_column_with_predictions(identification_model, unbiased_val_loader, device=args.device)


            # Updating correlation for CelebA
            print('Updating correlation for CelebA:')

            if args.balanced_loss == 1:

                print('Original correlation for CelebA:', args.correlation)

                proportion_correctly_classified = (all_preds == all_labels).mean().item()
                args.correlation = proportion_correctly_classified
                print('proportion_correctly_classified = ', args.correlation)

            elif args.balanced_loss == 8:

                print('Original correlation for CelebA:')
                for k in args.correlation.keys():
                    print(f"{k}: {args.correlation[k]}")

                df = biased_train_loader.dataset.attr_df

                men_dark    = ((df['Blond_Hair'] == 0) & (df['Male']==0)).sum()
                men_blond   = ((df['Blond_Hair'] == 1) & (df['Male']==0)).sum()
                women_dark  = ((df['Blond_Hair'] == 0) & (df['Male']==1)).sum()
                women_blond = ((df['Blond_Hair'] == 1) & (df['Male']==1)).sum()

                correlation_blond = women_blond / (women_blond + men_blond)
                correlation_dark  = women_dark / (men_dark + women_dark)

                proportion_blond = (women_blond + men_blond) / (women_blond + men_blond + men_dark + women_dark)

                args.correlation = {
                    'correlation_blond': correlation_blond,
                    'correlation_dark': correlation_dark,
                    'proportion_blond': proportion_blond
                    }
                print('Updated correlation for CelebA:')
                for k in args.correlation.keys():
                    print(f"{k}: {args.correlation[k]}")
    
            # Restore RNG states
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if torch.cuda.is_available() and 'cuda' in rng_state:
                torch.cuda.set_rng_state_all(rng_state['cuda'])


    # **************************************************************************************************************
    # **************************************************************************************************************




    # ==================================================================================================================
    # ====================================== INSERTING MASKS IN THE VANILLA MODEL ======================================

    if args.dataset == 'Bmnist':

        from src.masked_models import insert_masks_SimpleConvNet

        insert_masks_SimpleConvNet(model, tau=tau)



    # ******** FOR EXPERIMENTS USING ResNet18 ********
    if args.dataset in ['CelebA', 'Cifar10C']:

        from src.masked_models import insert_masks_resnet18

        insert_masks_resnet18(model, tau)




    if args.dataset == 'CivilComments':

        from src.masked_models import insert_masks_bert

        insert_masks_bert(model, model.mask_layers, model.names_masked_layers, where=WHERE_TO_MASK, layers_to_mask=LAYERS_TO_MASK, initial_tau=args.initial_tau)


        print('len(model.mask_layers) = ', len(model.mask_layers))       


        # Attach Privacy Head (Aux. classifier)
        model.bert.pooler = nn.Sequential(model.bert.pooler, torch.nn.Identity().to(args.device))
        args.MI = MI(privates=num_classes, device=args.device)
        args.PH = Privacy_head(model.bert.pooler, nn.Sequential(torch.nn.Linear(768, 2))).to(args.device)



    print(model)

    # ==================================================================================================================



    # Optimizer used to train the mask (m_i params)
    args.mask_optimizer = mask_optimizer(model, lr_m_i=args.lr_m_i, wd=args.weight_decay_m_i, momentum = args.momentum_sgd, optim='SGD')
    # args.mask_optimizer = mask_optimizer(model, optim='adam')


    # Optimizer used to train the Privacy Head (Aux. classifier)
    if args.dataset != 'MulticolorMNIST':
        args.PH_optimizer = torch.optim.SGD(args.PH.parameters(), lr=args.lr_p, momentum=args.momentum_sgd, weight_decay=args.weight_decay)
    else:
        args.PH_left_optimizer = torch.optim.SGD(args.PH_left.parameters(), lr=args.lr_p, momentum=args.momentum_sgd, weight_decay=args.weight_decay)
        args.PH_right_optimizer = torch.optim.SGD(args.PH_right.parameters(), lr=args.lr_p, momentum=args.momentum_sgd, weight_decay=args.weight_decay)
    



    # Cosine annealing scheduler for L.R. of mask (NOTE: THIS SCHED IS ONLY USED IF args.annealing_mask_lr == 1 or ==2)
    sched_mask = torch.optim.lr_scheduler.CosineAnnealingLR(args.mask_optimizer, T_max=args.max_debiasing_epochs)


    
    # ===========================================================================================================================
    #====================================================pretrain PH=============================================================

    start_time = time.time()
    print('started counting time!!!')

    if args.dataset != 'CivilComments':
        PH_path = vanilla_model_path.replace('-without_masks', '').replace('.pth','-PH.pth')

    else:
        PH_path = vanilla_model_path.replace('-without_masks', '').replace('.pt', f'-PH_epoch_{refine_ep}_lr_{args.lr_p}.pth')
    print('PH_path=', PH_path)

    # PH filenames for MulticolorMNIST
    PH_path_left  = PH_path.split('.pth')[0]+'_left.pth'
    PH_path_right = PH_path.split('.pth')[0]+'_right.pth'
    


    if ((args.dataset != 'MulticolorMNIST' and not Path(PH_path).exists()) or
        (args.dataset == 'MulticolorMNIST' and not (Path(PH_path_left).exists() and Path(PH_path_right).exists()))
        ):
        
        print(f"\n Pretraining the Privacy Head {correlation} with Seed {seed}:")
        
        for epoch in range(1, refine_ep + 1):
            print(f"\nPretrain Epoch {epoch}")
            # pretrain_PH(model,args, biased_train_vloader) 
            pretrain_PH(model,args, biased_train_loader, mode='vanilla')    


            if args.dataset in ['Bmnist', 'CelebA', 'CivilComments']:
                _ = test(model, args,
                        val_loader if args.dataset in ['CivilComments'] else unbiased_val_loader,
                        mode='vanilla', desc='On val. set')
            
            _ = test(model, args, test_loader, mode='vanilla', desc='On test set')


            # save PH after each epoch
            if args.dataset in ['CivilComments']:
                torch.save(args.PH.state_dict(), PH_path.replace('-PH', f'-PH_epoch_{epoch}_lr_{args.lr_p}'))

            
        if args.dataset != 'MulticolorMNIST':
            torch.save(args.PH.state_dict(), PH_path)
        else:
            torch.save(args.PH_left.state_dict(),  PH_path_left)
            torch.save(args.PH_right.state_dict(), PH_path_right)
    
    else:

        if args.dataset != 'MulticolorMNIST':
            print(f"\nLoading Privacy Head from {PH_path}")
            checkpoint = torch.load(PH_path)
            args.PH.load_state_dict(checkpoint)
            args.PH.to(args.device)
            # test_vanilla(model, args, test_loader)
            # if args.dataset == 'CelebA':
            #     test_vanilla(model, args, test_loader_conflicting, desc='Test vanilla (bias-conflicting set)')

        else:
            print(f"\nLoading Privacy Heads from {PH_path_left}, {PH_path_right}")
            
            checkpoint_left = torch.load(PH_path_left)
            args.PH_left.load_state_dict(checkpoint_left)
            args.PH_left.to(args.device)

            checkpoint_right = torch.load(PH_path_right)
            args.PH_right.load_state_dict(checkpoint_right)
            args.PH_right.to(args.device)

            # test_vanilla(model, args, test_loader)
            # if args.dataset == 'CelebA':
            #     test_vanilla(model, args, test_loader_conflicting, desc='Test vanilla (bias-conflicting set)')




    
    # ===========================================================================================================================
    #========================================================debiasing===========================================================
    
    # start_time = time.time()
    print(f"\n Debiasing process for correlation {correlation} with Seed {seed}:")
    epochs = 0


    refine_ep = args.aux_classif_epochs     # REFINE_EP    (epochs for training the aux. classifier)


    # -------------------
    if args.dataset in ['CivilComments', 'MulticolorMNIST']:

        dict_train_val_test['best_avg_acc_mask_layers'] = None
        dict_train_val_test['best_worst_group_acc_mask_layers'] = None
        dict_train_val_test['best_weighted_avg_acc_mask_layers'] = None


        if args.dataset == 'CivilComments':
            dict_train_val_test['names_masked_layers'] = model.names_masked_layers

            for partition in ['train', 'val', 'test']:
                for group_metric in ['acc_0_0', 'acc_0_1', 'acc_1_0', 'acc_1_1', 'acc_avg', 'acc_weighted_avg', 'acc_worst_group']:
                    dict_train_val_test[partition][group_metric] = []


        if args.dataset == 'MulticolorMNIST':
            for partition in ['train', 'val', 'test']:
                for group_metric in ['acc_alig_alig', 'acc_alig_conf', 'acc_conf_alig', 'acc_conf_conf', 'acc_avg', 'acc_weighted_avg', 'acc_worst_group']:
                    dict_train_val_test[partition][group_metric] = []


        best_val_acc_avg = -1
        best_val_acc_weighted_avg = -1
        best_val_acc_worst_group = -1

    # --------------------


    # -----------------------------------
    if annealing_temperature == 'cosine':
        T_max = args.max_debiasing_epochs
        placeholder_optim_temperature = torch.optim.SGD(params = nn.Linear(1, 1).parameters(), lr=1.0)
        cosine_annealer_temperature = torch.optim.lr_scheduler.CosineAnnealingLR(placeholder_optim_temperature, T_max)
    # -----------------------------------

    best_val_acc_pruned = -1 




    # ===================== FOR RESUMING LATER, IF NECESSARY =====================
    if args.resume_exec:

        with open(args.dict_to_resume, 'rb') as f:
            my_dict_to_resume = pickle.load(f)


        for mask_layer, state_loaded_mask_layer in zip(model.mask_layers, my_dict_to_resume['final_mask_layers']):
            mask_layer.load_state_dict(state_loaded_mask_layer)
            # import collections
            # mask_layer.m_i.data = state_loaded_mask_layer['m_i'].data if isinstance(state_loaded_mask_layer, collections.OrderedDict) else state_loaded_mask_layer.m_i.data # mask.mu


        epochs = my_dict_to_resume['resume']['last_epoch']

        if args.dataset in ['Bmnist', 'CelebA', 'CivilComments']:
            best_val_acc_pruned = my_dict_to_resume['resume']['best_val_acc_pruned']

        if args.dataset in ['CivilComments']:
            best_val_acc_avg = my_dict_to_resume['resume']['best_val_acc_avg']
            best_val_acc_weighted_avg = my_dict_to_resume['resume']['best_val_acc_weighted_avg']
            best_val_acc_worst_group = my_dict_to_resume['resume']['best_val_acc_worst_group']

        random.setstate(my_dict_to_resume['resume']['rng_python'])
        np.random.set_state(my_dict_to_resume['resume']['rng_numpy'])
        torch.set_rng_state(my_dict_to_resume['resume']['rng_torch'])
        torch.cuda.set_rng_state_all(my_dict_to_resume['resume']['rng_cuda_all'])

        args.mask_optimizer.load_state_dict(my_dict_to_resume['resume']['optimizer_state'])
        tau = my_dict_to_resume['resume']['tau']

        dict_train_val_test = my_dict_to_resume

        dict_train_val_test['dict_resumed'] = args.dict_to_resume
    # =============================================================================




    while True:
        epochs += 1
        print(f"\nEpoch {epochs}")
        # loss_task, acc_task, MI_private, acc_private = train_PH_prune(model, args, biased_train_vloader, balanced_loss=balanced_loss) 
        loss_task, acc_task, MI_private, acc_private = train_PH_prune(model, args, biased_train_loader, balanced_loss=balanced_loss, modif_MI=modif_MI, logs_dict=dict_train_val_test)


        with torch.no_grad():


            if args.dataset == 'CivilComments':
                update_layernorm_masks(model)



            # Evaluating the pruned model on training, val and test data
            if args.dataset != 'CivilComments':     # For CivilComments, We avoid evaluating the masked/pruned model on the training set, to save time
                train_loss_pruned, train_acc_pruned   = test(model, args, biased_train_loader,
                                                            balanced_loss=balanced_loss,   
                                                            mode='pruned', desc='Eval. pruned on training set ',
                                                            return_values_per_group= True if args.dataset in ['MulticolorMNIST', 'CivilComments'] else False)
            else:
                train_loss_pruned = None
                train_acc_pruned = None, None, None, None, None, None, None, None

            if args.dataset in ['Bmnist', 'CelebA', 'CivilComments']:
                val_loss_pruned, val_acc_pruned       = test(model, args, 
                                                             val_loader if args.dataset in ['CivilComments'] else unbiased_val_loader,
                                                             balanced_loss=balanced_loss,   
                                                             mode='pruned', desc='Validating pruned            ',
                                                             return_values_per_group = True if args.dataset in ['MulticolorMNIST', 'CivilComments'] else False)
            
            test_loss_pruned, test_acc_pruned     = test(model, args, test_loader,
                                                         balanced_loss=balanced_loss,   
                                                         mode='pruned', desc='Testing pruned               ',
                                                         return_values_per_group = True if args.dataset in ['MulticolorMNIST', 'CivilComments'] else False)
            
            if args.dataset == 'CelebA':
                _, _ = test(model, args, test_loader_conflicting,
                            balanced_loss=balanced_loss,   
                            mode='pruned', desc='Testing pruned (bias-conflicting set)               ')



        # --------------------------------------------------------
        if args.dataset in ['MulticolorMNIST', 'CivilComments']:
            acc_groups = {'train': {},
                          'val': {},
                          'test': {}
                          }

            (train_acc_pruned,
             acc_groups['train']['group_1'], acc_groups['train']['group_2'], acc_groups['train']['group_3'], acc_groups['train']['group_4'], 
             acc_groups['train']['avg'], acc_groups['train']['weighted_avg'], acc_groups['train']['worst_group']
             ) = train_acc_pruned
            
            if args.dataset in ['CivilComments']:
                (val_acc_pruned,
                acc_groups['val']['group_1'], acc_groups['val']['group_2'], acc_groups['val']['group_3'], acc_groups['val']['group_4'], 
                acc_groups['val']['avg'], acc_groups['val']['weighted_avg'], acc_groups['val']['worst_group']
                ) = val_acc_pruned


            (test_acc_pruned,
             acc_groups['test']['group_1'], acc_groups['test']['group_2'], acc_groups['test']['group_3'], acc_groups['test']['group_4'], 
             acc_groups['test']['avg'], acc_groups['test']['weighted_avg'], acc_groups['test']['worst_group']
             ) = test_acc_pruned


            if args.dataset == 'MulticolorMNIST':
                list_group_metrics_str = ['alig_alig', 'alig_conf', 'conf_alig', 'conf_conf']
            elif args.dataset == 'CivilComments':
                list_group_metrics_str = ['0_0', '0_1', '1_0', '1_1']

            for partition in (['train', 'val', 'test'] if args.dataset in ['CivilComments'] else ['train', 'test']):
                for placeholder_group_metric, group_metric in zip(['group_1', 'group_2', 'group_3', 'group_4', 'worst_group', 'avg', 'weighted_avg'],
                                                                    list_group_metrics_str + ['worst_group', 'avg', 'weighted_avg']):
                    dict_train_val_test[partition]['acc_' + group_metric].append(acc_groups[partition][placeholder_group_metric])

            if args.dataset in ['CivilComments']:
                if acc_groups['val']['avg'] >= best_val_acc_avg:
                    best_val_acc_avg = acc_groups['val']['avg']
                    dict_train_val_test['best_avg_acc_mask_layers'] = [copy.deepcopy(layer.state_dict()) for layer in model.mask_layers]

                if acc_groups['val']['weighted_avg'] >= best_val_acc_weighted_avg:
                    best_val_acc_weighted_avg = acc_groups['val']['weighted_avg']
                    dict_train_val_test['best_weighted_avg_acc_mask_layers'] = [copy.deepcopy(layer.state_dict()) for layer in model.mask_layers]

                if acc_groups['val']['worst_group'] >= best_val_acc_worst_group:
                    best_val_acc_worst_group = acc_groups['val']['worst_group']
                    dict_train_val_test['best_worst_group_acc_mask_layers'] = [copy.deepcopy(layer.state_dict()) for layer in model.mask_layers]

        # --------------------------------------------------------




        # --------------------
        if args.dataset in ['Bmnist', 'CelebA', 'CivilComments']:
            if val_acc_pruned >= best_val_acc_pruned:
                best_val_acc_pruned = val_acc_pruned
                mask_layers = model.mask_layers # save for pruning later...  
                dict_train_val_test['best_mask_layers'] = [copy.deepcopy(layer.state_dict()) for layer in model.mask_layers]

        # --------------------

        
        
        sparsity_report_pruned = calculate_global_sparsity(model)    # sparsity of the model pruned based on the m_i's obtained in the end of the epoch




        # ----------------------
        dict_train_val_test['train']['loss_task'].append(loss_task)
        dict_train_val_test['train']['acc_task'].append(acc_task)
        dict_train_val_test['train']['MI_private'].append(MI_private)
        dict_train_val_test['train']['acc_private'].append(acc_private)
        dict_train_val_test['train']['loss_task_pruned'].append(train_loss_pruned)
        dict_train_val_test['train']['acc_task_pruned'].append(train_acc_pruned)



        dict_train_val_test['test']['loss_task_pruned'].append(test_loss_pruned)
        dict_train_val_test['test']['acc_task_pruned'].append(test_acc_pruned)


        if args.dataset in ['Bmnist', 'CelebA', 'CivilComments']:
            

            dict_train_val_test['val']['loss_task_pruned'].append(val_loss_pruned)
            dict_train_val_test['val']['acc_task_pruned'].append(val_acc_pruned)


        dict_train_val_test['sparsity_pruned'].append(sparsity_report_pruned)
        # --------------------------



        # ---------------------------------------------------
        # tau = update_tau_periodic(model, args, biased_train_vloader, epochs, tau) 
        if annealing_temperature == 'periodic':                                      
            tau = update_tau_periodic(model, args, biased_train_loader, epochs, tau, period=args.tau_update_period, factor=args.tau_update_factor, refine_ep=refine_ep)  
    
        elif annealing_temperature == 'cosine':
            placeholder_optim_temperature.step()
            cosine_annealer_temperature.step()
            tau = cosine_annealer_temperature.get_last_lr()[0]
            print(f"\nEnd of epoch {epochs} ==========> NEW TEMPERATURE = {tau}\n")
            if args.refine_PH:
                for ep in range(refine_ep):   
                    pretrain_PH(model, args, biased_train_loader, mode='pruned') 
        # ---------------------------------------------------

        for module in model.mask_layers:
            module.tau = tau


        # ------- added (to overwrite dict every epoch) -------
        dict_train_val_test['final_mask_layers'] = [copy.deepcopy(layer.state_dict()) for layer in model.mask_layers]  



        # =========================== FOR RESUMING ==========================
        dict_train_val_test['resume'] = {}
        dict_train_val_test['resume']['optimizer_state'] = args.mask_optimizer.state_dict()
        dict_train_val_test['resume']['mask_threshold'] = 0.5
        dict_train_val_test['resume']['tau'] = tau

        if args.dataset in ['Bmnist', 'CelebA', 'CivilComments']:
            dict_train_val_test['resume']['best_val_acc_pruned'] = best_val_acc_pruned
        
        if args.dataset in ['CivilComments']:
            dict_train_val_test['resume']['best_val_acc_avg'] = best_val_acc_avg
            dict_train_val_test['resume']['best_val_acc_weighted_avg'] = best_val_acc_weighted_avg
            dict_train_val_test['resume']['best_val_acc_worst_group'] = best_val_acc_worst_group


        dict_train_val_test['resume']['last_epoch'] = epochs
        dict_train_val_test['resume']['rng_python'] = random.getstate()
        dict_train_val_test['resume']['rng_numpy'] = np.random.get_state()
        dict_train_val_test['resume']['rng_torch'] = torch.get_rng_state()
        dict_train_val_test['resume']['rng_cuda_all'] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        # ===================================================================




        with open(dict_filename, 'wb') as f:
            pickle.dump(dict_train_val_test, f)
        # ----------------------------------------------------------------


        #  added: for cosine annealing of the mask L.R.
        if args.annealing_mask_lr == 1:
            sched_mask.step()
            print(f"\nEnd of epoch {epochs} ==========> NEW MASK L.R. = {sched_mask.get_last_lr()[0]}\n")
        if args.annealing_mask_lr == 2:
            if tau < args.tau_min:    
                if start_anneal_epoch == -1:
                    start_anneal_epoch = epochs
                    print(f"Epoch {epochs} : START ANNEALING MASK L.R. -- for {args.max_debiasing_epochs} epochs.")
                sched_mask.step()
                print(f"\nEnd of epoch {epochs} ==========> NEW MASK L.R. = {sched_mask.get_last_lr()[0]}\n")


        # -------------------------
        if annealing_temperature == 'periodic' and args.annealing_mask_lr == 0: 
            if tau <= args.tau_min: 
                break

        if annealing_temperature == 'cosine' or args.annealing_mask_lr == 1:
            if epochs == args.max_debiasing_epochs:   
                break
        
        if args.annealing_mask_lr == 2:
            if (start_anneal_epoch != -1) and (epochs == args.max_debiasing_epochs + start_anneal_epoch):
                break
        # -------------------------

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTraining for correlation {correlation} with Seed {seed} completed in {elapsed_time:.2f} seconds.\n")



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù

if __name__ == '__main__':

    main_process()

# %%
