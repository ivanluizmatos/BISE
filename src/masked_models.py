import torch
import torch.nn as nn
from src.components import MaskLayer


# ======================================================================================================
# ======================================= For BiasedMNIST: model =======================================
class SimpleConvNet(nn.Module):
    def __init__(self, tau, num_classes=10, kernel_size=7, feature_pos='post'):
        super(SimpleConvNet, self).__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            MaskLayer(16, tau),
            #StepLayer(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            MaskLayer(32, tau),
            #StepLayer(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            MaskLayer(64, tau),
            #StepLayer(),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            MaskLayer(128, tau),
            #StepLayer(),
        ]
        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.mask_layers = []
        for module in self.modules():
            if isinstance(module, MaskLayer):
                self.mask_layers.append(module)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if feature_pos not in ['pre', 'post', 'logits']:
            raise ValueError(feature_pos)

        self.feature_pos = feature_pos

    def forward(self, x, logits_only=True):
        pre_gap_feats = self.extracter(x)
        post_gap_feats = self.avgpool(pre_gap_feats)
        post_gap_feats = torch.flatten(post_gap_feats, 1)
        logits = self.fc(post_gap_feats)

        if logits_only:
            return logits
        elif self.feature_pos == 'pre':
            feats = pre_gap_feats
        elif self.feature_pos == 'post':
            feats = post_gap_feats
        else:
            feats = logits
        return logits, feats


class SimpleConvNetPure(nn.Module):
    def __init__(self, num_classes=10, kernel_size=7, feature_pos='post'):
        super(SimpleConvNetPure, self).__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # MaskLayer(16, tau),
            #StepLayer(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # MaskLayer(32, tau),
            #StepLayer(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # MaskLayer(64, tau),
            #StepLayer(),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            # MaskLayer(128, tau),
            #StepLayer(),
        ]
        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.mask_layers = []
        # for module in self.modules():
        #     if isinstance(module, MaskLayer):
        #         self.mask_layers.append(module)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if feature_pos not in ['pre', 'post', 'logits']:
            raise ValueError(feature_pos)

        self.feature_pos = feature_pos

    def forward(self, x, logits_only=True):
        pre_gap_feats = self.extracter(x)
        post_gap_feats = self.avgpool(pre_gap_feats)
        post_gap_feats = torch.flatten(post_gap_feats, 1)
        logits = self.fc(post_gap_feats)

        if logits_only:
            return logits
        elif self.feature_pos == 'pre':
            feats = pre_gap_feats
        elif self.feature_pos == 'post':
            feats = post_gap_feats
        else:
            feats = logits
        return logits, feats


def insert_masks_SimpleConvNet(model, tau):
    new_layers = []

    for layer in model.extracter:
        new_layers.append(layer)

        if isinstance(layer, (nn.ReLU, nn.Tanh)):
            prev = new_layers[-2]
            if isinstance(prev, nn.BatchNorm2d):
                channels = prev.num_features
            elif isinstance(prev, nn.Conv2d):
                channels = prev.out_channels
            else:
                raise RuntimeError("Cannot infer channels before ReLU")

            new_layers.append(MaskLayer(channels, tau))

    model.extracter = nn.Sequential(*new_layers)

    # refresh mask_layers list
    model.mask_layers = [
        m for m in model.modules() if isinstance(m, MaskLayer)
    ]

    # return model



# ======================================================================================================
# ===================================== For MulticolorMNIST: model =====================================

# Based on https://github.com/zhihengli-UR/DebiAN/blob/main/models/simple_cls.py
# (which, in its turn, is taken from https://github.com/alinlab/LfF/blob/master/module/mlp.py)

# Adding masks to the MLP:
class MaskedMLP(nn.Module):
    def __init__(self, num_class=10, tau=1.0, return_feat=False):
        super(MaskedMLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3 * 28*28, 100),
            nn.ReLU(inplace=True),
            MaskLayer(100, tau, connected_to_linear_layer=True),              

            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            MaskLayer(100, tau, connected_to_linear_layer=True),              

            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            MaskLayer(100, tau, connected_to_linear_layer=True),              
        )
        # self.identity = nn.Identity()                   
        self.classifier = nn.Linear(100, num_class)
        self.return_feat = return_feat

        
        self.mask_layers = []
        for module in self.modules():
            if isinstance(module, MaskLayer):
                self.mask_layers.append(module)


    def forward(self, x):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        x = self.classifier(x)

        if self.return_feat:
            return x, feat
        else:
            return x
        


# ======================================================================================================
# =============================== For CelebA and Corrupted-CIFAR10: model ==============================

def insert_masks_resnet18(model, tau):


        def get_module_name(model, target_layer):
            for name, module in model.named_modules():
                if module is target_layer:
                    return name
            return None

        def append_mask_and_name(model, mask, layer):
            model.mask_layers.append(mask)
            model.names_masked_layers.append(get_module_name(model, layer))
        


        # model.relu = nn.Sequential(model.relu, MaskLayer(64, tau))
        # append_mask_and_name(model, mask=model.relu[1], layer=model.relu)


        # ======== LAYER 1 ========

        # BasicBlock 0
        model.layer1[0].conv2 = nn.Sequential(MaskLayer(64, tau), model.layer1[0].conv2)
        append_mask_and_name(model, mask=model.layer1[0].conv2[0], layer=model.layer1[0].conv2)

        model.layer1[0] = nn.Sequential(model.layer1[0], MaskLayer(64, tau))
        append_mask_and_name(model, mask=model.layer1[0][1], layer=model.layer1[0])


        # BasicBlock 1
        model.layer1[1].conv2 = nn.Sequential(MaskLayer(64, tau), model.layer1[1].conv2)
        append_mask_and_name(model, mask=model.layer1[1].conv2[0], layer=model.layer1[1].conv2)

        model.layer1[1] = nn.Sequential(model.layer1[1], MaskLayer(64, tau))
        append_mask_and_name(model, mask=model.layer1[1][1], layer=model.layer1[1])


        # ======== LAYER 2 ========

        # BasicBlock 0
        model.layer2[0].conv2 = nn.Sequential(MaskLayer(128, tau), model.layer2[0].conv2)
        append_mask_and_name(model, mask=model.layer2[0].conv2[0], layer=model.layer2[0].conv2)

        model.layer2[0] = nn.Sequential(model.layer2[0], MaskLayer(128, tau))
        append_mask_and_name(model, mask=model.layer2[0][1], layer=model.layer2[0])

        # BasicBlock 1
        model.layer2[1].conv2 = nn.Sequential(MaskLayer(128, tau), model.layer2[1].conv2)
        append_mask_and_name(model, mask=model.layer2[1].conv2[0], layer=model.layer2[1].conv2)

        model.layer2[1] = nn.Sequential(model.layer2[1], MaskLayer(128, tau))
        append_mask_and_name(model, mask=model.layer2[1][1], layer=model.layer2[1])


        # ======== LAYER 3 ========

        # BasicBlock 0
        model.layer3[0].conv2 = nn.Sequential(MaskLayer(256, tau), model.layer3[0].conv2)
        append_mask_and_name(model, mask=model.layer3[0].conv2[0], layer=model.layer3[0].conv2)

        model.layer3[0] = nn.Sequential(model.layer3[0], MaskLayer(256, tau))
        append_mask_and_name(model, mask=model.layer3[0][1], layer=model.layer3[0])

        # BasicBlock 1
        model.layer3[1].conv2 = nn.Sequential(MaskLayer(256, tau), model.layer3[1].conv2)
        append_mask_and_name(model, mask=model.layer3[1].conv2[0], layer=model.layer3[1].conv2)

        model.layer3[1] = nn.Sequential(model.layer3[1], MaskLayer(256, tau))
        append_mask_and_name(model, mask=model.layer3[1][1], layer=model.layer3[1])


        # ======== LAYER 4 ========

        # BasicBlock 0
        model.layer4[0].conv2 = nn.Sequential(MaskLayer(512, tau), model.layer4[0].conv2)
        append_mask_and_name(model, mask=model.layer4[0].conv2[0], layer=model.layer4[0].conv2)

        model.layer4[0] = nn.Sequential(model.layer4[0], MaskLayer(512, tau))
        append_mask_and_name(model, mask=model.layer4[0][1], layer=model.layer4[0])

        # BasicBlock 1
        model.layer4[1].conv2 = nn.Sequential(MaskLayer(512, tau), model.layer4[1].conv2)
        append_mask_and_name(model, mask=model.layer4[1].conv2[0], layer=model.layer4[1].conv2)

        model.layer4[1] = nn.Sequential(model.layer4[1], MaskLayer(512, tau))
        append_mask_and_name(model, mask=model.layer4[1][1], layer=model.layer4[1])


        # return model



# ======================================================================================================
# ======================================= For CivilComments: model =====================================

def insert_masks_bert(module, list_mask_layers, names_masked_layers=None, prefix="", where='beforeActivLN', layers_to_mask='ALL_LINEAR_LAYERS', initial_tau=1):
    
    import transformers

    # ---------------------- Adding mask BEFORE nonlinearities / LayerNorm ----------------------
    if where == 'beforeActivLN':

        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix != '' else name
            if isinstance(child, nn.Linear) and name not in ['classifier', 'fc']:
                if (layers_to_mask == 'ALL_LINEAR_LAYERS') or ('attention' not in full_name) or (layers_to_mask == 'ALL_LINEAR_EXCEPT_QKV' and ('query' not in full_name) and ('key' not in full_name) and ('value' not in full_name)):
                    setattr(module, name, 
                            nn.Sequential(child, MaskLayer(child.out_features, tau=initial_tau, connected_to_linear_layer=True)))
                    print(full_name)
                    list_mask_layers.append(getattr(module,name)[1])
                    if names_masked_layers is not None:
                        names_masked_layers.append(full_name)
            else:
                insert_masks_bert(child, list_mask_layers, names_masked_layers, prefix=full_name, where=where, layers_to_mask=layers_to_mask)
    # -------------------------------------------------------------------------------------------

    # ---------------------- Adding mask AFTER nonlinearities / LayerNorm ----------------------
    elif where == 'afterActivLN':

        children = list(module.named_children())
        i = 0

        while i < len(children):

            name, child = children[i]
            full_name = f"{prefix}.{name}" if prefix != '' else name

            # Case 1: Linear with successor GELU/ReLU/Tanh/LayerNorm
            if isinstance(child, nn.Linear) and name not in ['classifier', 'fc'] and i+1 < len(children):
                next_name, next_child = children[i+1]
                
                if isinstance(next_child, (nn.LayerNorm, nn.GELU, nn.ReLU, nn.Tanh, transformers.activations.GELUActivation)):
                    if (layers_to_mask == 'ALL_LINEAR_LAYERS') or ('attention' not in f"{prefix}.{next_name}") or (layers_to_mask == 'ALL_LINEAR_EXCEPT_QKV' and ('query' not in f"{prefix}.{next_name}") and ('key' not in f"{prefix}.{next_name}") and ('value' not in f"{prefix}.{next_name}")):
                        setattr(module, next_name,
                                nn.Sequential(next_child, MaskLayer(child.out_features, tau=initial_tau, connected_to_linear_layer=True)))

                        print(f"{prefix}.{next_name}")
                        # list_mask_layers.append(seq[2])
                        list_mask_layers.append(getattr(module,next_name)[1])
                        names_masked_layers.append(f"{prefix}.{next_name}")
                    
                    i += 2
                    continue

            # Case 2: Linear lqyer not directly followed by LayerNorm/activation 
            if isinstance(child, nn.Linear) and name not in ['classifier', 'fc']:

                if (layers_to_mask == 'ALL_LINEAR_LAYERS') or ('attention' not in full_name) or (layers_to_mask == 'ALL_LINEAR_EXCEPT_QKV' and ('query' not in full_name) and ('key' not in full_name) and ('value' not in full_name)):
                    setattr(module, name, 
                            nn.Sequential(child, MaskLayer(child.out_features, tau=initial_tau, connected_to_linear_layer=True)))
                    print(full_name)
                    list_mask_layers.append(getattr(module,name)[1])
                    names_masked_layers.append(full_name)
                
            else:
                insert_masks_bert(child, list_mask_layers, names_masked_layers, prefix=full_name, where=where, layers_to_mask=layers_to_mask)

            i += 1
    # -------------------------------------------------------------------------------------------

    # ---------------------- Adding mask AFTER nonlinearities and BEFORE LayerNorm ----------------------
    elif where == 'afterActivBeforeLN':

        children = list(module.named_children())
        i = 0

        while i < len(children):

            name, child = children[i]
            full_name = f"{prefix}.{name}" if prefix != '' else name

            # Case 1: Linear with successor GELU/ReLU/Tanh/LayerNorm
            if isinstance(child, nn.Linear) and name not in ['classifier', 'fc'] and i+1 < len(children):
                next_name, next_child = children[i+1]

                if isinstance(next_child, nn.LayerNorm):
                    if (layers_to_mask == 'ALL_LINEAR_LAYERS') or ('attention' not in f"{prefix}.{next_name}") or (layers_to_mask == 'ALL_LINEAR_EXCEPT_QKV' and ('query' not in f"{prefix}.{next_name}") and ('key' not in f"{prefix}.{next_name}") and ('value' not in f"{prefix}.{next_name}")):
                        setattr(module, name,
                                nn.Sequential(child, MaskLayer(child.out_features, tau=1., connected_to_linear_layer=True)))

                        print(full_name)
                        # list_mask_layers.append(seq[2])
                        list_mask_layers.append(getattr(module,name)[1])
                        names_masked_layers.append(full_name)
                    
                    i += 2
                    continue                    

                
                if isinstance(next_child, (nn.GELU, nn.ReLU, nn.Tanh, transformers.activations.GELUActivation)):
                    
                    if (layers_to_mask == 'ALL_LINEAR_LAYERS') or ('attention' not in f"{prefix}.{next_name}") or (layers_to_mask == 'ALL_LINEAR_EXCEPT_QKV' and ('query' not in f"{prefix}.{next_name}") and ('key' not in f"{prefix}.{next_name}") and ('value' not in f"{prefix}.{next_name}")):
                        setattr(module, next_name,
                                nn.Sequential(next_child, MaskLayer(child.out_features, tau=1., connected_to_linear_layer=True)))

                        print(f"{prefix}.{next_name}")
                        # list_mask_layers.append(seq[2])
                        list_mask_layers.append(getattr(module,next_name)[1])
                        names_masked_layers.append(f"{prefix}.{next_name}")
                    
                    i += 2
                    continue

            # Case 2: Linear lqyer not directly followed by LayerNorm/activation 
            if isinstance(child, nn.Linear) and name not in ['classifier', 'fc']:

                if (layers_to_mask == 'ALL_LINEAR_LAYERS') or ('attention' not in full_name) or (layers_to_mask == 'ALL_LINEAR_EXCEPT_QKV' and ('query' not in full_name) and ('key' not in full_name) and ('value' not in full_name)):
                    setattr(module, name, 
                            nn.Sequential(child, MaskLayer(child.out_features, tau=1., connected_to_linear_layer=True)))
                    print(full_name)
                    list_mask_layers.append(getattr(module,name)[1])
                    names_masked_layers.append(full_name)
                
            else:
                insert_masks_bert(child, list_mask_layers, names_masked_layers, prefix=full_name, where=where, layers_to_mask=layers_to_mask)

            i += 1
    # -------------------------------------------------------------------------------------------