import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from src.train_eval_functions import pretrain_PH
from torch.utils.data import DataLoader, Subset
import copy



# ====================================================================


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: Outputs 1 if input >= 0, otherwise 0.
        """
        ctx.save_for_backward(input)  
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Identity gradient (treating y = x).
        """
        input, = ctx.saved_tensors
        grad_input = torch.ones_like(input)  
        return grad_input * grad_output



class MaskLayer(nn.Module):
    
    def __init__(self, num_neurons, tau, init_m_i = 0.0, device="cuda:0", connected_to_linear_layer=False):
        super(MaskLayer, self).__init__()

        self.tau = tau
        self.device = device
        self.m_i = nn.Parameter(torch.full((num_neurons,), init_m_i, device=self.device), requires_grad=True)
        self.mode = 'masked'
        self.connected_to_linear_layer = connected_to_linear_layer

    def forward(self, x):

        if self.mode == 'vanilla':
            m_hat_step = torch.ones_like(self.m_i, device=self.device)
            
        elif self.mode == 'pruned':
            m_hat_step = StepFunction.apply(self.m_i)

        elif self.mode == 'masked':
            self.m_hat = torch.sigmoid(self.m_i / self.tau)
            m_hat_step = StepFunction.apply(self.m_hat - 0.5)   # We can try other thresholds besides 0.5


        self.m_hat_step = m_hat_step
        m_hat_step_v = m_hat_step.view(1, -1, 1, 1)

        if not self.connected_to_linear_layer:
            masked_output = x * m_hat_step_v
        else:
            masked_output = x * m_hat_step

        return masked_output
    

# ====================================================================


class Hook():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module[1].register_forward_hook(self.hook_fn)
			self.name = module[0]
		else:
			self.hook = module[1].register_backward_hook(self.hook_fn)
			self.name = module[0]
	def hook_fn(self, module, input, output):
		self.input = input
		self.output = output
	def close(self):
		self.hook.remove()

class Privacy_head(torch.nn.Module):    # AUXILIARY CLASSIFIER (based in the "IRENE" paper: https://arxiv.org/abs/2210.00891)
	def __init__(self, bottleneck_layer, head_structure):
		super(Privacy_head, self).__init__()
		self.bottleneck = Hook(bottleneck_layer, backward=False)
		self.classifier = head_structure
	def forward(self):
		x = self.bottleneck.output.clone().detach()
		if len(x.size())>2:
			x = x.view(-1, np.prod((x.size())[1:]))
		x = self.classifier(x)
		return x
	def forward_attached(self):
		x = self.bottleneck.output
		if len(x.size())>2:
			x = x.view(-1, np.prod((x.size())[1:]))
		x = self.classifier(x)
		return x


# ====================================================================


def mask_optimizer(model, lr_m_i=1e-2, wd=1e-4, momentum=0.9, optim='SGD'):
    m_i_params = []
    
    for module in model.mask_layers:
        m_i_params.append(module.m_i)  


    if optim == 'SGD':
        optimizer3 = torch.optim.SGD([
            {'params': m_i_params, 'lr': lr_m_i, 'momentum': momentum, 'weight_decay': wd}
            ])
    else:   # optim == 'adam'
        optimizer3 = torch.optim.Adam([
            {'params': m_i_params}
            ])
        
    
    return optimizer3


# ====================================================================


# NOTE: Below, we compute the number of neurons/filters pruned.
#       In the paper, we focus on the number of individual weights pruned;
#       for that, we rely on the package 'simplify' (https://pypi.org/project/torch-simplify/)
#       (the computation of the "unstructured" sparsity is not included in the present code).

def calculate_global_sparsity(model, verbose=True):
    total_params = 0
    zero_params = 0

    for module in model.mask_layers:
        # if mode=='pruned':
        m_i = module.m_i.data
        total_params += m_i.numel()
        zero_params += (m_i < 0).sum().item()

    sparsity_percentage = (zero_params / total_params) * 100 if total_params > 0 else 0
    if verbose:
        print(f"Network Sparsity (pruned): {sparsity_percentage:.2f}% ({zero_params}/{total_params} zero values) [Nb. of pruned neurons/filters, not individual weights!]")
    return sparsity_percentage


# ====================================================================


def update_tau_periodic(model, args, loader, epoch, tau, period=5, factor=0.5, refine_ep=10):
    """
    Periodically reduces tau every `period` epochs, by multiplying it by `factor` < 1.
    """
    if epoch % period == 0:
        tau *= factor
        print(f"Epoch {epoch}: Periodic update, reducing tau to {tau:.10f}")
        for ep in range(1, refine_ep+1):
                pretrain_PH(model, args, loader, mode='pruned')

    return tau


# =====================================================================


# --------------- FOR EXPS ON UNSUPERVISED DEBIASING ---------------

def overwrite_bias_labels_with_predictions(identification_model, loader, device='cuda:0'):

    subset = loader.dataset
    parent = subset.dataset
    subset_indices = list(subset.indices)  # indices in the parent dataset

    # Make a non-shuffling loader for the subset
    eval_loader = DataLoader(copy.deepcopy(subset), batch_size=loader.batch_size, shuffle=False,
                            num_workers=loader.num_workers, pin_memory=loader.pin_memory)

    # Collect predictions in the same order as `subset_indices`
    all_preds = []
    all_labels = []
    all_private_labels = []

    identification_model.eval()
    with torch.no_grad():
        for inputs, labels, private_labels in eval_loader:
            inputs = inputs.to(device)
            outputs = identification_model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_private_labels.append(private_labels.cpu())

    all_preds = torch.cat(all_preds)        # length == len(subset)
    all_labels = torch.cat(all_labels)
    all_private_labels = torch.cat(all_private_labels)

    # Write predictions back into parent dataset at the correct positions
    loader.dataset.dataset.original_biased_targets = loader.dataset.dataset.biased_targets.clone()
    loader.dataset.dataset.biased_targets[subset_indices] = all_preds
    
    print(f'\n================ ANALYSIS OF IDENTIFICATION MODEL  =======================\n')

    print('------ OVERALL ------')
    print('Acc. on TARGET label: ', (all_preds == all_labels).float().mean().item() * 100)
    print('Acc. on BIAS label:   ', (all_preds == all_private_labels).float().mean().item() * 100)

    print('------ ON ALIGNED ONLY ------')
    aligned_filter = (all_labels == all_private_labels)
    print('Acc. on TARGET label: ', (all_preds[aligned_filter] == all_labels[aligned_filter]).float().mean().item() * 100)
    print('Acc. on BIAS label:   ', (all_preds[aligned_filter] == all_private_labels[aligned_filter]).float().mean().item() * 100)

    print('------ ON CONFLICTING ONLY ------')
    conflicting_filter = (all_labels != all_private_labels)
    print('Acc. on TARGET label: ', (all_preds[conflicting_filter] == all_labels[conflicting_filter]).float().mean().item() * 100)
    print('Acc. on BIAS label:   ', (all_preds[conflicting_filter] == all_private_labels[conflicting_filter]).float().mean().item() * 100)


    return all_preds, all_labels, all_private_labels



def overwrite_attr_column_with_predictions(
    model,
    loader,
    device,
    attr_df_name="attr_df",
    column="Male"
):

    model.eval()

    subset = loader.dataset
    parent = getattr(subset, "dataset", subset)
    subset_indices = list(getattr(subset, "indices", range(len(subset))))

    if not hasattr(parent, attr_df_name):
        raise AttributeError(f"Parent dataset has no '{attr_df_name}' attribute (expected a pandas DataFrame).")

    df = getattr(parent, attr_df_name)

    # Save original column
    orig_col_name = f"{column}_original"
    df[orig_col_name] = copy.deepcopy(df[column])

    eval_loader = DataLoader(
        copy.deepcopy(subset),
        batch_size= 64, # getattr(loader, "batch_size", 1),
        shuffle=False,
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", False),
        collate_fn=getattr(loader, "collate_fn", None),
    )

    all_preds = []
    all_labels = []
    all_private_labels = []


    with torch.no_grad():
        for inputs, labels, private_labels in eval_loader:

            inputs = inputs.to(device)
            outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1).long()

            all_preds.append(preds.cpu())

            if labels is not None:
                all_labels.append(labels.cpu())
                all_private_labels.append(private_labels.cpu())
        

    if len(all_preds) == 0:
        raise RuntimeError("No predictions collected. Check that the loader yields input tensors.")


    all_preds = torch.cat(all_preds).numpy()    # shape == (len(subset),)
    all_labels = torch.cat(all_labels).numpy() if len(all_labels) > 0 else None
    all_private_labels = torch.cat(all_private_labels).numpy() if len(all_private_labels) > 0 else None



    parent_indexer = subset_indices
    df.iloc[parent_indexer, df.columns.get_loc(column)] = all_preds

    setattr(loader.dataset, attr_df_name, df)


    print(f'\n================ ANALYSIS OF IDENTIFICATION MODEL  =======================\n')

    print('------ OVERALL ------')
    print('Acc. on TARGET label: ', (all_preds == all_labels).mean().item() * 100)
    print('Acc. on BIAS label:   ', (all_preds == all_private_labels).mean().item() * 100)

    print('------ ON ALIGNED ONLY ------')
    aligned_filter = (all_labels == all_private_labels)
    print('Acc. on TARGET label: ', (all_preds[aligned_filter] == all_labels[aligned_filter]).mean().item() * 100)
    print('Acc. on BIAS label:   ', (all_preds[aligned_filter] == all_private_labels[aligned_filter]).mean().item() * 100)

    print('------ ON CONFLICTING ONLY ------')
    conflicting_filter = (all_labels != all_private_labels)
    print('Acc. on TARGET label: ', (all_preds[conflicting_filter] == all_labels[conflicting_filter]).mean().item() * 100)
    print('Acc. on BIAS label:   ', (all_preds[conflicting_filter] == all_private_labels[conflicting_filter]).mean().item() * 100)


    print(f'\n================ ON EACH OF FOUR GROUPS  =======================\n')

    print('------ DARK-HAIRED MEN ------')
    dark_men_filter = ((all_labels==0) & (all_private_labels==0))
    print('Acc. on TARGET label: ', (all_preds[dark_men_filter] == all_labels[dark_men_filter]).mean().item() * 100)
    print('Acc. on BIAS label:   ', (all_preds[dark_men_filter] == all_private_labels[dark_men_filter]).mean().item() * 100)

    print('------ DARK-HAIRED WOMEN ------')
    dark_women_filter = ((all_labels==0) & (all_private_labels==1))
    print('Acc. on TARGET label: ', (all_preds[dark_women_filter] == all_labels[dark_women_filter]).mean().item() * 100)
    print('Acc. on BIAS label:   ', (all_preds[dark_women_filter] == all_private_labels[dark_women_filter]).mean().item() * 100)

    print('------ BLOND MEN ------')
    blond_men_filter = ((all_labels==1) & (all_private_labels==0))
    print('Acc. on TARGET label: ', (all_preds[blond_men_filter] == all_labels[blond_men_filter]).mean().item() * 100)
    print('Acc. on BIAS label:   ', (all_preds[blond_men_filter] == all_private_labels[blond_men_filter]).mean().item() * 100)

    print('------ BLONDE WOMEN ------')
    blond_women_filter = ((all_labels==1) & (all_private_labels==1))
    print('Acc. on TARGET label: ', (all_preds[blond_women_filter] == all_labels[blond_women_filter]).mean().item() * 100)
    print('Acc. on BIAS label:   ', (all_preds[blond_women_filter] == all_private_labels[blond_women_filter]).mean().item() * 100)


    return all_preds, all_labels, all_private_labels
