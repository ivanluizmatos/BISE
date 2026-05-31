import numpy as np
import torch
from src.auxiliary import AverageMeter, accuracy
from src.auxiliary import update_meters_MulticolorMNIST, update_meters_CivilComments
from src.losses import computeCEloss
from src.aux_and_reimplementations import update_layernorm_masks



def train_PH_prune(model, args, train_loader, balanced_loss, modif_MI, return_values_per_group = False, logs_dict=None):
    model.eval()    # NOTE: We use model.eval(), as we train the mask but not the model itself !! To update batchnorm, use model.train()


    for _, param in model.named_parameters():
        param.requires_grad = False
    

    for mask_layer in model.mask_layers:
        mask_layer.mode = 'masked'             # IMPORTANT!
        mask_layer.m_i.requires_grad = True      # IMPORTANT!


    if args.dataset != 'MulticolorMNIST':
    
        args.PH.train()

        loss_task_tot = AverageMeter('Loss_task', ':.4e')   
        loss_private_tot = AverageMeter('Loss', ':.4e')
        loss_tot = AverageMeter('Loss', ':.4e')
        MI_tot = AverageMeter('Regu', ':.4e')
        private_top1 = AverageMeter('Acc@1', ':6.2f')
        task_top1 = AverageMeter('Acc@1_task', ':6.2f')     



        if args.dataset == 'CivilComments':
            meters = {
                '0_0': AverageMeter('Acc@1', ':6.2f'),
                '0_1': AverageMeter('Acc@1', ':6.2f'),
                '1_0': AverageMeter('Acc@1', ':6.2f'),
                '1_1': AverageMeter('Acc@1', ':6.2f'),
            }


        for batch_data in train_loader:

            if args.dataset != 'CivilComments':
                data, target, private_label = batch_data
            else:
                data, target, group, private_label = batch_data
                group = group.to(args.device)
                update_layernorm_masks(model)



            data = data.to(args.device)
            target = target.to(args.device)
            private_label = private_label.to(args.device)
            output= model(data)
            output_private = args.PH()


            # -------------------- C.E. loss selection --------------------
            loss_task = computeCEloss(args, output, target, private_label)
            # -------------------------------------------------------------

            loss_private = args.criterion(output_private, private_label)

            # --------- modified: M.I. implementation selection ---------

            # NOTE: automatizing for different datasets
            if args.dataset in ['Bmnist', 'Cifar10C']:
                nb_classes = 10
            elif args.dataset in ['CelebA', 'CivilComments']:
                nb_classes = 2


            is_conflicting = (private_label != target)


            if modif_MI == 0:   # v0: original implementation (as in the "IRENE" paper: https://arxiv.org/abs/2210.00891)

                MI = args.MI(args.PH, private_label)
                MI_tot.update(MI.item(), data.size(0))

            elif modif_MI == 1 and (is_conflicting).nonzero().size(0) > 0:  # v1: only compute M.I. based on bias-conflicting samples

                MI = args.MI(args.PH, private_label, conflict_mask=(is_conflicting))
                MI_tot.update(MI.item(), data.size(0))   # SHOULD WE AVERAGE OVER (private_label != target).sum() INSTEAD OF data.size()?

            elif modif_MI == 2 and (is_conflicting).nonzero().size(0) > 0:  # v2: similar v1, but also use a portion of bias-aligned samples, to compute M.I.

                conflict_mask = (is_conflicting)

                indices_aligned = torch.where(~conflict_mask)[0]
                shuffle_order = torch.randperm(len(indices_aligned))
                indices_aligned_shuffled = indices_aligned[shuffle_order]

                B = len(data)
                num_aligned_to_take = round(B * (1 - args.correlation) / (nb_classes-1))

                indices_to_make_true = indices_aligned_shuffled[:min(num_aligned_to_take, len(indices_aligned_shuffled))]
                conflict_mask[indices_to_make_true] = True

                MI = args.MI(args.PH, private_label, conflict_mask=conflict_mask)
                MI_tot.update(MI.item(), data.size(0)) 

            else:
                MI = torch.tensor(0.)
            # ---------------------


            loss_task_tot.update(loss_task.item(), data.size(0))        
            loss_private_tot.update(loss_private.item(), data.size(0))

            # MI_tot.update(MI.item(), data.size(0))

            if isinstance(MI, torch.Tensor):
                MI = [MI]

            args.mask_optimizer.zero_grad()
            loss = args.alpha * loss_task + args.gamma * torch.mean(torch.stack(MI))
            loss.backward()

            args.PH_optimizer.zero_grad()
            loss_private.backward()

            args.mask_optimizer.step()


            args.PH_optimizer.step()
            args.PH_optimizer.zero_grad()

            loss_tot.update(loss.item(), data.size(0))

            acc1_private = accuracy(output_private, private_label, topk=(1,))
            private_top1.update(acc1_private[0], data.size(0))

            acc1_task = accuracy(output, target, topk=(1,))     
            task_top1.update(acc1_task[0], data.size(0))        



            if args.dataset == 'CivilComments':
                update_meters_CivilComments(output, target, private_label, meters)


            # # logging_batches  
            # if LOGGING_BATCHES and logs_dict is not None:
            #     logs_dict['train']['logs_per_batch']['loss_task'].append(loss_task.item())
            #     logs_dict['train']['logs_per_batch']['MI'].append(torch.mean(torch.stack(MI)).item())
            #     logs_dict['train']['logs_per_batch']['loss_tot'].append(loss.item())
            #     logs_dict['train']['logs_per_batch']['loss_private'].append(loss_private.item())
            #     logs_dict['train']['logs_per_batch']['acc_task'].append(acc1_task[0].item())
            #     logs_dict['train']['logs_per_batch']['acc_private'].append(acc1_private[0].item())
            #     logs_dict['train']['logs_per_batch']['sparsity'].append(calculate_global_sparsity(model,verbose=False))


        print(f'Train PH and Mask            :  loss_task={loss_task_tot.avg}, top1={task_top1.avg.item()}, MI = {MI_tot.avg}, top1_private={private_top1.avg.item()}      (num_batches={len(train_loader)})')  



        if args.dataset == 'CivilComments':

            update_layernorm_masks(model)

            avg_acc_groups = np.mean([meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg]).item()

            weighted_avg_acc_groups = np.average([meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg], 
                                                weights=[args.proportions[0][0], args.proportions[0][1], args.proportions[1][0], args.proportions[1][1]]).item()
            
            worst_acc_groups = np.min([meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg]).item()

            print(f" =================>   Accs:   0_0 = {meters['0_0'].avg:.4f},   0_1 = {meters['0_1'].avg:.4f},   1_0 = {meters['1_0'].avg:.4f},   1_1 = {meters['1_1'].avg:.4f},   " +
                f"avg={avg_acc_groups:.4f},   weighted_avg={weighted_avg_acc_groups:.4f},   worst={worst_acc_groups:.4f}")


            # TODO -- adapt code to return the following, for CivilComments:
            # return loss_task_tot.avg, task_top1.avg.item(), MI_tot.avg, private_top1.avg.item(), (meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg, avg_acc_groups, weighted_avg_acc_groups, worst_acc_groups)



        return loss_task_tot.avg, task_top1.avg.item(), MI_tot.avg, private_top1.avg.item()     




    else:   # SPECIFIC IMPLEMENTATION FOR MulticolorMNIST

        args.PH_left.train()
        args.PH_right.train()

        loss_task_tot = AverageMeter('Loss_task', ':.4e')   

        loss_private_tot_left = AverageMeter('Loss', ':.4e')
        loss_private_tot_right = AverageMeter('Loss', ':.4e')

        loss_tot = AverageMeter('Loss', ':.4e')

        MI_tot_left = AverageMeter('Regu', ':.4e')
        MI_tot_right = AverageMeter('Regu', ':.4e')

        private_top1_left = AverageMeter('Acc@1', ':6.2f')
        private_top1_right = AverageMeter('Acc@1', ':6.2f')


        task_top1 = AverageMeter('Acc@1_task', ':6.2f')     


        meters = {
            'aligned_aligned':         AverageMeter('Acc@1', ':6.2f'),
            'aligned_conflicting':     AverageMeter('Acc@1', ':6.2f'),
            'conflicting_aligned':     AverageMeter('Acc@1', ':6.2f'),
            'conflicting_conflicting': AverageMeter('Acc@1', ':6.2f')
            }
        

        for data, target, private_label in train_loader:                                                    
            
            private_label = private_label.squeeze().T
            private_label = private_label.to(args.device)
            private_label_left, private_label_right = private_label[0], private_label[1]
            
            data = data.to(args.device)
            target = target.to(args.device)
            private_label_left, private_label_right = private_label_left.to(args.device), private_label_right.to(args.device)

            output= model(data)

            output_private_left = args.PH_left()
            output_private_right = args.PH_right()

            # -------------------- C.E. loss selection --------------------
            loss_task = computeCEloss(args, output, target, private_label)
            # -------------------------------------------------------------

            loss_private_left = args.criterion(output_private_left, private_label_left)
            loss_private_right = args.criterion(output_private_right, private_label_right)


            # --------- modified: M.I. implementation selection ---------

            nb_classes = 10

            # assert modif_MI == 0

            if modif_MI == 0:   # v0: original implementation (as in the "IRENE" paper: https://arxiv.org/abs/2210.00891)

                MI_left = args.MI(args.PH_left, private_label_left)
                MI_tot_left.update(MI_left.item(), data.size(0))

                MI_right = args.MI(args.PH_right, private_label_right)
                MI_tot_right.update(MI_right.item(), data.size(0))



            # TODO: maybe implement variations of M.I. computation also for MulticolorMNIST

            elif modif_MI == 1:

                if (private_label_left != target).nonzero().size(0) > 0:  # v1: only compute M.I. based on bias-conflicting samples
                    MI_left = args.MI(args.PH_left, private_label_left, conflict_mask=(private_label_left != target))
                    MI_tot_left.update(MI_left.item(), data.size(0))   # SHOULD WE AVERAGE OVER (private_label != target).sum() INSTEAD OF data.size()?
                else:
                    MI_left = torch.tensor(0.)

                if (private_label_right != target).nonzero().size(0) > 0:  # v1: only compute M.I. based on bias-conflicting samples
                    MI_right = args.MI(args.PH_right, private_label_right, conflict_mask=(private_label_right != target))
                    MI_tot_right.update(MI_right.item(), data.size(0))   # SHOULD WE AVERAGE OVER (private_label != target).sum() INSTEAD OF data.size()?
                else:
                    MI_right = torch.tensor(0.)


            else:
                MI = torch.tensor(0.)
            # ---------------------


            loss_task_tot.update(loss_task.item(), data.size(0))        

            loss_private_tot_left.update(loss_private_left.item(), data.size(0))
            loss_private_tot_right.update(loss_private_right.item(), data.size(0))


            if isinstance(MI_left, torch.Tensor):
                MI_left = [MI_left]
            if isinstance(MI_right, torch.Tensor):
                MI_right = [MI_right]

            args.mask_optimizer.zero_grad()
            loss = args.alpha * loss_task + args.gamma * (torch.mean(torch.stack(MI_left)) + torch.mean(torch.stack(MI_right)))
            loss.backward()

            args.PH_left_optimizer.zero_grad()
            loss_private_left.backward()

            args.PH_right_optimizer.zero_grad()
            loss_private_right.backward()


            args.mask_optimizer.step()


            args.PH_left_optimizer.step()
            args.PH_left_optimizer.zero_grad()

            args.PH_right_optimizer.step()
            args.PH_right_optimizer.zero_grad()


            loss_tot.update(loss.item(), data.size(0))

            acc1_private_left = accuracy(output_private_left, private_label_left, topk=(1,))
            private_top1_left.update(acc1_private_left[0], data.size(0))

            acc1_private_right = accuracy(output_private_right, private_label_right, topk=(1,))
            private_top1_right.update(acc1_private_right[0], data.size(0))


            acc1_task = accuracy(output, target, topk=(1,))     
            task_top1.update(acc1_task[0], data.size(0))        


            update_meters_MulticolorMNIST(output, target, private_label, meters)


        print(f'Train PH and Mask            :  loss_task={loss_task_tot.avg}, top1={task_top1.avg.item()}, MI = {MI_tot_left.avg, MI_tot_right.avg}, top1_private={private_top1_left.avg.item(), private_top1_right.avg.item()}      (num_batches={len(train_loader)})')   
        
        avg_acc_groups = np.mean([meters['aligned_aligned'].avg, meters['aligned_conflicting'].avg, meters['conflicting_aligned'].avg, meters['conflicting_conflicting'].avg])
        print(f" =================>   Accs:   alig_alig = {meters['aligned_aligned'].avg:.4f},   alig_conf = {meters['aligned_conflicting'].avg:.4f},   conf_alig = {meters['conflicting_aligned'].avg:.4f},   conf_conf = {meters['conflicting_conflicting'].avg:.4f},   avg={avg_acc_groups}")


        return loss_task_tot.avg, task_top1.avg.item(), (MI_tot_left.avg, MI_tot_right.avg), (private_top1_left.avg.item(), private_top1_right.avg.item())

