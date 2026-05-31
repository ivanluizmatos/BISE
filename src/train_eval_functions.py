import numpy as np
from src.auxiliary import AverageMeter, accuracy
from src.auxiliary import update_meters_MulticolorMNIST, update_meters_CivilComments
from src.aux_and_reimplementations import update_layernorm_masks
from src.losses import computeCEloss


# ====================================================================


def train_vanilla(model, args, train_loader, return_values=False):
    model.train()
    loss_task_tot = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    if args.dataset == 'MulticolorMNIST':
        meters = {
            'aligned_aligned':         AverageMeter('Acc@1', ':6.2f'),
            'aligned_conflicting':     AverageMeter('Acc@1', ':6.2f'),
            'conflicting_aligned':     AverageMeter('Acc@1', ':6.2f'),
            'conflicting_conflicting': AverageMeter('Acc@1', ':6.2f')
            }



    for mask_layer in model.mask_layers:
        mask_layer.mode = 'vanilla' 
        mask_layer.m_i.requires_grad = False

    
    for group in args.optimizer.param_groups:
        print('--------> LEARNING RATE:   ', group['lr'])


    for batch_data in train_loader:

        if args.dataset != 'CivilComments':
            data, target, private_label = batch_data
        else:
            data, target, group, private_label = batch_data
            group = group.to(args.device)



        data, target, private_label = data.to(args.device), target.to(args.device), private_label.to(args.device)
        output = model(data)
        loss_task = args.criterion(output, target)
        loss_task_tot.update(loss_task.item(), data.size(0))
        loss_task.backward()
        args.optimizer.step()
        args.optimizer.zero_grad()
        acc1 = accuracy(output, target, topk=(1,))
        top1.update(acc1[0], data.size(0))

        if args.dataset == 'MulticolorMNIST':
            private_label = private_label.squeeze().T
            update_meters_MulticolorMNIST(output, target, private_label, meters)




    print(f'Train vanilla :  loss_task = {loss_task_tot.avg}, top1 = {top1.avg.item()}      (num_batches={len(train_loader)})')
    
    
    if args.dataset == 'MulticolorMNIST':
        avg_acc_groups = np.mean([meters['aligned_aligned'].avg, meters['aligned_conflicting'].avg, meters['conflicting_aligned'].avg, meters['conflicting_conflicting'].avg])
        print(f" =================>   Accs:   alig_alig = {meters['aligned_aligned'].avg:.4f},   alig_conf = {meters['aligned_conflicting'].avg:.4f},   conf_alig = {meters['conflicting_aligned'].avg:.4f},   conf_conf = {meters['conflicting_conflicting'].avg:.4f},   avg={avg_acc_groups}")

        if return_values:
            return loss_task_tot.avg, (top1.avg.item(), meters['aligned_aligned'].avg, meters['aligned_conflicting'].avg, meters['conflicting_aligned'].avg, meters['conflicting_conflicting'].avg, avg_acc_groups, )




    if return_values:
        return loss_task_tot.avg, top1.avg.item()


# ====================================================================


def test_vanilla(model, args, loader, desc='Test vanilla', return_values=False):
    model.eval()
    loss_task_tot =  AverageMeter('Loss', ':.4e')
    top1 = AverageMeter("Acc@1", ":6.2f")
    

    if args.dataset == 'MulticolorMNIST':
        meters = {
            'aligned_aligned':         AverageMeter('Acc@1', ':6.2f'),
            'aligned_conflicting':     AverageMeter('Acc@1', ':6.2f'),
            'conflicting_aligned':     AverageMeter('Acc@1', ':6.2f'),
            'conflicting_conflicting': AverageMeter('Acc@1', ':6.2f')
            }
    


    elif args.dataset == 'CivilComments':
        meters = {
            '0_0': AverageMeter('Acc@1', ':6.2f'),
            '0_1': AverageMeter('Acc@1', ':6.2f'),
            '1_0': AverageMeter('Acc@1', ':6.2f'),
            '1_1': AverageMeter('Acc@1', ':6.2f'),
    }


    for mask_layer in model.mask_layers:
        mask_layer.mode = 'vanilla' 


    for batch_data in loader:
        
        if args.dataset != 'CivilComments':
            data, labels, private_label = batch_data
        else:
            data, labels, group, private_label = batch_data
            group = group.to(args.device)


        data, labels, private_label = data.to(args.device), labels.to(args.device), private_label.to(args.device)
        output = model(data)
        loss_task = args.criterion(output, labels)
        loss_task_tot.update(loss_task.item(), data.size(0))
        acc1 = accuracy(output, labels, topk=(1,))
        top1.update(acc1[0], data.size(0))

        if args.dataset == 'MulticolorMNIST':
            private_label = private_label.squeeze().T
            update_meters_MulticolorMNIST(output, labels, private_label, meters)



        elif args.dataset == 'CivilComments':
            update_meters_CivilComments(output, labels, private_label, meters)


    print(f'{desc} :  loss_task = {loss_task_tot.avg}, top1 = {top1.avg.item()}      (num_batches={len(loader)})')



    if args.dataset == 'MulticolorMNIST':
        avg_acc_groups = np.mean([meters['aligned_aligned'].avg, meters['aligned_conflicting'].avg, meters['conflicting_aligned'].avg, meters['conflicting_conflicting'].avg])
        print(f" =================>   Accs:   alig_alig = {meters['aligned_aligned'].avg:.4f},   alig_conf = {meters['aligned_conflicting'].avg:.4f},   conf_alig = {meters['conflicting_aligned'].avg:.4f},   conf_conf = {meters['conflicting_conflicting'].avg:.4f},   avg={avg_acc_groups}")
        
        if return_values:
            return loss_task_tot.avg, (top1.avg.item(), meters['aligned_aligned'].avg, meters['aligned_conflicting'].avg, meters['conflicting_aligned'].avg, meters['conflicting_conflicting'].avg, avg_acc_groups)



    if args.dataset == 'CivilComments':


        avg_acc_groups = np.mean([meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg]).item()

        weighted_avg_acc_groups = np.average([meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg], 
                                            weights=[args.proportions[0][0], args.proportions[0][1], args.proportions[1][0], args.proportions[1][1]]).item()
        
        worst_acc_groups = np.min([meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg]).item()

        print(f" =================>   Accs:   0_0 = {meters['0_0'].avg:.4f},   0_1 = {meters['0_1'].avg:.4f},   1_0 = {meters['1_0'].avg:.4f},   1_1 = {meters['1_1'].avg:.4f},   " +
            f"avg={avg_acc_groups:.4f},   weighted_avg={weighted_avg_acc_groups:.4f},   worst={worst_acc_groups:.4f}")


        if return_values:
            return loss_task_tot.avg , (top1.avg.item(), meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg, avg_acc_groups, weighted_avg_acc_groups, worst_acc_groups)

        else:
            pass


    if return_values:
        return loss_task_tot.avg, top1.avg.item()


# ====================================================================


def pretrain_PH(model, args, train_loader, mode='pruned'):

    model.eval()
    
    for _, param in model.named_parameters():
        param.requires_grad = False
    
    
    for mask_layer in model.mask_layers:
        mask_layer.mode = mode
        mask_layer.m_i.requires_grad = False



    if args.dataset != 'MulticolorMNIST':

        args.PH.train()
        loss_private_tot = AverageMeter('Loss', ':.4e')
        private_top1 = AverageMeter('Acc@1', ':6.2f')

        for batch_data in train_loader:

            if args.dataset != 'CivilComments':
                data, _, private_label = batch_data
            else:
                data, _, group, private_label = batch_data
                group = group.to(args.device)


            data, private_label = data.to(args.device), private_label.to(args.device)
            output = model(data)
            output_private = args.PH()
            loss_private = args.criterion(output_private, private_label)
            loss_private_tot.update(loss_private.item(),data.size(0))

            args.PH_optimizer.zero_grad()
            loss_private.backward()
            args.PH_optimizer.step()
            args.PH_optimizer.zero_grad()
            acc1_private = accuracy(output_private, private_label, topk=(1,))
            private_top1.update(acc1_private[0], data.size(0))

        print(f'Pretraining PH :  private_loss = {loss_private_tot.avg}, top1_private = {private_top1.avg.item()}      (num_batches={len(train_loader)})')

    
    else:   # SPECIFIC IMPLEMENTATION FOR MulticolorMNIST

        args.PH_left.train()
        args.PH_right.train()

        loss_private_tot_left = AverageMeter('Loss', ':.4e')
        loss_private_tot_right = AverageMeter('Loss', ':.4e')

        private_top1_left = AverageMeter('Acc@1', ':6.2f')
        private_top1_right = AverageMeter('Acc@1', ':6.2f')


        for data, labels, private_label in train_loader:

            private_label = private_label.squeeze().T
            private_label = private_label.to(args.device)
            private_label_left, private_label_right = private_label[0], private_label[1]

            data, private_label_left, private_label_right = data.to(args.device), private_label_left.to(args.device), private_label_right.to(args.device)
        
            output = model(data)

            output_private_left = args.PH_left()
            loss_private_left = args.criterion(output_private_left, private_label_left)
            loss_private_tot_left.update(loss_private_left.item(), data.size(0))

            output_private_right = args.PH_right()
            loss_private_right = args.criterion(output_private_right, private_label_right)
            loss_private_tot_right.update(loss_private_right.item(), data.size(0))


            args.PH_left_optimizer.zero_grad()
            loss_private_left.backward()
            args.PH_left_optimizer.step()
            args.PH_left_optimizer.zero_grad()
            acc1_private_left = accuracy(output_private_left, private_label_left, topk=(1,))
            private_top1_left.update(acc1_private_left[0], data.size(0))

            args.PH_right_optimizer.zero_grad()
            loss_private_right.backward()
            args.PH_right_optimizer.step()
            args.PH_right_optimizer.zero_grad()
            acc1_private_right = accuracy(output_private_right, private_label_right, topk=(1,))
            private_top1_right.update(acc1_private_right[0], data.size(0))


        print(f'Pretraining PH_left  :  private_loss = {loss_private_tot_left.avg}, top1_private = {private_top1_left.avg.item()}      (num_batches={len(train_loader)})')   
        print(f'Pretraining PH_right :  private_loss = {loss_private_tot_right.avg}, top1_private = {private_top1_right.avg.item()}      (num_batches={len(train_loader)})')   
        print('--------')




# ====================================================================


def test(model, args, loader, balanced_loss=0, mode='masked', desc = 'Testing', return_values_per_group = False):
    model.eval()
    loss_task_tot = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')


    if args.dataset == 'CivilComments':
        meters = {
            '0_0': AverageMeter('Acc@1', ':6.2f'),
            '0_1': AverageMeter('Acc@1', ':6.2f'),
            '1_0': AverageMeter('Acc@1', ':6.2f'),
            '1_1': AverageMeter('Acc@1', ':6.2f'),
        }



    for mask_layer in model.mask_layers:
        mask_layer.mode = mode 

    if args.dataset != 'MulticolorMNIST':

        private_top1 = AverageMeter('Acc@1', ':6.2f')

        for batch_data in loader:

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

            loss_task_tot.update(loss_task.item(), data.size(0))
            acc1 = accuracy(output, target, topk=(1,))
            acc1_private = accuracy(output_private, private_label, topk=(1,))
            top1.update(acc1[0], data.size(0))
            private_top1.update(acc1_private[0], data.size(0))


            if args.dataset == 'CivilComments':
                update_meters_CivilComments(output, target, private_label, meters)


        print(desc + f':  loss_task = {loss_task_tot.avg}, top1 = {top1.avg.item()}, top1_private = {private_top1.avg.item()}      (num_batches={len(loader)})')
    


        if args.dataset == 'CivilComments':

            avg_acc_groups = np.mean([meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg]).item()

            weighted_avg_acc_groups = np.average([meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg], 
                                                weights=[args.proportions[0][0], args.proportions[0][1], args.proportions[1][0], args.proportions[1][1]]).item()
            
            worst_acc_groups = np.min([meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg]).item()

            print(f" =================>   Accs:   0_0 = {meters['0_0'].avg:.4f},   0_1 = {meters['0_1'].avg:.4f},   1_0 = {meters['1_0'].avg:.4f},   1_1 = {meters['1_1'].avg:.4f},   " +
                f"avg={avg_acc_groups:.4f},   weighted_avg={weighted_avg_acc_groups:.4f},   worst={worst_acc_groups:.4f}")

            if return_values_per_group:
                return loss_task_tot.avg , (top1.avg.item(), meters['0_0'].avg, meters['0_1'].avg, meters['1_0'].avg, meters['1_1'].avg, avg_acc_groups, weighted_avg_acc_groups, worst_acc_groups)
            



    else:       # SPECIFIC IMPLEMENTATION FOR MulticolorMNIST
        
        private_top1_left = AverageMeter('Acc@1', ':6.2f')
        private_top1_right = AverageMeter('Acc@1', ':6.2f')

        meters = {
            'aligned_aligned':         AverageMeter('Acc@1', ':6.2f'),
            'aligned_conflicting':     AverageMeter('Acc@1', ':6.2f'),
            'conflicting_aligned':     AverageMeter('Acc@1', ':6.2f'),
            'conflicting_conflicting': AverageMeter('Acc@1', ':6.2f')
            }

        for data, target, private_label in loader:

            private_label = private_label.squeeze().T
            private_label = private_label.to(args.device)
            private_label_left, private_label_right = private_label[0], private_label[1]

            data = data.to(args.device)
            target = target.to(args.device)

            private_label_left, private_label_right = private_label_left.to(args.device), private_label_right.to(args.device)
            
            output= model(data)

            output_private_left  = args.PH_left()
            output_private_right = args.PH_right()

            # -------------------- C.E. loss selection --------------------
            loss_task = computeCEloss(args, output, target, private_label)
            # -------------------------------------------------------------

            loss_task_tot.update(loss_task.item(), data.size(0))
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0], data.size(0))

            update_meters_MulticolorMNIST(output, target, private_label, meters)

            acc1_private_left = accuracy(output_private_left, private_label_left, topk=(1,))
            private_top1_left.update(acc1_private_left[0], data.size(0))

            acc1_private_right = accuracy(output_private_right, private_label_right, topk=(1,))
            private_top1_right.update(acc1_private_right[0], data.size(0))


        print(desc + f':  loss_task = {loss_task_tot.avg}, top1 = {top1.avg.item()}, top1_private_left = {private_top1_left.avg.item()}, top1_private_right = {private_top1_right.avg.item()}      (num_batches={len(loader)})')     

        avg_acc_groups = np.mean([meters['aligned_aligned'].avg, meters['aligned_conflicting'].avg, meters['conflicting_aligned'].avg, meters['conflicting_conflicting'].avg])

        weighted_avg_acc_groups = None

        worst_acc_groups = np.min([meters['aligned_aligned'].avg, meters['aligned_conflicting'].avg, meters['conflicting_aligned'].avg, meters['conflicting_conflicting'].avg]).item()

        print(f" =================>   Accs:   alig_alig = {meters['aligned_aligned'].avg:.4f},   alig_conf = {meters['aligned_conflicting'].avg:.4f},   conf_alig = {meters['conflicting_aligned'].avg:.4f},   conf_conf = {meters['conflicting_conflicting'].avg:.4f},   avg={avg_acc_groups:.4f},   worst={worst_acc_groups:.4f}")

        if return_values_per_group:
            return loss_task_tot.avg, (top1.avg.item(), meters['aligned_aligned'].avg, meters['aligned_conflicting'].avg, meters['conflicting_aligned'].avg, meters['conflicting_conflicting'].avg, avg_acc_groups, weighted_avg_acc_groups, worst_acc_groups)





    return loss_task_tot.avg , top1.avg.item()
