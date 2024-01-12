import sys
import numpy as np
sys.path.append('../')
sys.path.append('./')

from Dual_Signal_Fusion_based_Map_Completion.options.train_options import TrainOptions
from Dual_Signal_Fusion_based_Map_Completion.utils.visualizer import Visualizer
from Dual_Signal_Fusion_based_Map_Completion.models import create_model
from Dual_Signal_Fusion_based_Map_Completion.data_loader import get_data_loader_multistage
from datetime import datetime
import os


class MultiTrainer:
    def __init__(self, opt, model, train_dl, val_dl, visualizer):
        self.opt = opt
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.visualizer = visualizer

    def fit(self):
        best_f1_score = 0.0
        # training phase
        tot_iters = 0
        early_stopping = EarlyStopper(patience=30, min_delta=0)

        for epoch in range(1, self.opt.n_epochs + 1):
            print(f'epoch {epoch}/{self.opt.n_epochs}')
            ep_time = datetime.now()

            train_tot_loss = 0
            train_tot_metrics = 0
            train_tot_metrics1 = 0
            train_tot_metrics2 = 0

            for i, data in enumerate(self.train_dl):
                self.model.train()
                self.model.set_input(data)
                iter_loss, iter_metrics, iter_metrics1, iter_metrics2 = self.model.optimize_parameters()
                iter_metrics = iter_metrics.numpy()

                train_tot_loss += iter_loss.item()
                train_tot_metrics += iter_metrics
                print("[Epoch %d/%d] [Batch %d/%d] [Loss: %f] [Precision: %f] [Recall: %f] [F1: %f] [CL IOU: %f]" % (epoch, opt.n_epochs, i, len(train_dl), iter_loss.item(), iter_metrics[0], iter_metrics[1], iter_metrics[2], iter_metrics[3]))
                tot_iters += 1

                # validating phase

                if tot_iters % opt.sample_interval == 0:
                    self.model.eval()
                    tot_loss = 0
                    tot_metrics = 0
                    tot_metrics1 = 0
                    tot_metrics2 = 0

                    for i, data in enumerate(self.val_dl):
                        self.model.set_input(data)
                        iter_loss, iter_metrics, iter_metrics1, iter_metrics2, loss1, loss2, loss3 = self.model.test()
                        tot_loss += iter_loss.item()
                        loss1 += loss1.item()
                        loss2 += loss2.item()
                        loss3 += loss3.item()
                        tot_metrics += iter_metrics.numpy()
                        tot_metrics1 += iter_metrics1.numpy()
                        tot_metrics2 += iter_metrics2.numpy()

                    tot_loss /= len(self.val_dl)
                    loss1 /= len(self.val_dl)
                    loss2 /= len(self.val_dl)
                    loss3 /= len(self.val_dl)
                    tot_metrics /= len(self.val_dl)
                    tot_metrics1 /= len(self.val_dl)
                    tot_metrics2 /= len(self.val_dl)

                    # visualize
                    model.compute_visuals()
                    save_result = tot_iters % self.opt.update_html_freq == 0
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                    val_losses = model.get_current_losses()
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, i / len(self.train_dl), val_losses)

                    if tot_metrics2[2] > best_f1_score:
                        best_f1_score = tot_metrics2[2]
                        print('best_f1_score: ', best_f1_score)
                        self.model.save_networks('latest')
                        self.model.save_networks(epoch)
                        with open(os.path.join(opt.checkpoints_dir, opt.name, 'results.txt'), 'a') as f:
                            f.write(
                                'epoch\t{}\titer\t{}\tloss_Dice\t{:.6f}\n'.format(epoch, tot_iters, tot_loss))
                            f.write(
                                'epoch\t{}\titer\t{}\tloss_Dice\t{:.6f}\tprecision\t{:.4f}\trecall\t{:.4f}\tf1\t{:.4f}\tcl_iou\t{:.4f}\n'.
                                format(epoch, tot_iters, loss1, tot_metrics[0], tot_metrics[1], tot_metrics[2], tot_metrics[3]))
                            f.write(
                                'epoch\t{}\titer\t{}\tloss_Dice\t{:.6f}\tprecision\t{:.4f}\trecall\t{:.4f}\tf1\t{:.4f}\tcl_iou\t{:.4f}\n'.
                                format(epoch, tot_iters, loss2, tot_metrics1[0], tot_metrics1[1], tot_metrics1[2], tot_metrics1[3]))
                            f.write(
                                'epoch\t{}\titer\t{}\tloss_Dice\t{:.6f}\tprecision\t{:.4f}\trecall\t{:.4f}\tf1\t{:.4f}\tcl_iou\t{:.4f}\n\n'.
                                format(epoch, tot_iters, loss3, tot_metrics2[0], tot_metrics2[1], tot_metrics2[2], tot_metrics2[3]))
                            f.close()

                    # early stop
                    if early_stopping.early_stop(tot_metrics2[2]):
                        print('early stop')
                        break

            train_tot_loss /= len(self.train_dl)
            train_tot_metrics /= len(self.train_dl)
            train_tot_metrics1 /= len(self.train_dl)
            train_tot_metrics2 /= len(self.train_dl)

            print('=================time cost: {}==================='.format(datetime.now() - ep_time))
            print("[Epoch %d/%d] [Loss: %f] [Precision: %f] [Recall: %f] [F1: %f] [CL IOU: %f]" % (epoch, opt.n_epochs, train_tot_loss, train_tot_metrics[0], train_tot_metrics[1], train_tot_metrics[2], train_tot_metrics[3]))

            self.model.update_learning_rate()


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_f1 = -np.inf
        print('early stop initial ')

    def early_stop(self, validation_f1):
        print('now validation_loss(f1) and min: ', validation_f1, self.max_validation_f1)
        if validation_f1 > self.max_validation_f1:
            self.max_validation_f1 = validation_f1
            self.counter = 0
            print('< early stop patience: ', self.counter)
        elif validation_f1 < (self.max_validation_f1 + self.min_delta):
            self.counter += 1
            print('> early stop patience: ', self.counter)
            if self.counter >= self.patience:
                return True
        return False


if __name__ == '__main__':
    opt = TrainOptions().parse()
    print(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.train_pattern == 'DSFNet':
        train_dl = get_data_loader_multistage(opt.dataroot, 'train')
        val_dl = get_data_loader_multistage(opt.dataroot, 'val')
        visualizer = Visualizer(opt)
        trainer = MultiTrainer(opt, model, train_dl, val_dl, visualizer)
        trainer.fit()

# python Dual_Signal_Fusion_based_Map_Completion/train.py --name test --dataroot D:/DataSet/multi_data_down/train_log_GKS_500/ --lam 0.2 --batch_size 8 --train_pattern DSFNet --net_trans DSFNet --model DSFNet
