import sys
sys.path.append('../')
sys.path.append('./')
from Dual_Signal_Fusion_based_Map_Completion.options.test_options import TestOptions
from Dual_Signal_Fusion_based_Map_Completion.models import create_model
from Dual_Signal_Fusion_based_Map_Completion.data_loader import get_data_loader_multistage
from Dual_Signal_Fusion_based_Map_Completion.utils.visualizer import save_images
from Dual_Signal_Fusion_based_Map_Completion.utils import html
import os


class Tester:
    def __init__(self, opt, model, test_dl):
        self.opt = opt
        self.model = model
        self.test_dl = test_dl

    def pred(self):
        self.model.eval()
        # create a website
        web_dir = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.epoch))  # define the website directory
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        tot_loss = 0
        tot_metrics = 0
        for i, data in enumerate(self.test_dl):
            self.model.set_input_test(data)
            iter_loss, _, iter_metrics, iter_road_metrics = self.model.testtest()
            tot_loss += iter_loss.item()
            tot_metrics += iter_metrics.numpy()
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            print('img_path: ', img_path)
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        tot_loss /= len(self.test_dl)
        tot_metrics /= len(self.test_dl)
        print('loss_Dice\t{:.6f}\tprecision\t{:.4f}\trecall\t{:.4f}\tf1\t{:.4f}\tiou\t{:.4f}\n'.format(tot_loss, tot_metrics[0], tot_metrics[1], tot_metrics[2], tot_metrics[3]))
        webpage.save()  # save the HTML


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    test_dl = get_data_loader_multistage(opt.dataroot, 'test')
    tester = Tester(opt, model, test_dl)
    tester.pred()
