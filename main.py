import os
import cv2
import time
import shutil
import random
import datetime
import argparse
import numpy as np
import logging as logger
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import CosineSimilarity
from sklearn.metrics import f1_score, roc_auc_score

import warnings

warnings.filterwarnings("ignore")
from PIL import Image

from losses import MyInfoNCE
from models.hrnet import RGBDomain, NoiseDomain

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(message)s',
                   datefmt='%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='train', help='one of [train, val, test_single, flist]')
parser.add_argument('--input_size', type=int, default=1024, help='size of resized input')
parser.add_argument('--edge_width', type=int, default=3, help='width of edge mask')
parser.add_argument('--gt_ratio', type=int, default=4, help='resolution of input / output')
parser.add_argument('--train_bs', type=int, default=4, help='train batch size')
parser.add_argument('--test_bs', type=int, default=3, help='test batch size')
parser.add_argument('--save_res', type=int, default=1, help='whether to save the output')
parser.add_argument('--metric', type=str, default='cosine', help='metric for loss and clustering')
parser.add_argument('--out_dir', type=str, default=None, help='output dir')
parser.add_argument('--path_input', type=str, default='demo/input/', help='path of input')
parser.add_argument('--path_gt', type=str, default='demo/gt/', help='path of ground-truth (could be empty)')
parser.add_argument('--nickname', type=str, default='demo', help='short name for the dataset')
args = parser.parse_args()
logger.info(args)

date_now = datetime.datetime.now()
date_now = 'Log_v%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
args.out_dir = date_now


np.random.seed(666666)
torch.manual_seed(666666)
torch.cuda.manual_seed(666666)
torch.backends.cudnn.deterministic = True

class EdgeMaskGenerator(nn.Module):
    """generate the 'edge bar' for a 0-1 mask Groundtruth of a image
    Algorithm is based on 'Morphological Dilation and Difference Reduction'
    
    Which implemented with fixed-weight Convolution layer with weight matrix looks like a cross,
    for example, if kernel size is 3, the weight matrix is:
        [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]

    """
    def __init__(self, kernel_size = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    def _dilate(self, image, kernel_size=3):
        """Doings dilation on the image

        Args:
            image (_type_): 0-1 tensor in shape (B, C, H, W)
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert image.shape[2] > kernel_size and image.shape[3] > kernel_size, "Image must be larger than kernel size"
        
        kernel = torch.zeros((1, 1, kernel_size, kernel_size))
        kernel[0, 0, kernel_size // 2: kernel_size//2+1, :] = 1
        kernel[0, 0, :,  kernel_size // 2: kernel_size//2+1] = 1
        kernel = kernel.float()
        # print(kernel)
        res = F.conv2d(image, kernel.view([1,1,kernel_size, kernel_size]),stride=1, padding = kernel_size // 2)
        return (res > 0) * 1.0


    def _find_edge(self, image, kernel_size=3, return_all=False):
        """Find 0-1 edges of the image

        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        image = torch.tensor(image).float()
        shape = image.shape
        
        if len(shape) == 2:
            image = image.reshape([1, 1, shape[0], shape[1]])
        if len(shape) == 3:
            image = image.reshape([1, shape[0], shape[1], shape[2]])   
        assert image.shape[1] == 1, "Image must be single channel"
        
        dilation = self._dilate(image, kernel_size=kernel_size)
        
        erosion = self._dilate(1-image, kernel_size=kernel_size)

        diff1 = -torch.abs(dilation - image) + 1
        diff1 = 1.0 - (diff1 > 0) * 1.0
        # res = dilate(diff)
        diff1 = diff1.numpy()
        diff2 = -torch.abs(erosion - image) + 1
        diff2 = (diff2 > 0) * 1.0
        # res = dilate(diff)
        diff2 = diff2.numpy()

        diff3 = -torch.abs(erosion-dilation) + 1
        diff3 = (diff3>0) * 1.0
        diff3 = diff3.numpy()
        return diff1, diff2, diff3
    
    def forward(self, x, return_all=False):
        """
        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        return self._find_edge(x, self.kernel_size, return_all=return_all)



class MyDataset(Dataset):
    def __init__(self, num=0, file='', choice='train'):
        self.num = num
        self.choice = choice
        self.filelist = file

        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
        ])
        self.size = args.input_size
        self.albu = A.Compose([
            A.RandomScale(scale_limit=(-0.5, 0.0), p=0.75),
            A.PadIfNeeded(min_height=self.size, min_width=self.size, p=1.0),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
                A.Transpose(p=1),
            ], p=0.75),
            A.ImageCompression(quality_lower=50, quality_upper=95, p=0.75),
            A.OneOf([
                A.OneOf([
                    A.Blur(p=1),
                    A.GaussianBlur(p=1),
                    A.MedianBlur(p=1),
                    A.MotionBlur(p=1),
                ], p=1),
                A.OneOf([
                    A.Downscale(p=1),
                    A.GaussNoise(p=1),
                    A.ISONoise(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.RandomToneCurve(p=1),
                    A.Sharpen(p=1),
                ], p=1),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
                    A.GridDistortion(p=1),
                ], p=1),
            ], p=0.25),
        ])

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return self.num

    def load_item(self, idx):
        fname1, fname2 = self.filelist[idx]

        img = Image.open(fname1).convert('RGB')
        img = np.array(img)
        H, W, _ = img.shape
        # Warning: need to change the shape of mask from [H, W] ---> [H, W, 1]
        # otherwise, self.resize will ouput wrong mask 
        if fname2 == '':
            mask = np.zeros([H, W])
        else:
            mask = Image.open(fname2).convert('L')
        mask = np.expand_dims(np.array(mask), -1)

        mask[mask>0] = 255.

        if self.choice == 'train' and random.random() < 0.75:
            aug = self.albu(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
        im_no = img
        img = cv2.resize(img, (self.size, self.size))
        im_no = cv2.resize(im_no, (self.size//args.gt_ratio, self.size//args.gt_ratio) )
        mask = cv2.resize(mask, (self.size // args.gt_ratio, self.size // args.gt_ratio))
        mask = np.expand_dims(mask, -1)
        mask = thresholding(mask)

        img = img.astype('float') / 255.
        im_no = im_no.astype('float')
        mask = mask.astype('float') / 255.
        return self.transform(img), self.transform(im_no), self.tensor(mask[:, :, :1]), H, W, fname1.split('/')[-1], fname2

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class MVFEM(nn.Module):
    def __init__(self, net_list=['', '']):
        super().__init__()
        self.lr = 1e-4
        self.noiseprint = NoiseDomain(img_size=256).cuda()
        self.gen_edge = EdgeMaskGenerator(args.edge_width).cuda()

        net_weight, noise_weight = net_list
        cur_net = RGBDomain()
        self.rgb = nn.DataParallel(cur_net).cuda()
        if net_weight != '':
            self.load(net_weight, noise_weight)

        self.extractor_optimizer = optim.Adam(self.rgb.parameters(), lr=self.lr)
        self.noise_optimizer = optim.Adam(self.noiseprint.parameters(), lr=1e-3)
        self.save_dir = 'weights/' + args.out_dir
        if args.type == 'train':
            rm_and_make_dir(self.save_dir)
        self.myInfoNCE = MyInfoNCE(metric=args.metric)
        self.clustering = KMeans(verbose=False, n_clusters=2, distance=CosineSimilarity)
    

    def process(self, Ii, In, Mg, mask_path=None, isTrain=False):
        self.extractor_optimizer.zero_grad()
        self.noise_optimizer.zero_grad()
        if isTrain:
            Fo = self.rgb(Ii)
            Fo = Fo.permute(0, 2, 3, 1)
            B, H, W, C = Fo.shape

            No = self.noiseprint(In)
            No = No.permute(0, 2, 3, 1)

            Fo = torch.cat([Fo, No], dim=3)
            Fo = F.normalize(Fo, dim=3) # B * H * W * (32 + 21)
        else:
            with torch.no_grad():
                Fo = self.rgb(Ii)
                Fo = Fo.permute(0, 2, 3, 1)
                B, H, W, C = Fo.shape

                
                No = self.noiseprint(In)
                No = No.permute(0, 2, 3, 1)

                Fo = torch.cat([Fo, No], dim=3)
                Fo = F.normalize(Fo, dim=3)

        if isTrain:
            info_nce_loss = []
            Mg = Mg.cpu()
            pos_ed, neg_ed, edge = self.gen_edge(Mg)
            for idx in range(B):
                Fo_idx = Fo[idx]

                pos_idx = pos_ed[idx][0]
                neg_idx = neg_ed[idx][0]

                query = Fo_idx[pos_idx == 1]
                negative = Fo_idx[neg_idx == 1]
                if negative.size(0) == 0 or query.size(0) == 0:
                    logger.info("Eg: mask_path %s" % mask_path[idx])
                    continue
                dict_size = 1000
                query_sample = query[torch.randperm(query.size()[0])[:dict_size]]
                negative_sample = negative[torch.randperm(negative.size(0))[:dict_size]]
                info_nce_loss.append(self.myInfoNCE(query_sample, query_sample, negative_sample))

            edge_loss = torch.mean(torch.stack(info_nce_loss).squeeze())
            info_nce_loss = []
            for idx in range(B):
                Fo_idx = Fo[idx]
                Mg_idx = Mg[idx][0]
                pos_idx = Mg_idx + edge[idx][0]
                neg_idx = pos_idx + pos_ed[idx][0]

                query = Fo_idx[pos_idx == 0]
                negative = Fo_idx[neg_idx == 1]
                if negative.size(0) == 0:
                    negative = Fo_idx[Mg_idx==1]
                if query.size(0) == 0:
                    query = Fo_idx[Mg_idx==0]
               
                if negative.size(0) == 0 or query.size(0) == 0:
                    logger.info("Fu3: mask_path %s" % mask_path[idx])
                    continue
                dict_size = 1000
                query_sample = query[torch.randperm(query.size()[0])[:dict_size]]
                negative_sample = negative[torch.randperm(negative.size(0))[:dict_size]]
                info_nce_loss.append(self.myInfoNCE(query_sample, query_sample, negative_sample))

            full_loss = torch.mean(torch.stack(info_nce_loss).squeeze())
            self.backward(edge_loss+full_loss)
            return full_loss, edge_loss
            
        else:
            with torch.no_grad():
                Mo = None
                Mo_reverse=None
                Fo = torch.flatten(Fo, start_dim=1, end_dim=2)
                result = self.clustering(x=Fo, k=2)
                Lo_batch = result.labels
                for idx in range(B):
                    Lo = Lo_batch[idx]
                    Lo_reverse = 1 - Lo
                    if torch.sum(Lo) > torch.sum(1 - Lo):
                        Lo = 1 - Lo
                        Lo_reverse = 1 - Lo
                    Lo = Lo.view(H, W)[None, :, :, None]
                    Lo_reverse = Lo_reverse.view(H, W)[None, :, :, None]
                    Mo = torch.cat([Mo, Lo], dim=0) if Mo is not None else Lo
                    Mo_reverse = torch.cat([Mo_reverse, Lo_reverse], dim=0) if Mo_reverse is not None else Lo_reverse
                Mo = Mo.permute(0, 3, 1, 2)
                Mo_reverse = Mo_reverse.permute(0, 3, 1, 2)
                return Mo, Mo_reverse
                    

    def backward(self, batch_loss=None):
        if batch_loss:
            batch_loss.backward()
            self.extractor_optimizer.step()
            self.noise_optimizer.step()

    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.rgb.state_dict(),
                   self.save_dir + path + '%s_weights.pth' % self.rgb.module.name)
        if self.noiseprint:
            torch.save(self.noiseprint.state_dict(),
                    self.save_dir + path + '%s_weights.pth' % self.noiseprint.name)


    def load(self, path='', noise_path=''):
        weights_file = torch.load('weights/' + path)
        self.rgb.load_state_dict(weights_file)
        logger.info('Loaded [%s] from [%s]' % (self.rgb.module.name, path))
        if self.noiseprint and noise_path:
            weights = torch.load('weights/' + noise_path)
            self.noiseprint.load_state_dict(weights)
            logger.info('Loaded [%s] from [%s]' % (self.noiseprint.name, noise_path))
        


class ForgeryForensics():
    def __init__(self):
        self.train_npy_list = [
            # name, repeat_time
            ('tampCOCO_sp_train_199901.npy', 1),
            ('tampCOCO_cm_train_199347.npy', 1),
            ('tampCOCO_bcm_train_199357.npy', 1),
            ('CASIA_train_5123.npy', 40),
            ('IMD20_train_2010.npy', 20),
        ]
        self.train_file = None
        for item in self.train_npy_list:
            self.train_file_tmp = np.load('flist/' + item[0])
            for _ in range(item[1]):
                self.train_file = np.concatenate(
                    [self.train_file, self.train_file_tmp]) if self.train_file is not None else self.train_file_tmp

        self.train_num = len(self.train_file)
        train_dataset = MyDataset(num=self.train_num, file=self.train_file, choice='train')

        self.val_npy_list = [
            # name, nickname
            ('NC16_564.npy', 'NIST'),
            ('Columbia_180.npy', 'Columbia'),
            ('COVERAGE_100.npy', 'COVERAGE'),
            ('CASIA_920.npy', 'CASIA'),
        ]
        self.val_file_list = []
        for item in self.val_npy_list:
            self.val_file_tmp = np.load('flist/' + item[0])
            self.val_file_list.append(self.val_file_tmp)

        self.train_bs = args.train_bs
        self.test_bs = args.test_bs
        self.model = MVFEM([
            # ('', ''),  # Training from scratch
        ]).cuda()
        self.n_epochs = 99999
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_bs, num_workers=self.train_bs,
                                       shuffle=True)
        logger.info('Train on %d images.' % self.train_num)
        for idx, file_list in enumerate(self.val_file_list):
            logger.info('Test on %s (#%d).' % (self.val_npy_list[idx][0], len(file_list)))

    def train(self):
        cnt, batch_losses, edge_losses = 0, [], []
        best_score = 0
        scheduler1 = ReduceLROnPlateau(self.model.extractor_optimizer, mode='max', factor=0.9, patience=3, min_lr=1e-8)
        scheduler2 = ReduceLROnPlateau(self.model.noise_optimizer, mode='max', factor=0.6, patience=3, min_lr=1e-8)
        self.model.train()
        for epoch in range(1, self.n_epochs + 1):
            start_time = time.perf_counter()
            for items in self.train_loader:
                cnt += self.train_bs
                Ii, In, Mg = (item.cuda() for item in items[:3])  # Input, Ground-truth Mask
                mask_path = items[-1]
                batch_loss, edge_loss = self.model.process(Ii, In, Mg, mask_path, isTrain=True)
                

                batch_losses.append(batch_loss.item())
                edge_losses.append(edge_loss.item())
                if cnt % (self.train_bs * 20) == 0:
                    logger.info('Tra (%6d/%6d): G: %5.4f Eg: %5.4f lr1: %5.8f lr2: %5.8f' % 
                                (cnt, self.train_num, np.mean(batch_losses), np.mean(edge_losses), 
                                scheduler1.get_last_lr()[0], scheduler2.get_last_lr()[0]))
                if cnt % int((self.train_loader.dataset.__len__() / 80) // self.train_bs * self.train_bs) == 0:
                    logger.info('Ep%03d(%6d/%6d): Tra: G: %5.4f Eg: %5.4f lr1: %5.8f lr2: %5.8f time: %s' % 
                                (epoch, cnt, self.train_num, np.mean(batch_losses), np.mean(edge_losses),
                                scheduler1.get_last_lr()[0], scheduler2.get_last_lr()[0], 
                                format_time(time.perf_counter()-start_time)))
                    start_time = time.perf_counter()
                    tmp_score = self.val()
                    scheduler1.step(tmp_score)
                    scheduler2.step(tmp_score)
                    self.model.save('Ep%03dCnt%06d_%5.4f/'% (epoch, cnt, tmp_score))
                    if tmp_score > best_score:
                        best_score = tmp_score
                        logger.info('Score: %5.4f (Best) time: %s' % (best_score, format_time(time.perf_counter() - start_time)))
                        self.model.save('Ep%03d_%5.4f/' % (epoch, tmp_score))
                    else:
                        logger.info('Score: %5.4f time: %s' % (tmp_score, format_time(time.perf_counter() - start_time)))
                    self.model.train()
                    start_time = time.perf_counter()
                    batch_losses = []
                    edge_losses = []
            cnt = 0

    def val(self):
        tmp_score = []
        for idx in range(len(self.val_file_list)):
            P_F1, P_IOU = ForensicTesting(self.model, bs=self.test_bs, test_npy=self.val_npy_list[idx][0],
                                        test_file=self.val_file_list[idx])
            tmp_score.append(P_F1)
            tmp_score.append(P_IOU)
            logger.info('%s(#%d): PF1:%5.4f, PIOU:%5.4f' % (
            self.val_npy_list[idx][1], len(self.val_file_list[idx]), P_F1, P_IOU))
        if args.type == 'val':
            logger.info('Score: %5.4f' % np.mean(tmp_score))
        return np.mean(tmp_score)
        



def ForensicTesting(model, bs=1, test_npy='', test_file=None):
    if test_file is None:
        test_file = np.load('flist/' + test_npy)
    test_num = len(test_file)
    test_dataset = MyDataset(test_num, test_file, choice='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=min(48, bs), shuffle=False)
    model.eval()
    f1, iou = [], []
    if args.save_res == 1:
        path_out = './demo/output/'
        rm_and_make_dir(path_out)

    for items in test_loader:
        Ii, In, Mg, Hg, Wg = (item.cuda() for item in items[:-2])
        filename = items[-2]

        Mo, Mo_reverse = model.process(Ii, In, None, None)

        Mg, Mo, Mo_reverse = convert(Mg), convert(Mo), convert(Mo_reverse)

        if args.save_res == 1:
            Hg, Wg = Hg.cpu().numpy(), Wg.cpu().numpy()
            for i in range(Ii.shape[0]):
                res = cv2.resize(Mo[i], (Wg[i].item(), Hg[i].item()))
                res = thresholding(res)
                res = Image.fromarray(res.astype(np.uint8))
                res.save(path_out + filename[i][:-4] + '.png')

        for i in range(Mo.shape[0]):
            Mo_resized = thresholding(cv2.resize(Mo[i], (Mg[i].shape[:2][::-1])))[..., None]
            Mo_resized_reverse = thresholding(cv2.resize(Mo_reverse[i], (Mg[i].shape[:2][::-1])))[..., None]
            f1.append(max([f1_score(Mg[i].flatten(), Mo_resized.flatten(), average='macro'), 
                                f1_score(Mg[i].flatten(), Mo_resized_reverse.flatten(), average='macro')
                                ]))
            iou.append(max([metric_iou(Mo_resized / 255., Mg[i] / 255.),
                           metric_iou(Mo_resized_reverse / 255., Mg[i] / 255.)
                            ]))
    Pixel_F1 = np.mean(f1)
    Pixel_IOU = np.mean(iou)
    if args.type == 'test_single':
        logger.info('%s Score: F1: %5.4f, IoU: %5.4f' % (Pixel_F1, Pixel_IOU))
    return Pixel_F1, Pixel_IOU


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def convert(x):
    x = x * 255.
    return x.permute(0, 2, 3, 1).cpu().detach().numpy()


def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x


def generate_flist(path_input, path_gt, nickname):
    # NOTE: The image and ground-truth should have the same name.
    # Example:
    # path_input = 'tampCOCO/sp_images/'
    # path_gt = 'tampCOCO/sp_masks/'
    # nickname = 'tampCOCO_sp'
    res = []
    flag = False
    flist = sorted(os.listdir(path_input))
    for file in flist:
        name = file.rsplit('.', 1)[0]
        path_mask = path_gt + name + '.png'
        # path_mask = path_gt + name + '.tif'
        # path_mask = path_gt + name + '_gt.png'
        if not os.path.exists(path_mask):
            path_mask = ''
            flag = True
        res.append((path_input + file, path_mask))
    save_name = '%s_%s.npy' % (nickname, len(res))
    np.save('flist/' + save_name, np.array(res))
    if flag:
        logger.info('Note: The following score is meaningless since no ground-truth is provided.')
    return save_name


def metric_iou(prediction, groundtruth):
    intersection = np.logical_and(prediction, groundtruth)
    union = np.logical_or(prediction, groundtruth)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    if np.sum(intersection) + np.sum(union) == 0:
        iou = 1
    return iou

def format_time(seconds):
    days, rem = divmod(int(seconds), 86400)  
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"  


if __name__ == '__main__':
    if args.type == 'train':
        model = ForgeryForensics()
        model.train()
    elif args.type == 'val':
        model = ForgeryForensics()
        model.val()
    elif args.type == 'test_single':
        model = MVFEM(net_list=[
            'Multi_View_RGB_weights.pth', 'MIX_noiseprint_weights.pth'
        ]).cuda()
        file_npy = generate_flist(args.path_input, args.path_gt, args.nickname)
        ForensicTesting(model, test_npy=file_npy)
    elif args.type == 'flist':
        generate_flist(args.path_input, args.path_gt, args.nickname)
