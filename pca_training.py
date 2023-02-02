import os
# n_threads = "8"
# #
# # os.environ["MKL_NUM_THREADS"] = n_threads
# # os.environ["NUMEXPR_NUM_THREADS"] = n_threads
# # os.environ["OMP_NUM_THREADS"] = n_threads
#
# os.environ["OMP_NUM_THREADS"] = n_threads # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = n_threads # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = n_threads # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads# export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = n_threads # export NUMEXPR_NUM_THREADS=1
#






from sklearn.decomposition import PCA
import torch
import os
import glob
from tqdm import tqdm
import sys
import numpy as np
import joblib
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader



sys.path.append("../vg_bench")
from model import network
from parser_new import parse_arguments
from util_new import get_subset

args = parse_arguments()

if torch.cuda.is_available():
  dev = "cuda:0"
  args.device = "cuda:0"

  print('Using GPU')
else:
  dev = "cpu"

device = torch.device(dev)

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




def load_techniques():
    """Loads techniques into the ensemble"""
    print('loading techniques')
    model1 = network.GeoLocalizationNet(args).to(device)
    model2 = network.GeoLocalizationNet(args).to(device)
    args.aggregation = 'gem'
    model3 = network.GeoLocalizationNet(args).to(device)
    model4 = network.GeoLocalizationNet(args).to(device)
    args.backbone = 'vgg16'
    #args.aggregation = 'netvlad'
    model5 = network.GeoLocalizationNet(args).to(device)
    model6 = network.GeoLocalizationNet(args).to(device)

    model1.load_state_dict(torch.load('pre-trained_VPR_networks/pitts_resnet_netvlad_partial.pth'))
    model2.load_state_dict(torch.load('pre-trained_VPR_networks/msls_resnet_netvlad_partial.pth'))
    model3.load_state_dict(torch.load('pre-trained_VPR_networks/pitts_resnet_gem_partial.pth'))
    model4.load_state_dict(torch.load('pre-trained_VPR_networks/msls_resnet_gem_partial.pth'))
    model5.load_state_dict(torch.load('pre-trained_VPR_networks/pitts_vgg16_gem_partial.pth'))
    model6.load_state_dict(torch.load('pre-trained_VPR_networks/msls_vgg16_gem_partial.pth'))

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()

    techniques = [model1, model2, model3, model4, model5, model6]
    #techniques = [model1, model2]
    print('Techniques loaded')

    return techniques





class PCADataset_new(data.Dataset):
    def __init__(self, data_path):
        self.tmp_list = []

        assert(os.path.exists(data_path))
        if 'msls' in data_path:
            print('test')
            filenames = get_subset(data_path, 100)

        elif 'pitts' in data_path:
            filenames = get_subset(data_path, 10)
        else:
            filenames = get_subset(data_path, 5)

        for filename in filenames:


            self.tmp_list.append(filename)

    def __getitem__(self, index):
        img = base_transform(Image.open(self.tmp_list[index]).resize((640,360)).convert('RGB'))
        #print(np.asarray(img).shape)
        #img = torch.FloatTensor(img)#.to(device)
        if img.dim() > 2:
            img = torch.permute(img, (2, 0, 1))
        if img.dim() < 3:
            img = torch.unsqueeze(img, 0)
        #img = torch.unsqueeze(img, 0)
        #print(np.asarray(img).shape)
        img = torch.permute(img, (1, 2, 0))
        #print(np.asarray(img).shape)


        return img, index

    def __len__(self):
        return len(self.tmp_list)





def get_pca(pca_dim, feature_dims, techniques, datasets):
    pca_list = []


    for j in range(len(techniques)):
        dataset = datasets[(j % 2)]
        pca_path = 'PCA/pca_{}_tech{}.pkl'.format(dataset, j)
        if os.path.exists(pca_path):
            print('loaded PCA for tech {} and dataset {}'.format(j, dataset))
            pca_list.append(joblib.load(pca_path))

        elif feature_dims[j] > pca_dim:
            print('training PCA for tech {} and dataset {}'.format(j, dataset))

            pca = PCA(pca_dim)
            training_images = []


            tmp_labels = []
            tmp_list = []
            img_list = []
            print('test1')

            data_path = 'datasets/{}/images/test/database/'.format(dataset)

            if dataset == 'msls' or dataset == 'pitts30k':
                data_path = 'datasets/{}/images/train/database/'.format(dataset)


            print('test2')
            data_path = os.path.join(os.getcwd(), data_path)

            pca_dataset = PCADataset_new(data_path)

            data_loader = DataLoader(pca_dataset, batch_size=1,
                    shuffle=True)

            features = np.empty([len(pca_dataset),feature_dims[j]], dtype=np.float32)
            print(features.nbytes)
            for img, indices in tqdm(data_loader, ncols=100):
                #print('TEST2')
                img = img.to(device)
                desc = techniques[j](img).squeeze().cpu().detach().numpy()

                features[indices.numpy(), :] = desc


            #features = np.asarray(features)
            print(features.shape)
            pca.fit(features)
            pca_list.append(pca)

            joblib.dump(pca, pca_path)


    return pca_list

# feature_dims = [16384, 16384, 256, 256, 512, 512]
#
# #datasets = ['st_lucia', 'pitts30k', 'eynsham', 'san_francisco', 'tokyo247']
#
# datasets = ['msls', 'pitts30k']
#
# techniques = load_techniques()
#
# pca_list = get_pca(1024, feature_dims, techniques, datasets)
