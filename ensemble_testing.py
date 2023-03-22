import numpy as np
import torch
from torch import nn
import os
from PIL import Image
import random
import glob
import math
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_generative import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KernelDensity
import joblib
from collections import Counter
from tqdm import tqdm
import torchvision.transforms as transforms

from torch.utils.data.dataset import Subset


import sys
sys.path.append("../vg_bench")
from parser_new import parse_arguments
from test_vg import test_vg
from model import network
from datasets_ws import BaseDataset
from util_vg import compute_pca
from pca_training import get_pca
from weight_function_training import get_weight_functions

from util_new import fuse_similarities
from util_new import get_output_shape
from util_new import norm_func


base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



args = parse_arguments()

if torch.cuda.is_available():
  dev = "cuda:0"
  args.device = "cuda:0"

  print('Using GPU')
else:
  dev = "cpu"

device = torch.device(dev)


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

    #techniques = [model1, model2, model3, model4, model5, model6]
    techniques = [model3, model4, model5, model6]
    print('Techniques loaded')

    return techniques





class WeightFunctionClass():
    """

    """
    def __init__(self, techniques, weights_functions, weight_function_type, generative,features_dim, mean_desc_list=None):

        self.weights_functions = weights_functions
        print(self.weights_functions)
        self.weight_function_type = weight_function_type
        self.techniques = techniques
        self.generative = generative
        self.mean_desc_list = mean_desc_list
        self.features_dim = features_dim

    def get_weights(self, features):
        pred_list = []

        weights_start_time = time.time()

        if generative ==False:


            for i in range(len(features)):
                print('Calculating weights for tech {}'.format(i))

                if self.weight_function_type == 'NN_classifier':
                    sigmoid = torch.nn.Sigmoid()
                    desc = torch.from_numpy(features[i]).to(device)

                    pred = sigmoid(self.weights_functions[i][0](desc))
                    #print(pred)
                    pred = pred.cpu().detach().numpy()
                    print(pred.shape)
                    pred_list.append(pred)
                    if i == 0:
                        #print(pred_list[i])
                        pitts_count = 0
                        msls_count = 0
                        for j in range(len(pred_list[i])):
                            if pred_list[i][j][0] > pred_list[i][j][1]:
                                pitts_count += 1
                            if pred_list[i][j][0] < pred_list[i][j][1]:
                                msls_count += 1
                        print('pitts_count: {}'.format(pitts_count))
                        print('msls_count: {}'.format(msls_count))


                else:
                    test = time.time()
                    pred = self.weights_functions[i][0].predict_proba(features[i])
                    # if i < 3:
                    print('weights in feature space of tech {}: {}'.format(i, pred))
                    print(pred.shape)
                    pred_list.append(pred)
                    print('predict proba time: {}'.format(time.time()-test))
                    # if i == 4:
                    #     #print(pred_list[i])
                    #     pitts_count = 0
                    #     msls_count = 0
                    #     for j in range(len(pred_list[i])):
                    #         if pred_list[i][j][0] > pred_list[i][j][1]:
                    #             pitts_count += 1
                    #         if pred_list[i][j][0] < pred_list[i][j][1]:
                    #             msls_count += 1
                    #     print('pitts_count: {}'.format(pitts_count))
                    #     print('msls_count: {}'.format(msls_count))





            scores = pred_list
            # for i in range(pred_list[0].shape[0]):
            #     avg_score = np.array([0.,0.])
            #     for j in range(len(features)):
            #         #avg_score += np.array([1.,1.])
            #         avg_score += pred_list[j][i]
            #
            #
            #     avg_score = avg_score/np.sum(avg_score)
            #     scores.append(avg_score)



        if generative == True:

            for i in range(len(features)):
                print('Calculating weights for tech {}'.format(i))
                #n_features = get_output_shape(techniques[i], (1, 3, 100, 100))[1]
                pred = self.weights_functions[i][0].score_samples(features[i])
                print(pred.shape)
                print(self.mean_desc_list[i].shape)
                max_lik = np.exp((self.weights_functions[i][0].score_samples(self.mean_desc_list[i].reshape(1, -1))[0]/self.features_dim[i]))

                pred1 = pred/self.features_dim[i]
                pred2 = norm_func(np.exp(pred1), 0, max_lik)

                pred_list.append(pred2)


            scores = pred_list

            print('Scores Example:')
            for i in range(len(features)):
                print(scores[i][0])



        print('TIME TO CALCULATE WEIGHTS: {}'.format(time.time()-weights_start_time))

        return scores




techniques = load_techniques()
training_dataset_names = ['pitts30k', 'msls']

pca_datasets = ['pitts30k', 'msls']#, args.dataset_name]

#pca_datasets = ['st_lucia', 'st_lucia', args.dataset_name]
#feature_dims = [16384, 16384, 256, 256, 512, 512]
#feature_dims_new = [1024, 1024, 256, 256, 512, 512]
feature_dims_new = [256, 256, 512, 512]


#pca_list = get_pca(1024, feature_dims, techniques, pca_datasets)
pca_list = []
#assert(len(pca_list)==2)
#print(pca_list[1])



weight_function_type = args.weight_function
#weight_function_type = 'KDE'
if weight_function_type == 'KNN' or weight_function_type == 'NN_classifier':
    generative = False

else:
    generative = True
ds_aware = args.ds_aware
#ds_aware = False
print('DS AWARE = {}'.format(ds_aware))
#tweakpara_list = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0]
#tweakpara_list = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
#bandwidth_list = [0.1, 0.2]
recall_dict = {}
#tweakpara_list = [5000]
#tweakpara_list = [1, 2, 3]#, 4, 5]

#tweakpara_list = [1, 3, 5, 10, 15, 25, 50, 100]
if args.weight_function == 'KDE':
    tweakpara_list = [float(i) for i in args.tweakpara_list.split(',')]
else:
    tweakpara_list = [int(i) for i in args.tweakpara_list.split(',')]

#tweakpara_list = [1, 10, 100, 1000, 5000, 10000]

#tweakpara_list = [0.5]

mean_desc_list = []
#

training_data_path_list = []
for i in range(len(training_dataset_names)):
    data_path = '../datasets_vg/datasets/{}/images/test/database/'.format(training_dataset_names[i])
    if training_dataset_names[i] == 'msls' or training_dataset_names[i] == 'pitts30k':
        data_path = '../datasets_vg/datasets/{}/images/train/database/'.format(training_dataset_names[i])
    training_data_path_list.append(os.path.join(os.getcwd(), data_path))




"""Testing Loop"""
import time

start_time = time.time()

for i in range(len(tweakpara_list)):

    recall_1 = []

    if ds_aware == True:
        weights_functions, mean_desc_list = get_weight_functions(training_data_path_list, [tweakpara_list[i]], weight_function_type, techniques, pca_list, feature_dims_new, training_dataset_names)
    if ds_aware == False:
        weights_functions = None
    weight_function_obj = WeightFunctionClass(techniques, weights_functions, weight_function_type, generative, feature_dims_new, mean_desc_list)


    full_features_dim = 64
    args.features_dim = 16384
    args.infer_batch_size = 1

    #pca = compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

    ######################################### DATASETS #########################################
    test_ds = BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    # random_indices = random.sample(range(0, len(test_ds)), 1000)
    # test_ds = Subset(test_ds, random_indices)

    print(f"Test set: {test_ds}")

    ######################################### TEST on TEST SET #########################################
    #args.test_method = 'single_query'
    test_start_time = time.time()
    print('NUM QUERIES: {}'.format(test_ds.queries_num))
    recalls, recalls_str, matching_score, result_array = test_vg(device, techniques, args, test_ds, weight_function_obj.get_weights ,  args.test_method, pca=None, ds_aware=ds_aware, sim_function=fuse_similarities, load_desc=True, n_queries=test_ds.queries_num, feature_pca=pca_list, generative=generative, fuse_type=args.fuse_type)
    print(f"Recalls on {test_ds}: {recalls_str}")
    print('Matching score of decision function {} on dataset {}: {}'.format(weight_function_type, args.dataset_name, matching_score))

    if ds_aware == True:
        #saving results array to disk
        np.save('result_arrays/{}{}_{}.npy'.format(args.weight_function, tweakpara_list[i], args.dataset_name), result_array)


    recall_1.append(recalls[0])
    recall_dict['tweakpara_{}'.format(tweakpara_list[i])] = '{}'.format(recall_1[0])

print('recall scores: {}'.format(recall_dict))
print('total elapsed time: {}'.format(time.time()-start_time))
print('actual test time: {}'.format(time.time()-test_start_time))
