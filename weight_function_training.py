import os
n_threads = "12"
#
# os.environ["MKL_NUM_THREADS"] = n_threads
# os.environ["NUMEXPR_NUM_THREADS"] = n_threads
# os.environ["OMP_NUM_THREADS"] = n_threads

os.environ["OMP_NUM_THREADS"] = n_threads # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = n_threads # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = n_threads # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads# export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = n_threads # export NUMEXPR_NUM_THREADS=1



from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KernelDensity
import torch

import glob
from tqdm import tqdm
import sys
import numpy as np
import joblib
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader

import time

from train_NN_classifier import get_NN_classifier, initiate_classifier

sys.path.append("../vg_bench")

from model import network
from parser_new import parse_arguments
from pca_training import get_pca

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

    techniques = [model3, model4, model5, model6]
    #techniques = [model3, model4]
    print('Techniques loaded')

    return techniques

class DecisionDataset(data.Dataset):
  """
  Pytorch Dataset class that creates a dataset used to train weight functions
  
 
  """
  
  
  
    def __init__(self, data_path_list):
        self.img_path_list = []
        self.labels = []

        for i in range(len(data_path_list)):
            
            assert(os.path.exists(data_path_list[i]))

            print(data_path_list[i])
            if 'msls' in data_path_list[i]:
                count = 0
                filenames = self.get_subset(data_path_list[i], 2000)
                for filename in filenames:
                    # if (count % 10) == 0:
                    self.labels.append(1)
                    self.img_path_list.append(filename)
                    count+=1

            elif 'pitts30k' in data_path_list[i]:
                filenames = self.get_subset(data_path_list[i], 200)
                for filename in filenames:
                    self.labels.append(0)
                    self.img_path_list.append(filename)

            else:
                filenames = self.get_subset(data_path_list[i], 10)
                for filename in filenames:
                    # if 'pitts30k' in data_path_list[i]:
                    #     self.labels.append(0)
                    if 'eynsham'in data_path_list[i]:
                        self.labels.append(0)

                    else:
                        self.labels.append(0)

                    self.img_path_list.append(filename)
            assert(len(self.labels)==len(self.img_path_list))

    def __getitem__(self, index):
        img = base_transform(Image.open(self.img_path_list[index]).resize((640,360)).convert('RGB'))
        #print(np.asarray(img).shape)
        #img = torch.FloatTensor(img)#.to(device)
        # if img.dim() > 2:
        #     img = torch.permute(img, (2, 0, 1))
        # if img.dim() < 3:
        #     img = torch.unsqueeze(img, 0)
        # #img = torch.unsqueeze(img, 0)
        # #print(np.asarray(img).shape)
        # img = torch.permute(img, (1, 2, 0))
        # #print(np.asarray(img).shape)

        label = self.labels[index]
        return img, index, label

    def __len__(self):
        return len(self.img_path_list)

    def get_subset(self, data_path, subsampling_factor):

        filenames = sorted(glob.glob(os.path.join(data_path, '*.jpg')))

        zone_dict = {}


        for filename in filenames:
            zone = filename.split("@")[3]

            if zone in zone_dict:
                tmp_list = zone_dict[zone]
                tmp_list.append(filename)
                zone_dict[zone] = tmp_list

            else:
                zone_dict[zone] = [filename]

        zone_letter_dict = dict.fromkeys(zone_dict.keys())


        for key in zone_letter_dict:
            #print(key)
            zone_list = zone_dict[key]
            #print(len(zone_dict[key]))

            for filename in zone_list:
                zone_letter = filename.split("@")[4]
                if type(zone_letter_dict[key]) is dict:
                    if zone_letter in zone_letter_dict[key]:
                        tmp_list = zone_letter_dict[key][zone_letter]
                        tmp_list.append(filename)
                        zone_letter_dict[key][zone_letter] = tmp_list
                    else:
                        zone_letter_dict[key][zone_letter] = [filename]

                else:
                    zone_letter_dict[key] = {zone_letter : [filename]}

        final_filenames = []

        for i in zone_letter_dict.keys():

            for j in zone_letter_dict[i].keys():
                tmp_list = sorted(zone_letter_dict[i][j])
                final_filenames.extend(tmp_list)
                print('number of filenames for zone {} letter {}: {}'.format(i, j, len(zone_letter_dict[i][j])))

        #print(*final_filenames[0:20], sep='\n')


        subset = []

        for i in range(len(final_filenames)):
            if (i % subsampling_factor) == 0:
                subset.append(final_filenames[i])

        return subset




def get_weight_functions(data_path_list, tweakpara_list, decision_func_name, techniques, pca_list, feature_dims, dataset_names):
    """
    Function that loads/trains weight functions
    
    Input:
    data_path_list      list, contains the data paths of the training datasets
    tweakpara_list      list, contains hyperparameter values for which weight functions need to be obtained
    decision_func_name  string, type of weight function used
    techniques          list, contains the techniques of the ensemble
    pca_list            list, contains the pca models that can be used
    feature_dims        list, contains the descriptor dimensionality of all techniques
    dataset_names       list, names of the training datasets
    
    
    Output:
    decision_function_list    list, contains the weight functions
    mean_desc_list            list, contains the mean of the training desciptors in all feature spaces
    """
  
  
    decision_function_list = []
    training_needed = False
    mean_desc_list = []




    for i in range(len(techniques)):
        tech_decision_function_list = []

        if decision_func_name == 'KNN' or decision_func_name == 'NN_classifier':
            ### With discriminative methods, both training ref sets are used ###
            training_datasets = dataset_names
            features_list = [None, None]
            data_path_list_new = data_path_list

        if decision_func_name == 'KDE' or decision_func_name == 'GMM':
          ### With generative methods, only one training ref set is used ###

            features_list = [None]
            if (i % 2) == 0:
                ### for techs 0 and 2, pitts is used ###
                training_datasets = [dataset_names[0]]
                data_path_list_new = [data_path_list[0]]
                # if i < 2:
                #     pca = pca_list[0][i]
            if (i % 2) != 0:
                ### for techs 1 and 3, MSLS is used ###
                training_datasets = [dataset_names[1]]
                data_path_list_new = [data_path_list[1]]
                # if i < 2:
                #     pca = pca_list[1][i]






        for j in range(len(tweakpara_list)):
            #descriptor_path = 'descriptors/'
            decision_function_path = 'saved_models/{}/{}_tech_{}_tweakpara_{}.pkl'.format(decision_func_name, decision_func_name, i+2, tweakpara_list[j])
            decision_function_path_NN = 'saved_models/{}/{}_tech_{}_tweakpara_{}.pth'.format(decision_func_name, decision_func_name, i+2, tweakpara_list[j])
            #assert(os.path.exists(decision_function_path))
            
            if os.path.exists(decision_function_path):
                ### check if weight function can be loaded from disk ###
                print('loaded {} for tech {} and tweakpara {}'.format(decision_func_name, i+2, tweakpara_list[j]))
                tech_decision_function_list.append(joblib.load(decision_function_path))
            if os.path.exists(decision_function_path_NN):
                ### check if weight function can be loaded from disk, in the case of NN classifier ###
                print('loaded {} for tech {} and tweakpara {}'.format(decision_func_name, i+2, tweakpara_list[j]))
                model = initiate_classifier(feature_dims[i], 2 ,  tweakpara_list[j])
                model.load_state_dict(torch.load(decision_function_path_NN))
                model.eval()
                tech_decision_function_list.append(model)

            else:
                ### if weight function for tech i and hyperparameter j cannot be loaded, train it ###
                labels = []
                for k in range(len(training_datasets)):
                   
                    descriptor_path = 'descriptors/{}_TRAIN_descriptors_tech{}.npy'.format(training_datasets[k], i+2)
                    if isinstance(features_list[k], np.ndarray):
                        ### check if descriptors of training ref set k are still in memory ###
                        print('{} train descriptors (tech {}) still in memory'.format(training_datasets[k], i+2))
                        new_labels = [k]*(features_list[k].shape[0])
                        labels.extend(new_labels)

                    elif os.path.exists(descriptor_path):
                        ### Check if descriptors of training ref set k can be loaded from disk ###
                        features_list[k] = np.load(descriptor_path)
                        print('Loaded {} train descriptors (tech {}) from disk'.format(training_datasets[k], i+2))
                        new_labels = [k]*(features_list[k].shape[0])
                        #new_labels = [1]*(features_list[k].shape[0])
                        labels.extend(new_labels)



                    else:
                        ### Otherwise, generate training descriptors ###

                        print('test1')
                        #time.sleep(5)
                        dataset = DecisionDataset([data_path_list_new[k]])
                        data_loader = DataLoader(dataset, batch_size=1,
                                shuffle=True)
                        print('test2')
                        #time.sleep(5)
                        features = np.empty([len(dataset),feature_dims[i]])


                        print('test3')
                        #time.sleep(5)
                        for imgs, indices, label in tqdm(data_loader, ncols=100):
                            imgs = imgs.to(device)
                            desc = techniques[i](imgs).squeeze().cpu().detach().numpy()
                            # if i < 2:
                            #     desc = pca_list[(i % 2)].transform(desc.reshape(1,-1))
                            #     print('applying pca {} for tech {}'.format((i % 2), i))
                            #     if decision_func_name == 'KDE' or decision_func_name == 'GMM':
                            #         desc = pca.transform(desc.reshape(1,-1))
                            #     if decision_func_name == 'KNN' or decision_func_name == 'NN_classifier':
                            #         desc = pca_list[k][i].transform(desc.reshape(1,-1))

                            features[indices.numpy(), :] = desc
                            labels.append(k)


                        print('test4')
                        #time.sleep(5)
                        features_list[k] = (features)

                        np.save(descriptor_path, features)
                        del features
                        print('Generated {} train descriptors (tech {})'.format(training_datasets[k], i+2))




                if decision_func_name == 'KNN':
                    ### Concatenate features of both training reference sets ###
                    features_array = np.concatenate((features_list[0], features_list[1]), axis=0)
                    print(features_array.shape)
                    print(len(labels))
                    ### Use features and labels to train KNN ###
                    model = KNeighborsClassifier(n_neighbors=tweakpara_list[j], leaf_size=100).fit(features_array, labels)
                    #joblib.dump(model,  decision_function_path)
                    ### Append trained KNN model to list of weight functions for tech i ###
                    tech_decision_function_list.append(model)


                if decision_func_name == 'NN_classifier':
                    features_array = np.concatenate((features_list[0], features_list[1]), axis=0)
                    print(features_array.shape)
                    ### Train model using the get_NN_classifier function ###
                    model = get_NN_classifier(feature_dims[i], tweakpara_list[j], features_array, labels)
                    torch.save(model.state_dict(), decision_function_path_NN)
                    #print('NOT IMPLEMENTED YET')
                    tech_decision_function_list.append(model)

                if decision_func_name == 'KDE':
                    print(features_list[0].shape)
                    ### Train KDE model using the single corresponding training ref set ###
                    model = KernelDensity(kernel='gaussian', bandwidth=tweakpara_list[j], leaf_size=20).fit(features_list[0])
                    #joblib.dump(model,  decision_function_path)
                    tech_decision_function_list.append(model)
                    #print(model.score_samples(features_list[0]).shape)
                    mean_desc_list.append(np.mean(features_list[0], 0))

                if decision_func_name == 'GMM':
                    model = GaussianMixture(n_components=tweakpara_list[j]).fit(features_list[0])
                    #joblib.dump(model,  decision_function_path)
                    tech_decision_function_list.append(model)
                    mean_desc_list.append(np.mean(features_list[0], 0))
                    
            ### Combine the weight functions for each feature space into a list ###
            decision_function_list.append(tech_decision_function_list)


    return decision_function_list, mean_desc_list



# techniques = load_techniques()
#
# datasets = ['pitts30k', 'msls']
# data_path_list = []
# for i in range(len(datasets)):
#     data_path = 'datasets/{}/images/test/database/'.format(datasets[i])
#     if datasets[i] == 'msls' or datasets[i] == 'pitts30k':
#         data_path = 'datasets/{}/images/train/database/'.format(datasets[i])
#     data_path_list.append(os.path.join(os.getcwd(), data_path))
#
# tweakpara_list = [1]
# decision_func_name = 'NN_classifier'
# feature_dims_old = [16384, 16384, 256, 256, 512, 512]
# feature_dims_new = [1024, 1024, 256, 256, 512, 512]
# #feature_dims_new = [256, 256]#, 512, 512]
#
#
# pca_list = get_pca(1024, feature_dims_old, techniques, datasets)
#
# #pca_list = []
#
# decision_function_list = get_weight_functions(data_path_list, tweakpara_list, decision_func_name, techniques, pca_list, feature_dims_new, datasets)
