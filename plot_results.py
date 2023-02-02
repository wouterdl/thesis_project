import numpy as np
import torch
from torch import nn
import os
from PIL import Image
import random
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from torch.utils.data.dataset import Subset

import sys
sys.path.append("../vg_bench")
from parser_vg import parse_arguments
from test_vg import test_vg
from model import network
from datasets_ws import BaseDataset
from util_vg import compute_pca

import torchvision.transforms as transforms
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

pitts_path = os.path.join(os.getcwd(), 'datasets/pitts30k/images/test/queries/')
msls_path = os.path.join(os.getcwd(), 'datasets/msls/images/val/queries/')

tokyo_path = os.path.join(os.getcwd(), 'datasets/tokyo247/images/test/database/')
stlucia_path = os.path.join(os.getcwd(), 'datasets/st_lucia/images/test/database/')
eynsham_path = os.path.join(os.getcwd(), 'datasets/eynsham/images/test/database/')
san_francisco_path = os.path.join(os.getcwd(), 'datasets/san_francisco/images/test/database/')


def load_images(img_paths, n_images):
    img_lengths = []
    img_list = []
    for value in img_paths:
        print('loading images from folder: {}'.format(value))
        count = 0

        tmp_list = []
        for filename in glob.glob(os.path.join(value, "*.jpg")):
            tmp_list.append(filename)

        random.shuffle(tmp_list)

        if len(tmp_list) > n_images:
            tmp_list = tmp_list[:n_images]

        for filename in tmp_list:
            im = np.asarray(Image.open(filename).resize((640,480)).convert('RGB'))
            print(im.shape)
            #im = im.flatten()
            #print(im.shape)
            im_tf = base_transform(im)#.permute(2, 0, 1)
            img_list.append(im_tf)
            count += 1

        img_lengths.append(count)
    print('DIMENSION TEST, TRAIN IMGS: {}'.format(img_list[0].shape))
    #for i in range(len(img_lengths)):
            #print('Loaded {} images of dataset {}'.format(img_lengths[i], img_paths[i]))
    return img_list, img_lengths

def apply_descriptor(descriptor, img_list, descriptor_arg):
    for i in range(len(img_list)):
        #print(type(img_list[i]))
        if descriptor_arg == 'resnet18' or descriptor_arg == 'resnet18_NV_pitts' or descriptor_arg == 'resnet18_trained' or descriptor_arg == 'resnet18_NV_msls':
            #print('TESTSETSTSTE')
            #print('image: {}'.format(img_list[i].shape))
            img = torch.FloatTensor(img_list[i]).to(device)
            if img.dim() > 2:
                #img = torch.permute(img, (2, 0, 1))
                print('DIMENSION TEST, TRAIN IMGS: {}'.format(img.size()))
            if img.dim() < 3:
                img = torch.unsqueeze(img, 0)
            img = torch.unsqueeze(img, 0)
            if descriptor_arg == 'resnet18_trained':
                sigmoid = torch.nn.Sigmoid()

                img_list[i] = sigmoid(descriptor(img)).squeeze().cpu().detach().numpy()
            else:
                img_list[i] = descriptor(img).squeeze().cpu().detach().numpy()#, channel_axis=-1)
        #print(img_list[i].shape)
        if descriptor_arg == 'hog':
            img_list[i] = descriptor(img_list[i], channel_axis=-1)

        else:
            img_list[i] = img_list[i].flatten()

    #img_list = np.array(img_list)
    print(type(img_list[0]))
    print('DIMENSION TEST, TRAIN DESCRIPTORS: {}'.format(img_list[0].shape))
    return img_list


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
    #techniques = [model1]
    #techniques = [model1, model2]
    print('Techniques loaded')

    return techniques

def load_training_datasets(technique_nr, datasets, n_images_train):

    #img_list, img_lengths = load_images(img_paths, n_images_train)

    #desc_list_train = apply_descriptor(technique, img_list, 'resnet18_NV_pitts')

    desc_list_train = []
    img_lengths = []

    for dataset in datasets:
        print('loading descriptors of {} train set'.format(dataset))
        desc_array = np.load('descriptors/{}_TRAIN_descriptors_tech{}.npy'.format(dataset, technique_nr))

        print(desc_array.shape)
        # if desc_array.shape[1] == 1024:
        #     print('reducing dimnesionality')
        #     desc_array = desc_array[:,0:512]
        #     assert(desc_array.shape[1] == 512)
        resize_factor = math.ceil(desc_array.shape[0]/n_images_train)
        before_len = len(desc_list_train)
        for i in range(desc_array.shape[0]):
            if (i % resize_factor) == 0:
                desc_list_train.append(desc_array[i])
        after_len = len(desc_list_train)

        img_lengths.append(after_len-before_len)

    print(len(desc_list_train))
    #assert(len(desc_list_train)==2*n_images_train)
    print('Max value train descriptor: {}'.format(np.amax(desc_list_train[0])))
    print('Min value train descriptor: {}'.format(np.amin(desc_list_train[0])))

    #img_lengths = int(len(desc_list_train)/2)

    return desc_list_train, img_lengths

def load_test_dataset(n_images_test, technique_nr, method):




    result_array = np.load('result_arrays/{}_{}.npy'.format(method, args.dataset_name))

    desc_array = np.load('descriptors/{}_QUERY_descriptors_tech{}.npy'.format(args.dataset_name, technique_nr))
    # if desc_array.shape[1] == 1024:
    #     print('reducing dimnesionality')
    #     desc_array = desc_array[:,0:512]
    #     assert(desc_array.shape[1] == 512)
    desc_list_test = list(desc_array)
    colors = []

    if n_images_test == 0:
        n_images_test = result_array.shape[0]

    for i in range(n_images_test):

        if result_array[i, 1] == 1:
            colors.append('g')
        if result_array[i, 1] == 0:
            colors.append('r')

    assert len(desc_list_test) == len(colors)

    indices = [*range(n_images_test)]

    return desc_list_test, colors, indices


def apply_tsne(desc_list_train, desc_list_test, n_images_train, method):
    sc = StandardScaler()
    pca = PCA(n_components=100)
    tsne = TSNE(perplexity=100)

    tsne_after_pca = Pipeline([
        ('std_scaler', sc),
        ('pca', pca),
        ('tsne', tsne)
    ])

    desc_list = []
    print('desc list length: {}'.format(len(desc_list)))
    print('desc list train length: {}'.format(len(desc_list_train)))
    print('desc list test length: {}'.format(len(desc_list_test)))
    desc_list_train.extend(desc_list_test)
    print('desc list train length: {}'.format(len(desc_list_train)))
    desc_list = desc_list_train
    print('desc list length: {}'.format(len(desc_list)))
    desc_list = np.array(desc_list)
    print('total desc list shape: {}'.format(desc_list.shape))


    tsne_data = tsne_after_pca.fit_transform(desc_list)

    print('TSNE SHAPE BEFORE: {}'.format(tsne_data.shape))
    #assert tsne_data.shape[0] == 2000
    #if method == 'KDE' or method == 'GMM':
    #
    # if 'KDE' in method or 'GMM' in method:
    #     tsne_data = np.concatenate((tsne_data[0:n_images_train], tsne_data[(2*n_images_train):]))
    #tsne_data = tsne_data[0:n_images_train]
    #assert tsne_data.shape[0] == 1000
    print('TSNE SHAPE AFTER: {}'.format(tsne_data.shape))

    return tsne_data

def create_plot(tsne_data_list, colors, n_images_train, indices, method, technames):
    from matplotlib.figure import figaspect
    from matplotlib.lines import Line2D
    fig_dir = os.path.join(os.getcwd(), 'plots/result_plots/{}'.format(args.dataset_name))
    pitts_color = ['b']*n_images_train[0]
    msls_color = ['m']*n_images_train[1]

    fig_weights, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))

    for i in range(len(technames)):
        tsne_data = tsne_data_list[i]

        fig = plt.figure()
        ax = fig.add_subplot(111)


        if 'KDE' in method or 'GMM' in method:
            fig_dir = os.path.join(os.getcwd(), 'plots/result_plots/{}/generative'.format(args.dataset_name))


            if (i % 2) == 0:
                train_img_no = n_images_train[0]
                new_colors = ['b']*train_img_no + colors
                train_colors = ['b']*train_img_no



            else:
                train_img_no = n_images_train[1]
                new_colors = ['m']*train_img_no + colors
                train_colors = ['m']*train_img_no

            tsne_data = np.concatenate((tsne_data[0:train_img_no], tsne_data[(2*train_img_no):]))

            #plot (incorrect) matches

            test_correct_handle = mlines.Line2D([], [], color='green', marker='.', linestyle='None',
                                  markersize=10, label='Correct matches')
            test_incorrect_handle = mlines.Line2D([], [], color='red', marker='.', linestyle='None',
                                  markersize=10, label='Incorrect matches')
            ax.scatter(tsne_data[:,0], tsne_data[:,1], c=new_colors, marker='o', linewidth=1, s=2)

            custom_dots = [ Line2D([0], [0], marker='o', color='w', label='Pitts30k', markerfacecolor='b'), Line2D([0], [0], marker='o', color='w', label='MSLS', markerfacecolor='m'),
                            Line2D([0], [0], marker='o', color='w', label='Correct matches', markerfacecolor='g'), Line2D([0], [0], marker='o', color='w', label='Incorrect matches', markerfacecolor='r')]

            ax.legend(handles=custom_dots)

            fig.savefig(os.path.join(fig_dir, 'matches_{}_{}.png'.format(method, i, 2)), dpi=2000)


            #create weights plot of current tech

            ax_curr = axes.flat[i]
             #+ ['m']*n_images_train


            ax_curr.scatter(tsne_data[:1*train_img_no,0], tsne_data[:1*train_img_no,1], c=train_colors, marker='o', linewidth=.5, s=.5)


            result_array = np.load('result_arrays/{}_{}.npy'.format(method, args.dataset_name))
            result_array = result_array[result_array[:, 0].argsort()]
            print('result array example: {}'.format(result_array[0]))

            gradient_values = []

            for j in range(len(indices)):
                gradient_values.append(result_array[indices[j], 2+i])

            plot = ax_curr.scatter(tsne_data[1*train_img_no:,0], tsne_data[1*train_img_no:,1], c=gradient_values, cmap='autumn_r', marker='o', linewidth=0.5, s=0.5, vmax=10.)#, vmin=0., vmax=1.)
            ax_curr.set_title(technames[i], fontsize=10)
            ax_curr.set_xticklabels([])
            ax_curr.set_yticklabels([])

        else:
            fig_dir = os.path.join(os.getcwd(), 'plots/result_plots/{}/discriminative'.format(args.dataset_name))
            train_colors = ['b']*n_images_train[0] + ['m']*n_images_train[1]
            new_colors = train_colors + colors

            total_train_img = n_images_train[0]+n_images_train[1]


            #Plot (in)correct matches
            ax.scatter(tsne_data[:,0], tsne_data[:,1], c=new_colors, marker='o', linewidth=1, s=2)

            custom_dots = [ Line2D([0], [0], marker='o', color='w', label='Pitts30k', markerfacecolor='b'), Line2D([0], [0], marker='o', color='w', label='MSLS', markerfacecolor='m'),
                            Line2D([0], [0], marker='o', color='w', label='Correct matches', markerfacecolor='g'), Line2D([0], [0], marker='o', color='w', label='Incorrect matches', markerfacecolor='r')]

            ax.legend(handles=custom_dots)

            fig.savefig(os.path.join(fig_dir, 'matches_{}_{}.png'.format(method, i, 2)), dpi=2000)

            #create weights plot of current tech
            #ax2 = fig.add_subplot(212)
            ax_curr = axes.flat[i]
             #+ ['m']*n_images_train


            ax_curr.scatter(tsne_data[:total_train_img,0], tsne_data[:total_train_img,1], c=train_colors, marker='o', linewidth=.5, s=.5)


            result_array = np.load('result_arrays/{}_{}.npy'.format(method, args.dataset_name))
            result_array = result_array[result_array[:, 0].argsort()]
            print('result array example: {}'.format(result_array[0]))

            gradient_values = []

            for j in range(len(indices)):
                gradient_values.append(result_array[indices[j], 2+i])

            plot = ax_curr.scatter(tsne_data[total_train_img:,0], tsne_data[total_train_img:,1], c=gradient_values, cmap='autumn_r', marker='o', linewidth=0.5, s=0.5)#, vmin=0., vmax=1.)
            ax_curr.set_title(technames[i], fontsize=10)
            ax_curr.set_xticklabels([])
            ax_curr.set_yticklabels([])
            #ax_curr.axis('off')







    #Create weights plot
    fig_weights.colorbar(plot, ax=axes.ravel().tolist())
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    #fig_weights.title('train and test descriptors of {} method'.format(method))
    fig_weights.savefig(os.path.join(fig_dir, 'weights_{}.png'.format(method, 2)), dpi=1000)


    #
    #     if 'KDE' in method or 'GMM' in method:
    #         new_colors = ['b']*n_images_train[i % 0] + colors
    #
    #     else:
    #         new_colors = ['b']*n_images_train[0] + ['m']*n_images_train[1] + colors
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #
    #     if (i % 2) == 0:
    #         train_handle = mlines.Line2D([], [], color='blue', marker='.', linestyle='None',
    #                           markersize=10, label='Pittsburgh')
    #
    #
    #         #tsne_data = np.concatenate((tsne_data[0:n_images_train], tsne_data[(2*n_images_train):]))
    #         train_colors = ['b']*n_images_train[0]
    #
    #     else:
    #         train_handle = mlines.Line2D([], [], color='blue', marker='.', linestyle='None', markersize=10, label='MSLS')
    #         #tsne_data = np.concatenate((tsne_data[n_images_train:2*n_images_train], tsne_data[(2*n_images_train):]))
    #         train_colors = ['m']*n_images_train[1]
    #     train_colors = ['b']*n_images_train[0] + ['m']*n_images_train[1]
    #     #msls_handle = mlines.Line2D([], [], color='m', marker='.', linestyle='None',
    #     #                      markersize=10, label='MSLS')
    #     test_correct_handle = mlines.Line2D([], [], color='green', marker='.', linestyle='None',
    #                           markersize=10, label='Correct matches')
    #     test_incorrect_handle = mlines.Line2D([], [], color='red', marker='.', linestyle='None',
    #                           markersize=10, label='Incorrect matches')
    #     ax.scatter(tsne_data[:,0], tsne_data[:,1], c=new_colors, marker='o', linewidth=1, s=2)
    #
    #     custom_dots = [ Line2D([0], [0], marker='o', color='w', label='Pitts30k', markerfacecolor='b'), Line2D([0], [0], marker='o', color='w', label='MSLS', markerfacecolor='m'),
    #                     Line2D([0], [0], marker='o', color='w', label='Correct matches', markerfacecolor='g'), Line2D([0], [0], marker='o', color='w', label='Incorrect matches', markerfacecolor='r')]
    #
    #     ax.legend(handles=custom_dots)
    #
    #     fig.savefig(os.path.join(fig_dir, 'matches_{}_{}.png'.format(method, i, 2)), dpi=2000)
    #
    #
    #
    #     #ax2 = fig.add_subplot(212)
    #     ax_curr = axes.flat[i]
    #      #+ ['m']*n_images_train
    #
    #
    #     ax_curr.scatter(tsne_data[:1*n_images_train,0], tsne_data[:1*n_images_train,1], c=train_colors, marker='o', linewidth=.5, s=.5)
    #
    #
    #     result_array = np.load('result_arrays/{}_{}.npy'.format(method, args.dataset_name))
    #     result_array = result_array[result_array[:, 0].argsort()]
    #     print('result array example: {}'.format(result_array[0]))
    #
    #     gradient_values = []
    #
    #     for j in range(len(indices)):
    #         gradient_values.append(result_array[indices[j], 2+i])
    #
    #     plot = ax_curr.scatter(tsne_data[1*n_images_train:,0], tsne_data[1*n_images_train:,1], c=gradient_values, cmap='autumn_r', marker='o', linewidth=0.5, s=0.5)#, vmin=0., vmax=1.)
    #     ax_curr.set_title(technames[i], fontsize=10)
    #     ax_curr.set_xticklabels([])
    #     ax_curr.set_yticklabels([])
    #     #ax_curr.axis('off')
    #
    #         #plot.gca().axes.get_yaxis().set_visible(False)
    #         #plot.gca().axes.get_xaxis().set_visible(False)
    #         #ax_curr.title('{}'.format(technames[i]))
    #         #plt.colorbar(plot2, ax=ax2)
    #         #fig.savefig(os.path.join(fig_dir, '{}_{}'.format(method, technr)), dpi=2000)
    #
    #
    # fig_weights.colorbar(plot, ax=axes.ravel().tolist())
    # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    #
    # #fig_weights.title('train and test descriptors of {} method'.format(method))
    # fig_weights.savefig(os.path.join(fig_dir, 'weights_{}.png'.format(method, 2)), dpi=1000)
    #
    # #else:




n_images_train = 1000
n_images_test = 0
img_paths_init = [pitts_path, msls_path]#, tokyo_path, stlucia_path, san_francisco_path]
#img_paths = [stlucia_path, san_francisco_path, pitts_path, msls_path]
#img_paths = [tokyo_path, san_francisco_path]

#methods = ['KNN_1', 'KNN_10', 'KNN_100', 'avg', 'dyn_mpf', 'NN_classifier_1']
#methods = ['KNN_1', 'KNN_100', 'NN_classifier_1', 'NN_classifier_3']
methods = ['KDE_0.01']

technames  = ['ResNet - GeM - Pittsburgh',
    'ResNet - GeM - MSLS', 'VGG16 - GeM - Pittsburgh', 'VGG16 - GeM - MSLS']
#methods = ['tech1', 'tech2']

techniques = load_techniques()

tsne_data_list = []
#img_paths = []
datasets_init = ['pitts30k', 'msls']

for i in range(len(methods)):
    for j in range(len(techniques)):
        img_paths = []
        datasets = []

        if methods[i] == 'GMM' or methods[i] == 'KDE':
            if (j % 2) == 0:
                #img_paths = [img_paths_init[0]]
                img_paths.append(img_paths_init[0])

                img_paths.append(img_paths_init[1])

                datasets.append(datasets_init[0])
                datasets.append(datasets_init[1])


            else:
                img_paths.append(img_paths_init[1])
                img_paths.append(img_paths_init[0])
                datasets.append(datasets_init[1])
                datasets.append(datasets_init[0])

        else:
            img_paths.append(img_paths_init[0])
            img_paths.append(img_paths_init[1])
            datasets.append(datasets_init[0])
            datasets.append(datasets_init[1])
            #img_paths = [img_paths_init[1]]
        desc_list_train, img_lengths = load_training_datasets(j+2, datasets, n_images_train)

        desc_list_test, colors, indices = load_test_dataset(n_images_test, j+2, methods[i])

        tsne_data = apply_tsne(desc_list_train, desc_list_test, img_lengths, methods[i])

        tsne_data_list.append(tsne_data)

    #print('tsne list length: {}'.format(len(tsne_data_list)))
    create_plot(tsne_data_list, colors, img_lengths, indices, methods[i], technames)
