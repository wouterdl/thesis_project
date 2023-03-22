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
from parser_new import parse_arguments
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

    print('Techniques loaded')

    return techniques

def load_training_datasets(technique_nr, datasets, n_images_train):



    desc_list_train = []
    img_lengths = []

    for dataset in datasets:
        print('loading descriptors of {} train set'.format(dataset))
        desc_array = np.load('descriptors/{}_TRAIN_descriptors_tech{}.npy'.format(dataset, technique_nr))

        print(desc_array.shape)

        resize_factor = math.ceil(desc_array.shape[0]/n_images_train)
        before_len = len(desc_list_train)
        for i in range(desc_array.shape[0]):
            if (i % resize_factor) == 0:
                desc_list_train.append(desc_array[i])
        after_len = len(desc_list_train)

        img_lengths.append(after_len-before_len)

    print(len(desc_list_train))

    print('Max value train descriptor: {}'.format(np.amax(desc_list_train[0])))
    print('Min value train descriptor: {}'.format(np.amin(desc_list_train[0])))

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
        #Append green color if correct, red if incorrect, based on result_array

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
            ### Generating plots for generative method ###

            fig_dir = os.path.join(os.getcwd(), 'plots/result_plots/{}/generative'.format(args.dataset_name))


            if (i % 2) == 0:
                # If technique is trained on pitts (tech0, tech2), get blue colors for training data
                train_img_no = n_images_train[0]
                new_colors = ['b']*train_img_no + colors
                train_colors = ['b']*train_img_no



            else:
                # If technique is trained on msls (tech1, tech3), get magenta colors for training data
                train_img_no = n_images_train[1]
                new_colors = ['m']*train_img_no + colors
                train_colors = ['m']*train_img_no

            # concatenate TSNE descriptors of correct training data and test data
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
            ### Generating plots for discriminative method ###


            fig_dir = os.path.join(os.getcwd(), 'plots/result_plots/{}/discriminative'.format(args.dataset_name))

            #append colors of pitts (blue), msls (magenta) and test data (green/red)
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

            ax_curr = axes.flat[i]

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



    #Create weights plot of all techs combined
    fig_weights.colorbar(plot, ax=axes.ravel().tolist())
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    #save to disk
    fig_weights.savefig(os.path.join(fig_dir, 'weights_{}.png'.format(method, 2)), dpi=1000)







def main():
    n_images_train = 1000
    n_images_test = 0
    img_paths_init = [pitts_path, msls_path]

    methods = [args.method_instance]

    technames  = ['ResNet - GeM - Pittsburgh',
        'ResNet - GeM - MSLS', 'VGG16 - GeM - Pittsburgh', 'VGG16 - GeM - MSLS']


    techniques = load_techniques()

    tsne_data_list = []

    datasets_init = ['pitts30k', 'msls']


    for i in range(len(methods)):
        for j in range(len(techniques)):
            img_paths = []
            datasets = []


            img_paths.append(img_paths_init[0])
            img_paths.append(img_paths_init[1])
            datasets.append(datasets_init[0])
            datasets.append(datasets_init[1])

            desc_list_train, img_lengths = load_training_datasets(j, datasets, n_images_train)

            desc_list_test, colors, indices = load_test_dataset(n_images_test, j, methods[i])

            tsne_data = apply_tsne(desc_list_train, desc_list_test, img_lengths, methods[i])

            tsne_data_list.append(tsne_data)

        create_plot(tsne_data_list, colors, img_lengths, indices, methods[i], technames)

if __name__ == "__main__":
    main()
