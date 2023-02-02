import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import os
import glob
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from random import randrange
import random


import sys
sys.path.append("../vg_bench")
from model import network
from parser_new import parse_arguments

args = parse_arguments()

if torch.cuda.is_available():
  dev = "cuda:0"
  args.device = "cuda:0"

  print('Using GPU')
else:
  dev = "cpu"

device = torch.device(dev)

# class DescriptorDataset(data.Dataset):
#     def __init__(self, datasets, n_images, technique, n_features):
#         super().__init__()
#
#         self.descriptors = []
#         self.labels = []
#
#         #self.techniques, self.n_features_list = self.load_techniques()
#         self.technique = technique
#         self.n_features = n_features
#
#
#         tmp_descriptors, tmp_labels = self.get_descriptors(self.technique, datasets, n_images)
#         self.descriptors.extend(tmp_descriptors)
#         self.labels.extend(tmp_labels)
#         del tmp_descriptors, tmp_labels
#
#         self.descriptors = np.asarray(self.descriptors)
#
#
#     def __getitem__(self, index):
#         sample = self.descriptors[index]
#
#         label = self.labels[index]
#
#         return sample, label
#
#     def __len__(self):
#         return len(self.labels)
#
#
#
#     def get_descriptors(self, technique, datasets, n_images):
#
#         print('getting descriptors of train datasets')
#         img_list = []
#         descriptors = []
#         labels = []
#         print(datasets)
#
#         for dataset in datasets:
#
#             tmp_labels = []
#             tmp_list = []
#             data_path = 'datasets/{}/images/train/database/'.format(dataset)
#
#             data_path = os.path.join(os.getcwd(), data_path)
#
#             for filename in glob.glob(os.path.join(data_path, '*.jpg')):
#
#                 tmp_list.append(filename)
#                 if dataset == 'pitts30k':
#                     tmp_labels.append(0)
#
#                 if dataset == 'msls':
#                     tmp_labels.append(1)
#
#             ziplist = list(zip(tmp_list, tmp_labels))
#             random.shuffle(ziplist)
#             tmp_list, tmp_labels = zip(*ziplist)
#             if len(tmp_list) > n_images:
#                 tmp_list = tmp_list[:n_images]
#             if len(tmp_labels) > n_images:
#                 tmp_labels = tmp_labels[:n_images]
#
#
#             img_list.extend(tmp_list)
#             labels.extend(tmp_labels)
#
#         for img_path in img_list:
#             #print('TEST2')
#             img = np.asarray(Image.open(img_path).resize((640,360)).convert('RGB'))
#             img = torch.FloatTensor(img).to(device)
#             if img.dim() > 2:
#                 img = torch.permute(img, (2, 0, 1))
#             if img.dim() < 3:
#                 img = torch.unsqueeze(img, 0)
#             img = torch.unsqueeze(img, 0)
#             desc = technique(img).squeeze().cpu().detach().numpy()
#
#             descriptors.append(desc)
#
#         print('descriptor shape: {}'.format(np.asarray(descriptors).shape))
#
#
#         #descriptors = pca.fit_transform(np.asarray(descriptors))
#         #descriptors = np.asarray(descriptors)
#         #print('descriptors length: {}'.format(len(descriptors)))
#         #print('descriptor shape: {}'.format(descriptors.shape))
#
#
#
#         return descriptors, labels

#
# def load_techniques():
#     def get_output_shape(model, image_dim):
#         return model(torch.rand(*(image_dim)).to(device)).data.shape
#
#
#
#     print('loading techniques')
#     model1 = network.GeoLocalizationNet(args).to(device)
#     model2 = network.GeoLocalizationNet(args).to(device)
#     args.aggregation = 'gem'
#     model3 = network.GeoLocalizationNet(args).to(device)
#     model4 = network.GeoLocalizationNet(args).to(device)
#     args.backbone = 'vgg16'
#     #args.aggregation = 'netvlad'
#     model5 = network.GeoLocalizationNet(args).to(device)
#     model6 = network.GeoLocalizationNet(args).to(device)
#
#     model1.load_state_dict(torch.load('pre-trained_VPR_networks/pitts_resnet_netvlad_partial.pth'))
#     model2.load_state_dict(torch.load('pre-trained_VPR_networks/msls_resnet_netvlad_partial.pth'))
#     model3.load_state_dict(torch.load('pre-trained_VPR_networks/pitts_resnet_gem_partial.pth'))
#     model4.load_state_dict(torch.load('pre-trained_VPR_networks/msls_resnet_gem_partial.pth'))
#     model5.load_state_dict(torch.load('pre-trained_VPR_networks/pitts_vgg16_gem_partial.pth'))
#     model6.load_state_dict(torch.load('pre-trained_VPR_networks/msls_vgg16_gem_partial.pth'))
#
#     model1.eval()
#     model2.eval()
#     model3.eval()
#     model4.eval()
#     model5.eval()
#     model6.eval()
#
#     techniques = [model1, model2, model3, model4, model5, model6]
#     print('Techniques loaded')
#
#     n_features_list = []
#
#
#     for tech in techniques:
#         dim =  get_output_shape(tech, (1, 3, 100, 100))[1]
#         n_features_list.append(dim)
#
#     return techniques, n_features_list



#
# def initiate_classifier(n_techs):
#
#     model = models.resnet18(pretrained=True)
#     model.conv1 = torch.nn.Conv2d(1, 64, (7,7), (2,2), (3,3))
#     num_ftrs = model.fc.in_features
#     model.fc = torch.nn.Linear(num_ftrs, n_techs)
#     model.layer1 = torch.nn.Identity()
#     model.layer2[1] = torch.nn.Identity()
#     model.layer3[1] = torch.nn.Identity()
#     model.layer4[1] = torch.nn.Identity()
#     #model.layer4[0].conv1 = torch.nn.Conv2d(128, 512, (3,3), (2,2), (1,1))
#
#     #model = torch.nn.Sequential(model, torch.nn.Softmax(2))
#     model = model.to(device)
#
#     return model


def initiate_classifier(n_features, output_size, n_layers):

    if n_layers == 1:
        layer1 = torch.nn.Linear(n_features, output_size)
        model = layer1

    if n_layers == 2:

        layer1 = torch.nn.Linear(n_features, int(max(n_features/4, output_size*4)))
        layer2 = torch.nn.Linear(int(max(n_features/4, output_size*4)), output_size)
        model = torch.nn.Sequential(layer1, layer2)

    if n_layers == 3:
        layer1 = torch.nn.Linear(n_features, int(max(n_features/2, output_size*8)))
        layer2 = torch.nn.Linear(int(max(n_features/2, output_size*8)), int(max(n_features/4, output_size*4)))
        layer3 = torch.nn.Linear(int(max(n_features/4, output_size*4)),output_size)
        model = torch.nn.Sequential(layer1, layer2, layer3)

    model = model.to(device)

    return model


# def initiate_classifier(tmp):
#     relu = torch.nn.ReLU()
#     conv1 = torch.nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
#     bn1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#
#     conv2 = torch.nn.Conv2d(64, 128, (3,3), (1,1), (1,1), bias=False)
#     bn2 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     conv3 = torch.nn.Conv2d(128, 128, (3,3), (1,1), (1,1), bias=False)
#     bn3 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#
#     avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
#     fc = torch.nn.Linear(128, 2, bias=True)
#
#     model = torch.nn.Sequential(conv1, bn1, relu, maxpool, conv2,  bn2, relu, conv3, bn3, avgpool, fc)
#     model = model.to(device)
#
#     return model

#
# def initiate_classifier(n_classes):
#
#     layer1 = torch.nn.Conv1d(1, 32, 3)
#     layer2 = torch.nn.Conv1d(32, 64, 3)
#     #layer3 = torch.nn.Conv1d(64, 64, 3)
#     #layer4 = torch.nn.Conv1d(64, 128, 3)
#     fc = torch.nn.Linear(63, n_classes)
#     model = torch.nn.Sequential(layer1, torch.nn.BatchNorm1d(32),  torch.nn.ReLU(), layer2, torch.nn.BatchNorm1d(64), torch.nn.ReLU(), fc, torch.nn.Softmax(n_classes))
#     model = model.to(device)
#
#     return model



def train_classifier(model, criterion, optimizer, num_epochs, train_features, train_labels, test_features, test_labels, batch_size, n_layers):
    print(torch.LongTensor(test_labels[0]))
    print(torch.tensor(test_labels[0]))
    test = torch.tensor(test_labels[0]).type(torch.LongTensor)
    test_array = np.array([1., 1.])
    #test.type(torch.LongTensor)
    print(test)
    print(test.type())
    print(test_array)
    best_acc = 0.0

    print('model test before training: {}'.format(model(torch.from_numpy(train_features[0]).to(device).float().unsqueeze(0))))
    print(train_labels[0])

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        #print(test_labels[0])
        for i in tqdm(range(len(train_labels))):

            model.train()
            optimizer.zero_grad()
            model = model.float()
            #model=model.double()

            features = torch.from_numpy(train_features[i]).to(device).float().unsqueeze(0)
            labels = torch.LongTensor([train_labels[i]])
            #print(labels)
            #labels = torch.Longtensor(train_labels[i])#.type(torch.LongTensor)
            #.type(torch.LongTensor)
            labels = labels.to(device)#.unsqueeze(1)
            #print(features.size())
            #print(type(labels))
            #print(labels.size())

            outputs = model(features)
            #print(outputs)
            # _, preds = torch.max(outputs, 1)
            # preds = preds.type(torch.FloatTensor)
            # preds = preds.to(device)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

            pred = outputs.squeeze(1)
            pred = pred.cpu().detach().numpy()
            #print('pred: {}'.format(pred))


            labels = labels.cpu().detach().numpy()
            #print('labels: {}'.format(labels))
            running_corrects += np.sum(np.equal(np.argmax(pred), labels))


        # for descriptors, labels in tqdm(dataloader_train, total = len(dataloader_train)):
        #     #print(descriptors.size())
        #     model.train()
        #     optimizer.zero_grad()
        #     #descriptors = descriptors.unsqueeze(0).unsqueeze(0)
        #     #print(descriptors.size())
        #     #descriptors = torch.permute(descriptors, (2, 0, 1, 3))
        #     #print(descriptors.size())
        #     descriptors = descriptors.to(device)
        #     labels = labels.type(torch.LongTensor)
        #     labels = labels.to(device)
        #     #print('TYPEEE {}'.format(labels.type()))
        #
        #
        #
        #     outputs = model(descriptors)
        #     _, preds = torch.max(outputs, 1)
        #     preds = preds.type(torch.FloatTensor)
        #     preds = preds.to(device)
        #     #labels = labels.unsqueeze(1)
        #
        #     #print(labels)
        #     loss = criterion(outputs, labels)
        #     #loss.requires_grad = True
        #
        #     loss.backward()
        #     optimizer.step()
        #
        #     running_loss += loss.item() * descriptors.size(0)
        #     #sigmoid = torch.nn.Sigmoid()
        #     pred = outputs.squeeze(1)
        #     pred = pred.cpu().detach().numpy()
        #     #print(pred)
        #     labels = labels.cpu().detach().numpy()
        #     running_corrects += np.sum(np.equal(np.argmax(pred), labels))
        #     #print(np.sum(np.equal(labels, labels)))
        #     #running_corrects += np.sum(np.equal(labels, labels))
        #     #running_corrects += torch.sum(torch.eq(outputs.squeeze(1), labels.squeeze(1)))
        #     #print('output : {}'.format(outputs.squeeze(1)))
        #     #print('labels : {}'.format(labels.squeeze(1)))

            #scheduler.step()
        #print(len(dataloader))
        epoch_loss = (running_loss/len(train_labels))/batch_size
        #print(running_corrects)
        epoch_acc = running_corrects/len(train_labels)/batch_size
        print('model test after training: {}'.format(model(torch.from_numpy(train_features[0]).to(device).float().unsqueeze(0))))
        print('Epoch Train loss: {}    Epoch Train accuracy: {}'.format(epoch_loss, epoch_acc))

        with torch.no_grad():
            #model.eval()
            running_corrects_test = 0
            test_count = 0

            for j in range(len(test_labels)):
                #print(test_labels[0])
                #print(j)
                #print(test_labels[j])
                #print(torch.from_numpy(test_array).to(device).float().unsqueeze(0))
                if j ==0:
                    print('model test in test loop: {}'.format(model(torch.from_numpy(train_features[0]).to(device).float().unsqueeze(0))))
                features_test = torch.from_numpy(test_features[j]).to(device).float().unsqueeze(0)
                #labels = test_labels[j].type(torch.LongTensor)
                #labels_test = torch.LongTensor([train_labels[j]])
                labels_test = torch.tensor(test_labels[j]).type(torch.LongTensor)

                #print(labels_test)
                labels_test = labels_test.to(device)

                outputs_test = model(features_test)
                #sigmoid = torch.nn.Sigmoid()
                # _, preds = torch.max(outputs_test, 1)
                # preds = preds.type(torch.FloatTensor)
                # preds = preds.to(device)

                pred_test = outputs_test.squeeze(1)

                pred_test = pred_test.cpu().detach().numpy()#.round()
                #print('pred: {}'.format(pred_test))
                labels_test = labels_test.cpu().detach().numpy()
                #print('labels: {}'.format(labels_test))
                #print(test_labels[j])
                #if np.argmax(pred_test) == 0:
                    #print('predicted class 0')

                #running_corrects_test += np.sum(np.equal(test1, labels_test))
                running_corrects_test += np.sum(np.equal(np.argmax(pred_test), labels_test))

                test_count+=1
            #
            #
            # for descriptors, labels in tqdm(dataloader_test, total = len(dataloader_test)):
            #     #descriptors = descriptors.unsqueeze(0).unsqueeze(0)
            #     descriptors = descriptors.to(device)
            #     labels = labels.type(torch.FloatTensor)
            #     labels = labels.to(device)
            #
            #     outputs = model(descriptors)
            #     #sigmoid = torch.nn.Sigmoid()
            #     pred = outputs.squeeze(1)
            #     pred = pred.cpu().detach().numpy().round()
            #     labels = labels.cpu().detach().numpy()
            #     running_corrects_test += np.sum(np.equal(np.argmax(pred), labels))
            #     #running_corrects_test += np.sum(np.equal(labels, labels))
            #
            #     # if test_count < 20:
            #     #     idx = randrange(batch_size)
            #     #     print(imgs.permute(0, 2, 3, 1).cpu().detach().numpy()[idx].shape)
            #     #     #im = Image.fromarray(imgs.permute(0, 2, 3, 1).cpu().detach().numpy()[idx].astype(np.uint8))
            #     #     tf = transforms.ToPILImage()
            #     #     im = tf(imgs[idx])
            #     #     im = im.save('test/label{}_output{}.jpg'.format(labels[idx], sigmoid(outputs[idx])))
            #
            #
            #     test_count+=1
            epoch_acc_test = running_corrects_test/len(test_labels)/batch_size
            #epoch_acc_test = running_corrects_test/len(test_labels)/batch_size
            print('Epoch Test accuracy: {}'.format(epoch_acc_test))

        if epoch_acc_test > best_acc:
            best_acc = epoch_acc_test
            print('New best model')
            best_model = model

    model.eval()
    best_model.eval()
    print('saving best model, with test accuracy {}'.format(best_acc))
    #torch.save(best_model.state_dict(), 'trained_NN_classifiers/trained_NN_classifier_{}layers_tech{}.pth'.format(n_layers, n_techniques))
    return best_model



# datasets = ['pitts30k', 'msls']
# n_images = 1000
# train_test_split = 0.8
# batch_size = 1
# num_epochs = 20
# #n_features = 16384


def get_NN_classifier(features_dim, n_layers, features, labels):

    #pos_weight = torch.tensor([10., 1.]).to(device).double()
    #print(pos_weight)
    train_test_split = 0.8
    criterion = torch.nn.CrossEntropyLoss()

    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    num_epochs = 10
    batch_size = 1

    model = initiate_classifier(features_dim, 2, n_layers)
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    random_indices = random.sample(range(0, len(labels)), len(labels))

    shuffled_features = features[random_indices]
    #shuffled_features = features
    del features
    shuffled_labels = [labels[i] for i in random_indices]

    # shuffled_labels = labels
    #
    # shuffled_features1 = shuffled_features[0:int((len(shuffled_features)/2))]
    # shuffled_features2 = shuffled_features[int((len(shuffled_features)/2)):]
    #
    # shuffled_labels1 = shuffled_labels[0:int((len(shuffled_labels)/2))]
    # shuffled_labels2 = shuffled_labels[int((len(shuffled_labels)/2)):]
    #
    # train_features = np.concatenate((shuffled_features1[0:int(train_test_split*len(shuffled_features1))], shuffled_features2[0:int(train_test_split*len(shuffled_features2))]), axis=0)
    # train_labels = np.concatenate((shuffled_labels1[0:int(train_test_split*len(shuffled_labels1))], shuffled_labels2[0:int(train_test_split*len(shuffled_labels2))]), axis=0)
    # test_features = np.concatenate((shuffled_features1[int(train_test_split*len(shuffled_features1)):],shuffled_features2[int(train_test_split*len(shuffled_features2)):]), axis=0)
    # test_labels = np.concatenate((shuffled_labels1[int(train_test_split*len(shuffled_labels1)):], shuffled_labels2[int(train_test_split*len(shuffled_labels2)):]), axis=0)
    # print('TEST')
    # print(train_features.shape)
    # print(train_labels.shape)
    # print(test_features.shape)
    # print(test_labels.shape)
    # print(np.count_nonzero(train_labels == 1))
    # print(np.count_nonzero(test_labels == 1))
    # print(train_labels[0])
    train_features = shuffled_features[0:int(train_test_split*len(labels))]
    train_labels = shuffled_labels[0:int(train_test_split*len(labels))]
    test_features = shuffled_features[int(train_test_split*len(labels)):]
    test_labels = shuffled_labels[int(train_test_split*len(labels)):]

    print(type(train_labels))
    print(type(test_labels))

    print('amount of label 1 in train labels: {}'.format(train_labels.count(1)))
    print('amount of label 1 in test labels: {}'.format(test_labels.count(1)))

    #train_labels = [0]*(int(train_test_split*len(labels)))
    #test_labels = [0]*(int((1-train_test_split)*len(labels)))

    #model = model.float()

    model = train_classifier(model, criterion, optimizer, num_epochs, train_features, train_labels, test_features, test_labels, batch_size, n_layers)


    return model



#
#
#
# criterion = torch.nn.CrossEntropyLoss()
# #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
#
# techniques, n_features_list = load_techniques()
# layers = [1, 2, 3]
#
# for j in range(len(layers)):
#
#     for i in range(len(techniques)):
#         print('TRAINING TECH {} WITH {} LAYERS'.format(i, layers[j]))
#         dataset = DescriptorDataset(datasets, n_images, techniques[i], n_features_list[i])
#         train_set, test_set = data.random_split(dataset, [round(train_test_split*len(dataset)), round((1-train_test_split)*len(dataset))])
#         dataloader_train = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
#         dataloader_test = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
#         print('TRAINING DATA SIZE: {}'.format(len(dataloader_train)))
#         print('TEST DATA SIZE: {}'.format(len(dataloader_test)))
#
#         classifier = initiate_classifier(n_features_list[i], 2, layers[j])
#         print(classifier)
#
#         optimizer = optim.Adam(classifier.parameters(), lr=0.0001)
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#
#         train_classifier(classifier, criterion, optimizer, num_epochs, dataloader_train, dataloader_test, scheduler, batch_size, n_features_list[i], layers[j], i)
