
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import os
import random
import sys
import time

from sklearn.decomposition import PCA

res = faiss.StandardGpuResources()

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def test_efficient_ram_usage(args, eval_ds, model, test_method="hard_resize"):
    """This function gives the same output as test(), but uses much less RAM.
    This can be useful when testing with large descriptors (e.g. NetVLAD) on large datasets (e.g. San Francisco).
    Obviously it is slower than test(), and can't be used with PCA.
    """

    model = model.eval()
    if test_method == 'nearest_crop' or test_method == "maj_voting":
        distances = np.empty([eval_ds.queries_num * 5, eval_ds.database_num], dtype=np.float32)
    else:
        distances = np.empty([eval_ds.queries_num, eval_ds.database_num], dtype=np.float32)

    with torch.no_grad():
        if test_method == 'nearest_crop' or test_method == 'maj_voting':
            queries_features = np.ones((eval_ds.queries_num * 5, args.features_dim), dtype="float32")
        else:
            queries_features = np.ones((eval_ds.queries_num, args.features_dim), dtype="float32")
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            features = model(inputs.to(args.device))
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            if test_method == "nearest_crop" or test_method == 'maj_voting':
                start_idx = (indices[0] - eval_ds.database_num) * 5
                end_idx = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                queries_features[indices, :] = features.cpu().numpy()
            else:
                queries_features[indices.numpy()-eval_ds.database_num, :] = features.cpu().numpy()

        queries_features = torch.tensor(queries_features).type(torch.float32).cuda()

        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            inputs = inputs.to(args.device)
            features = model(inputs)
            for pn, (index, pred_feature) in enumerate(zip(indices, features)):
                distances[:, index] = ((queries_features-pred_feature)**2).sum(1).cpu().numpy()
        del features, queries_features, pred_feature

    predictions = distances.argsort(axis=1)[:, :max(args.recall_values)]

    if test_method == 'nearest_crop':
        distances = np.array([distances[row, index] for row, index in enumerate(predictions)])
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each
    elif test_method == 'maj_voting':
        distances = np.array([distances[row, index] for row, index in enumerate(predictions)])
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query
    del distances

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break

    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str


def test_vg(device, techniques, args, eval_ds, model, test_method="hard_resize", pca=None, ds_aware=False, sim_function=None, n_queries=1000, load_desc=True, feature_pca=None, generative=False, fuse_type='avg_voting'):
    test_start_time = time.time()
    """Compute features of the given dataset and compute the recalls."""

    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"
    print('DEVICE: {}'.format(args.device))
    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method)

    # Creating array that will store the results per query image (correct/incorrectly matched, weights)
    result_array = np.empty((n_queries, 6))


    #####################################################
    #Generating database descriptors for every technique
    #####################################################


    #model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        #database_indices = random.sample(range(0, len(database_subset_ds)), n_database)
        #print(database_indices)
        #database_subset_ds = Subset(database_subset_ds, database_indices)
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))

        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        #query_indices = random.sample(range(0, len(queries_dataloader)), n_queries)
        #queries_dataloader_new = Subset(queries_dataloader, query_indices)

        print('size of reference dataloader: {}'.format(len(database_dataloader)))
        ref_features_list = []
        query_features_list = []
        features_dim_list = []
        tmp_features_dim_list = []
        pca_components = 1024

        #feature_pca = PCA(n_components = pca_components)


        def get_output_shape(model, image_dim):
            return model(torch.rand(*(image_dim)).to(device)).data.shape

        #features_dim_new = [1024, 1024, 256, 256, 512, 512]
        features_dim_new = [256, 256, 512, 512]
        if load_desc == True:

            for i in range(len(techniques)):
                ref_features_done = False
                query_features_done = False
                #features_dim_list.append(get_output_shape(techniques[i], (1, 3, 100, 100))[1])
                tmp_features_dim_list.append(get_output_shape(techniques[i], (1, 3, 100, 100))[1])

                #Load reference descriptors if possible
                if os.path.exists('descriptors/{}_REF_descriptors_tech{}.npy'.format(args.dataset_name, i)) == True:
                    ref_features = np.load('descriptors/{}_REF_descriptors_tech{}.npy'.format(args.dataset_name, i)).astype(np.float32)
                    ref_features_list.append(ref_features)
                    features_dim_list.append(ref_features.shape[1])
                    print('Loaded existing reference descriptors of {} (model {})'.format(args.dataset_name, i))
                    ref_features_done = True


                if test_method == "nearest_crop" or test_method == 'maj_voting':
                    all_features = np.empty((5 * eval_ds.queries_num + eval_ds.database_num, features_dim_new[i]), dtype="float32")

                #If reference descriptors cannot be loaded: calculate them
                if ref_features_done == False:
                    #all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")
                    all_features = np.empty((eval_ds.database_num, features_dim_new[i]), dtype="float32")
                    if tmp_features_dim_list[i] < pca_components:

                        features_dim_list.append(features_dim_new[i])

                    else:
                        #all_features = np.empty((eval_ds.database_num, 2048), dtype="float32")
                        features_dim_list.append(features_dim_new[i])


                    for inputs, indices in tqdm(database_dataloader, ncols=100):
                        features = techniques[i](inputs.to(device))
                        features = features.cpu().numpy()

                        # if tmp_features_dim_list[i] >= pca_components:
                        #
                        #     features = feature_pca[(i % 2)].transform(features)




                        if pca != None:
                            features = pca.transform(features)



                        all_features[indices.numpy(), :] = features

                        #labels_list.append(labels)

                    # if tmp_features_dim_list[i] >= pca_components:
                    #     print(features.shape)
                    #     #print(feature_pca)
                    #     all_features = feature_pca[i].transform(all_features)
                    #     all_features = np.ascontiguousarray(all_features)

                    ref_features_list.append(all_features)


                    np.save('descriptors/{}_REF_descriptors_tech{}.npy'.format(args.dataset_name, i), all_features)
                    print('saved reference features of tech {} on dataset {}'.format(i, args.dataset_name))

                #Load Query descriptors
                if os.path.exists('descriptors/{}_QUERY_descriptors_tech{}.npy'.format(args.dataset_name, i)) == True:
                    query_features = np.load('descriptors/{}_QUERY_descriptors_tech{}.npy'.format(args.dataset_name, i)).astype(np.float32)
                    query_features_list.append(query_features)

                    print('Loaded existing query descriptors of {} (model {})'.format(args.dataset_name, i))
                    query_features_done = True

                #If Query descriptors cannot be loaded: calculate them
                if query_features_done == False:
                    #all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

                    all_features = np.empty((eval_ds.queries_num, features_dim_new[i]), dtype="float32")




                    count = 0
                    for query_inputs, query_indices in tqdm(queries_dataloader, ncols=100):
                        #print(len(queries_dataloader))
                        #print(all_features.shape)
                        #print(query_indices)
                        features = techniques[i](query_inputs.to(device))
                        features = features.cpu().numpy()
                        # if tmp_features_dim_list[i] >= pca_components:
                        #
                        #     features = feature_pca[(i % 2)].transform(features)




                        if pca != None:
                            features = pca.transform(features)


                        #print(features.shape)
                        #print(query_indices)
                        all_features[count, :] = features
                        count+=1

                    query_features_list.append(all_features)


                    print(all_features.shape)
                    np.save('descriptors/{}_QUERY_descriptors_tech{}.npy'.format(args.dataset_name, i), all_features)
                    print('saved query features of tech {} on dataset {}'.format(i, args.dataset_name))



        if load_desc == False:

            for i in range(len(techniques)):
                features_dim_list.append(get_output_shape(techniques[i], (1, 3, 100, 100))[1])
                all_features = np.empty((eval_ds.database_num, features_dim_list[i]), dtype="float32")


                for inputs, indices in tqdm(database_dataloader, ncols=100):
                    features = techniques[i](inputs.to(device))
                    features = features.cpu().numpy()
                    if pca != None:
                        features = pca.transform(features)
                    #print('test features shape: {}'.format(features.shape))
                    #print(indices.numpy().shape)
                    #print(indices.numpy())
                    all_features[indices.numpy(), :] = features

                    #labels_list.append(labels)
                ref_features_list.append(all_features)

        print('all features shape: {}'.format(ref_features_list[0].shape))
        #print('all features shape: {}'.format(ref_features_list[1].shape))
        print('length of ref_features_list: {}'.format(len(ref_features_list)))

        #######################################################################
        #Generating query descriptors for every technique and make predictions
        #######################################################################

        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        # queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        # query_indices = random.sample(range(0, len(queries_subset_ds)), n_queries)
        # #print(query_indices)
        # queries_subset_ds = Subset(queries_subset_ds, query_indices)
        # queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
        #                                 batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))

        print('FEATURES DIM LIST: {}'.format(features_dim_list))

        feature_calc_time = time.time()

        labels_list = []

        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")
        #query_features = np.empty((n_queries, args.features_dim), dtype="float32")

        count = 0




        predictions = np.empty([n_queries, 20])
        matching_score = 'NaN'


        D_superlist = []
        prediction_superlist = []
        query_weights = []

        if ds_aware == True:

            query_weights = model(query_features_list)

            print('QUERY WEIGHTS CHECK:')
            print(len(query_weights))
            print(len(query_weights[0]))
        for i in range(len(techniques)):
            print('creating faiss index for technique {}'.format(i))

            nlist = 100
            m = 8
            k = 4
            print('test')
            print(features_dim_list[i])
            print(ref_features_list[0].shape)

            faiss_index = faiss.IndexFlatL2(features_dim_list[i])
            print(features_dim_list)


            faiss_index.add(ref_features_list[0].astype(np.float32))
            del ref_features_list[0]

            features_list = []
            print('describing query images')




            print('performing similarity search')
            features_array = np.squeeze(np.array(query_features_list[i].astype(np.float32)))
            print(features_array.shape)
            distances, predictions_tmp = faiss_index.search(features_array, eval_ds.database_num)
            #print(distances.shape)
            print('completed similarity search')
            #indices = predictions[0].argsort()
            #distances = distances[0][indices]/np.sum(distances[0][indices])

            prediction_superlist.append(predictions_tmp)
            D_superlist.append(distances)
        print('D_SUPERLIST:')
        print(len(D_superlist))
        print(D_superlist[0].shape)
        del ref_features_list, all_features
        count = 0

        for i in tqdm(range(eval_ds.queries_num), ncols=100):
            D_list = []
            for j in range(len(D_superlist)):
                indices = prediction_superlist[j][i].argsort()
                distances_D = D_superlist[j][i][indices]/np.sum(D_superlist[j][i][indices])
                #print(len(distances_D))

                if ds_aware == False:
                    weighted_D = distances_D
                    result_array[i, 2+j] = 0
                if ds_aware == True:
                    if generative == True:
                        weighted_D = distances_D*query_weights[j][i]
                        result_array[i, 2+j] = query_weights[j][i]
                    if generative == False:
                        #print((j%2))
                        #print(len(query_weights[i]))
                        #print(query_weights[i][(j % 2)])

                        weight = (query_weights[0][i] + query_weights[2][i]  + query_weights[1][i] + query_weights[3][i] )/4




                        #print(weight)
                        weighted_D = distances_D*weight[(j%2)]
                        result_array[i, 2+j] = query_weights[j][i][(j%2)]
                D_list.append(weighted_D)

            predictions[count, :] = sim_function(fuse_type, D_list, max(args.recall_values), args.indiv_tech)
            count += 1




    if test_method == 'nearest_crop':
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each query
    elif test_method == 'maj_voting':
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query


    #########################################################
    #### For each query, check if the predictions are correct
    #########################################################

    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    #for query_index, pred in enumerate(predictions):
    #print(predictions)
    # for j in range(len(predictions[0])):
    #print(query_indices)



    for j in range(n_queries):
        pred = predictions[j]
        #print(pred)
        #idx = query_indices[j]
        idx = j
        #print(positives_per_query[idx])
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[idx])):
                if i == 0:
                    result_array[j, 0], result_array[j, 1] = j, 1
                recalls[i:] += 1
                break
            else:
                if i == 0:
                    result_array[j, 0], result_array[j, 1] = j, 0

    # Divide by the number of queries*100, so the recalls are in percentages
    print('total recalls : {}'.format(recalls))
    recalls = recalls / n_queries* 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

    print('total test function time: {}'.format(time.time()-test_start_time))
    print('test function time excl. feature extraction/loading: {}'.format(time.time()-feature_calc_time))
    return recalls, recalls_str, matching_score, result_array


def top_n_voting(topn, predictions, distances, maj_weight):
    if topn == 'top1':
        n = 1
        selected = 0
    elif topn == 'top5':
        n = 5
        selected = slice(0, 5)
    elif topn == 'top10':
        n = 10
        selected = slice(0, 10)
    # find predictions that repeat in the first, first five,
    # or fist ten columns for each crop
    vals, counts = np.unique(predictions[:, selected], return_counts=True)
    # for each prediction that repeats more than once,
    # subtract from its score
    for val, count in zip(vals[counts > 1], counts[counts > 1]):
        mask = (predictions[:, selected] == val)
        distances[:, selected][mask] -= maj_weight * count/n
