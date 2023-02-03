import os
import glob

def get_subset(data_path, subsampling_factor):
    
    """
    Function that performs subsampling as described in the report
    
    Input:
    data_path           string, path to dataset
    subsampling_factor  int, factor of subsampling applied
    
    Output:
    subset              list, contains filenames of newly created subset
    
    """

    filenames = sorted(glob.glob(os.path.join(data_path, '*.jpg')))
    
    ### create disct with images sorted by zone ###
    zone_dict = {}


    for filename in filenames:
        ### Sort images by zone ###
        zone = filename.split("@")[3]

        if zone in zone_dict:
            tmp_list = zone_dict[zone]
            tmp_list.append(filename)
            zone_dict[zone] = tmp_list

        else:
            zone_dict[zone] = [filename]
            
    
    ### create dict sorting zone_dist by zone letter ###
    zone_letter_dict = dict.fromkeys(zone_dict.keys())


    for key in zone_letter_dict:
        ### sort by zone letter ###
        
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
        ### sort images of each zone-letter combination by filename (easting) ###

        for j in zone_letter_dict[i].keys():
            tmp_list = sorted(zone_letter_dict[i][j])
            final_filenames.extend(tmp_list)
            print('number of filenames for zone {} letter {}: {}'.format(i, j, len(zone_letter_dict[i][j])))

    #print(*final_filenames[0:20], sep='\n')


    subset = []
    
    ### only save 1 in *subsampling_factor* to subset ###
    for i in range(len(final_filenames)):
        if (i % subsampling_factor) == 0:
            subset.append(final_filenames[i])

    return subset


def fuse_similarities(type, D_list, n_recalls, tech=None):
    """Function that fuses similarity vectors
    
    Input:
    type:       type of fusing used (avg voting, max_voting, etc)
    D_list:     List containing the similarity vectors of all techniques
    n_recalls:  int, return recall@n_recalls
    
    Output:
    output      list, contains the *n_recalls* best mathcing ref images according to fused similarity vector

    """


    if type == 'avg_voting':
        ### Average all similarity vectors in D_list ###
        D_avg = sum(D_list)/len(D_list)
        ### order ref image indices by similarity score ###
        order = D_avg.argsort()
        output = order[:n_recalls]

    if type == 'individual':
        D_ind = D_list[tech]
        order = D_ind.argsort()

        output = order[:n_recalls]

    if type == 'max_voting':
        output = np.zeros(n_recalls)
        pred_list = []

        for i in range(len(D_list)):
            ### get closest match according to tech i ###
            pred = np.argmin(D_list[i])
            pred_list.append(pred)
        
        ### check which ref image was most commonly chosen as best match ###
        output[0] = Counter(pred_list).most_common(1)[0][0]

    if type == 'max_score':
        best_min_dist = 10000000000
        best_tech = None

        for i in range(len(D_list)):
            ### check minimal distance between query desc and any ref desc for tech i ###
            min_dist = np.amin(D_list[i])
            if min_dist < best_min_dist:
                ### if lowest distance so far, save it ###
                best_tech = i
                best_min_dist = min_dist


        ### return similarity vector of tech with lowest distance ###
        order = D_list[best_tech].argsort()
        output = order[:n_recalls]

    if type == 'dyn_mpf':
        sums = []

        # for j in range(len(D_list)):
        #     sums.append(sum(D_list[j]))

        D_array = np.array(D_list)

        combined_similarities = []
        scores = []


        print('sums of D vectors: {}'.format(sums))
        
        ### create list with every possible tech pair combination ###
        list_of_pairs = [(D_array[p1], D_array[p2]) for p1 in range(D_array.shape[0]) for p2 in range(p1+1,D_array.shape[0])]
        print('list of pairs length: {}'.format(len(list_of_pairs)))

        for i in range(len(list_of_pairs)):
            ### combain the similarity vectors of pair i ###
            D_1_norm = list_of_pairs[i][0]#/np.sum(list_of_pairs[i][0])
            D_2_norm = list_of_pairs[i][1]#/np.sum(list_of_pairs[i][1])
            combined_distances = D_1_norm + D_2_norm
            combined_similarity = 1 - combined_distances
            #print('max similarity score: {}'.format(np.amax(combined_similarity)))
            combined_similarities.append(combined_distances)
            
            ### sort the commbined similarity vector by highest similarity score/lowest distance ###
            combined_similarity_sorted = combined_similarity[np.argsort(-combined_similarity)]
            #combined_similarity_sorted = combined_similarity[np.argsort(combined_similarity)]
            assert np.argmax(combined_similarity_sorted) == 0
            ### create cropped sim vector without the first R elements ###
            combined_similarity_sorted_cropped = combined_similarity_sorted[R:]
            assert np.argmax(combined_similarity_sorted_cropped) == 0
            
            ### calculate score of pair i by calculating perceptual aliasing###
            score = combined_similarity_sorted[0]/combined_similarity_sorted_cropped[0]

            scores.append(score)

        print('chosen pairing: {}'.format(np.argmax(np.array(scores))))
        
        ### obtain combined similarity vector of pair with highest score ###
        D_final = combined_similarities[np.argmax(np.array(scores))]

        order = D_final.argsort()
        #indices = [item for item in range(D_list[0].shape[0])]

        output = order[:n_recalls]

    if type == 'random_pair':

        D_array = np.array(D_list)
        list_of_pairs = [(D_array[p1], D_array[p2]) for p1 in range(D_array.shape[0]) for p2 in range(p1+1,D_array.shape[0])]

        chosen_pair = random.choice(list_of_pairs)
        D_final = np.array([sum(x) for x in zip(chosen_pair[0], chosen_pair[1])])
        order = D_final.argsort()

        output = order[:n_recalls]




    return output


def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim)).to(device)).data.shape

def norm_func(input, input_low, input_high, output_low=0., output_high=1.):
    range_old = input_high - input_low
    range_new = output_high - output_low
    print('normalization from range {}-{} ({}) to range {}-{} ({})'.format(input_low, input_high, range_old, output_low, output_high, range_new))
    #norm = ratio*(output_high-output_low)

    norm = (((input - input_low) * range_new) / range_old) + output_low

    return norm
