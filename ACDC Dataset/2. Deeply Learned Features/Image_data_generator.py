#!/usr/bin/env python
# coding: utf-8

# ## Deep Learning Testing

# In[5]:


import os
import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import cv2



train_dir = r'C:\Users\alex1\Documents\Fundamentals_of_Data_Science\PFM\Datasets\ACDC Dataset\training_acdc\training'
test_dir = r'C:\Users\alex1\Documents\Fundamentals_of_Data_Science\PFM\Datasets\ACDC Dataset\testing_acdc\testing'


# In[56]:


class Dataset():

    def __init__(self, path, counter_max=0, type='4D'):

        self.path = path
        self.class_folders = [folder for folder in os.listdir(self.path) if 'class' in folder]
        self.dataset = {'img_filenames': [], 'msk_ed': [], 'msk_es': []}
        # self.es_frames = pd.read_excel(os.path.join(self.path,"ES_frames.xlsx"))
        self.patient_ids = []
        self.search_string_fr_ed = []
        self.search_string_msk_ed = []
        self.search_string_fr_es = []
        self.search_string_msk_es = []

        # img_path = os.path.join(path,'image')
        # seg_path = os.path.join(path,'segs')

        counter = 0

        for file in os.walk(path):

            if counter_max != 0 and counter > counter_max:
                break

            if file[0] == path:
                continue

            #             file[2].sort()
            #             if ".DS_Store" in file[2]:
            #                 file[2].remove(".DS_Store")

            self.dataset['msk_ed'].append(os.path.join(file[0], file[2][3]))
            self.dataset['msk_es'].append(os.path.join(file[0], file[2][4]))

            patient_id = os.path.basename(file[0])
            search_string = os.path.join(path,  patient_id , patient_id + file[2][2] + ".nii.gz")  #what is this for really?
            
            self.search_string_fr_ed.append(os.path.join(path,  patient_id,  file[2][-4]))
            self.search_string_msk_ed.append(os.path.join(path,  patient_id , file[2][-3]))
            self.search_string_fr_es.append(os.path.join(path,  patient_id ,  file[2][-2]))
            self.search_string_msk_es.append(os.path.join(path,  patient_id ,  file[2][-1]))


            #     pdb.set_trace()
            image_location = glob.glob(search_string)

            self.patient_ids.append(patient_id)

            self.dataset['img_filenames'].append(image_location)

            #print(self.dataset['img_filenames'][-1])

def MinMaxScaled(x):
    xnew= ((x-np.min(x))/(np.max(x)-np.min(x)))
    return xnew

def get_classes(path):
    
    class_list = []
    for file in os.walk(path):
        cont = 0
        if file[0] == path:
            continue
        with open ((os.path.join(file[0], file[2][0]))) as myfile:
            for line in myfile:
                cont += 1
                if cont==3:
                    classs = line.lstrip("Group: ")
                    classs = classs.rstrip("\n")
                    class_list.append(classs)
    
    y_array1 = LabelEncoder().fit_transform(class_list)
    y = to_categorical(y_array1, num_classes=5)
    
    return y


# Lets apply the cv2 resizing
def get_dataset_short(search_dir):
    ''' Returns each structure in the the center frame as a channel
     X_ed = (150,150, 3)
     X_es = (150,150,3)
     X_tot = (150, 150, 3) '''
    shape = (150, 150)

    dataset = Dataset(search_dir)
    final_array = []
    for x in range(len(dataset.patient_ids)):
        print(x)
        f_ed = nib.load(dataset.search_string_fr_ed[x])

        array = np.array(f_ed.dataobj)
        num_channels = array.shape[2]
        sel_channels = round(num_channels / 2)
        print("Selected Channels", sel_channels)
        array_1 = array[:, :, [sel_channels - 1]]
        #         array_1 = MinMaxScaled(array_1)
        array_11 = cv2.resize(array_1, shape, interpolation=cv2.INTER_CUBIC)  # Frame ED

        msk_ed = nib.load(dataset.search_string_msk_ed[x])

        array2 = np.array(msk_ed.dataobj)
        array_2 = array2[:, :, [sel_channels - 1]]
        #         array_2 = MinMaxScaled(array_2)
        array_22 = cv2.resize(array_2, shape, interpolation=cv2.INTER_CUBIC)  # Frame ES

        ed_ = MinMaxScaled(np.multiply(array_11, array_22))


        lved = MinMaxScaled(np.multiply(array_11, np.equal(array_22, 1)))
        #print("Patient: ", x, "Equal 1", "Max ", np.max(lved), "Min ", np.min(lved))

        myoed = MinMaxScaled(np.multiply(array_11, np.equal(array_22, 2)))
        #print("Patient: ", x, "Equal 2", "Max ", np.max(myoed), "Min ", np.min(myoed))

        rved = MinMaxScaled(np.multiply(array_11, np.equal(array_22, 3)))
        # print("Patient: ", x, "Equal 3", "Max ", np.max(rved), "Min ", np.min(rved))

        f_es = nib.load(dataset.search_string_fr_es[x])

        array3 = np.array(f_es.dataobj)
        array_3 = array3[:, :, [sel_channels - 1]]
        #         array_3 = MinMaxScaled(array_3)
        array_33 = cv2.resize(array_3, shape, interpolation=cv2.INTER_CUBIC)

        msk_es = nib.load(dataset.search_string_msk_ed[x])

        array4 = np.array(msk_es.dataobj)
        array_4 = array4[:, :, [sel_channels - 1]]
        #         array_4 = MinMaxScaled(array_4)
        array_44 = cv2.resize(array_4, shape, interpolation=cv2.INTER_CUBIC)

        es_ = np.multiply(array_33, array_44)
        #         es_input = np.resize(es_, (256, 256,1))

        lves = MinMaxScaled(np.multiply(array_33, np.equal(array_44, 1)))
        myoes = MinMaxScaled(np.multiply(array_33, np.equal(array_44, 2)))
        rves = MinMaxScaled(np.multiply(array_33, np.equal(array_44, 3)))

        # #         plt.imshow(lves[:,:,0])
        # #         plt.show()
        # print("Patient: ", x, "Equal 1", "Max ", np.max(lves), "Min ", np.min(lves))
        # #         plt.imshow(myoes[:,:,0])
        # #         plt.show()
        # print("Patient: ", x, "Equal 2", "Max ", np.max(myoes), "Min ", np.min(myoes))
        # #         plt.imshow(rves[:,:,0])
        # #         plt.show()
        # print("Patient: ", x, "Equal 3", "Max ", np.max(rves), "Min ", np.min(rves))

        # final = np.dstack((array_1, array_2, array_3, array_4))  #Frame and Mask without modification
        # final2 = np.dstack((array_11, array_22, array_33, array_44))   #Frame and Mask resized (not normalized)
        # final3 = np.dstack((array_11, array_22, ed_, array_33, array_44, es_))   #frame mask, multiplication

        final4 = np.dstack((lved, myoed, rved, lves, myoes, rves))

        final_array.append(final4)

    X_set = np.asanyarray(final_array)

    X_ed = X_set[:, :, :, [0, 1, 2]]
    print("X_ED array Shape ", X_ed.shape)
    print("X_ED - Min {} - Max {}".format(np.min(X_ed), np.max(X_ed)))

    X_es = X_set[:, :, :, [3, 4, 5]]
    print("X_ES array Shape ", X_es.shape)
    print("X_ES - Min {} - Max {}".format(np.min(X_es), np.max(X_ed)))

    return X_ed, X_es, X_set

# ----------------------------------------------------------------------------



# Adding the channels before and after the middle one
def get_dataset2(search_dir):
    '''# Adding the channels before and after the middle one
    Neighbours slices for each cycle and heart structure add up to 6x each patient'''
    shape = (150, 150)

    dataset = Dataset(search_dir)

    final_array2 = []
    for x in range(len(dataset.patient_ids)):
        final_array = []
        patient = x+1
        print("Patient  ----------------------------------- ", patient)
        f_ed = nib.load(dataset.search_string_fr_ed[x])

        array = f_ed.get_fdata()
        array = np.array(f_ed.dataobj)
        num_channels = array.shape[2]
        sel_c = round(num_channels / 2)
        print("Selected Channels", sel_c)
        array_1 = array[:, :, [sel_c - 1, sel_c, sel_c + 1]]

        array_11 = cv2.resize(array_1, shape, interpolation=cv2.INTER_CUBIC)  # Frame ED
        print(array_11.shape)

        msk_ed = nib.load(dataset.search_string_msk_ed[x])

        array2 = msk_ed.get_fdata()
        array2 = np.array(msk_ed.dataobj)
        array_2 = array2[:, :, [sel_c - 1, sel_c, sel_c + 1]]
        #         array_2 = MinMaxScaled(array_2)
        array_22 = cv2.resize(array_2, shape, interpolation=cv2.INTER_CUBIC)  # Frame ES

        print('ED')
        for i in range(array_1.shape[2]):
            print("Channel ", i)
            lved_ = MinMaxScaled(np.multiply(array_11[:, :, i], np.equal(array_22[:, :, i], 1)))
            # print("Patient: ", patient, "Equal 1", "Max ", np.max(lved_), "Min ", np.min(lved_))

            final_array.append(lved_)

            myoed = MinMaxScaled(np.multiply(array_11[:, :, i], np.equal(array_22[:, :, i], 2)))
            # print("Patient: ", patient, "Equal 2", "Max ", np.max(myoed), "Min ", np.min(myoed))

            final_array.append(myoed)

            rved = MinMaxScaled(np.multiply(array_11[:, :, i], np.equal(array_22[:, :, i], 3)))
            # print("Patient: ", patient, "Equal 3", "Max ", np.max(rved), "Min ", np.min(rved))

            final_array.append(rved)

        f_es = nib.load(dataset.search_string_fr_es[x])

        array3 = f_es.get_fdata()
        array3 = np.array(f_es.dataobj)
        array_3 = array3[:, :, [sel_c - 1, sel_c, sel_c + 1]]
        #         array_3 = MinMaxScaled(array_3)
        array_33 = cv2.resize(array_3, shape, interpolation=cv2.INTER_CUBIC)

        msk_es = nib.load(dataset.search_string_msk_ed[x])

        array4 = msk_es.get_fdata()
        array4 = np.array(msk_es.dataobj)
        array_4 = array4[:, :, [sel_c - 1, sel_c, sel_c + 1]]
        #         array_4 = MinMaxScaled(array_4)
        array_44 = cv2.resize(array_4, shape, interpolation=cv2.INTER_CUBIC)

        #         es_ = np.multiply(array_33, array_44)
        # #         es_input = np.resize(es_, (256, 256,1))

        print('ES')

        for i in range(array_1.shape[2]):
            lves = MinMaxScaled(np.multiply(array_33[:, :, i], np.equal(array_44[:, :, i], 1)))
            # print("Patient: ", patient, "Equal 1", "Max ", np.max(lves), "Min ", np.min(lves))
            final_array.append(lves)
            myoes = MinMaxScaled(np.multiply(array_33[:, :, i], np.equal(array_44[:, :, i], 2)))
            # print("Patient: ", patient, "Equal 2", "Max ", np.max(myoes), "Min ", np.min(myoes))
            final_array.append(myoes)
            rves = MinMaxScaled(np.multiply(array_33[:, :, i], np.equal(array_44[:, :, i], 3)))
            # print("Patient: ", patient, "Equal 3", "Max ", np.max(rves), "Min ", np.min(rves))
            final_array.append(rves)

        test = np.dstack(final_array)

        final_array2.append(test)

    X_set = np.asanyarray(final_array2)

    X_ed = X_set[:, :, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
    print("X_ED array Shape ", X_ed.shape)
    print("X_ED - Min {} - Max {}".format(np.min(X_ed), np.max(X_ed)))


    X_es = X_set[:, :, :, [9, 10, 11, 12, 13, 14, 15, 16, 17]]
    print("X_ES array Shape ", X_es.shape)
    print("X_ES - Min {} - Max {}".format(np.min(X_es), np.max(X_ed)))

    return X_ed, X_es, X_set

# Adding the channels before and after the middle one
def get_dataset3(search_dir):
    '''# Stacks the Slices vertically before and after the middle one
        # X_ex =(150,150,9)
        # X_tot = (150,150,18)'''
    shape = (150, 150)

    dataset = Dataset(search_dir)

    final_array2 = []
    for x in range(len(dataset.patient_ids)):
        final_array = []
        patient = x+1
        print("Patient  ----------------------------------- ", patient)
        f_ed = nib.load(dataset.search_string_fr_ed[x])

        array = f_ed.get_fdata()
        array = np.array(f_ed.dataobj)
        num_channels = array.shape[2]
        sel_c = round(num_channels / 2)
        print("Selected Channels", sel_c)
        array_1 = array[:, :, [sel_c - 1, sel_c, sel_c + 1]]

        array_11 = cv2.resize(array_1, shape, interpolation=cv2.INTER_CUBIC)  # Frame ED
        print(array_11.shape)

        msk_ed = nib.load(dataset.search_string_msk_ed[x])

        array2 = msk_ed.get_fdata()
        array2 = np.array(msk_ed.dataobj)
        array_2 = array2[:, :, [sel_c - 1, sel_c, sel_c + 1]]
        #         array_2 = MinMaxScaled(array_2)
        array_22 = cv2.resize(array_2, shape, interpolation=cv2.INTER_CUBIC)  # Frame ES

        print('ED')
        for i in range(array_1.shape[2]):
            print("Channel ", i)
            lved_ = MinMaxScaled(np.multiply(array_11[:, :, i], np.equal(array_22[:, :, i], 1)))
            # print("Patient: ", patient, "Equal 1", "Max ", np.max(lved_), "Min ", np.min(lved_))

            final_array.append(lved_)

            myoed = MinMaxScaled(np.multiply(array_11[:, :, i], np.equal(array_22[:, :, i], 2)))
            # print("Patient: ", patient, "Equal 2", "Max ", np.max(myoed), "Min ", np.min(myoed))

            final_array.append(myoed)

            rved = MinMaxScaled(np.multiply(array_11[:, :, i], np.equal(array_22[:, :, i], 3)))
            # print("Patient: ", patient, "Equal 3", "Max ", np.max(rved), "Min ", np.min(rved))

            final_array.append(rved)

        f_es = nib.load(dataset.search_string_fr_es[x])

        array3 = f_es.get_fdata()
        array3 = np.array(f_es.dataobj)
        array_3 = array3[:, :, [sel_c - 1, sel_c, sel_c + 1]]
        #         array_3 = MinMaxScaled(array_3)
        array_33 = cv2.resize(array_3, shape, interpolation=cv2.INTER_CUBIC)

        msk_es = nib.load(dataset.search_string_msk_ed[x])

        array4 = msk_es.get_fdata()
        array4 = np.array(msk_es.dataobj)
        array_4 = array4[:, :, [sel_c - 1, sel_c, sel_c + 1]]
        #         array_4 = MinMaxScaled(array_4)
        array_44 = cv2.resize(array_4, shape, interpolation=cv2.INTER_CUBIC)

        #         es_ = np.multiply(array_33, array_44)
        # #         es_input = np.resize(es_, (256, 256,1))

        print('ES')

        for i in range(array_1.shape[2]):
            lves = MinMaxScaled(np.multiply(array_33[:, :, i], np.equal(array_44[:, :, i], 1)))
            # print("Patient: ", patient, "Equal 1", "Max ", np.max(lves), "Min ", np.min(lves))
            final_array.append(lves)
            myoes = MinMaxScaled(np.multiply(array_33[:, :, i], np.equal(array_44[:, :, i], 2)))
            # print("Patient: ", patient, "Equal 2", "Max ", np.max(myoes), "Min ", np.min(myoes))
            final_array.append(myoes)
            rves = MinMaxScaled(np.multiply(array_33[:, :, i], np.equal(array_44[:, :, i], 3)))
            # print("Patient: ", patient, "Equal 3", "Max ", np.max(rves), "Min ", np.min(rves))
            final_array.append(rves)

        test = np.hstack(final_array)

        final_array2.append(test)

    X_set = np.asanyarray(final_array2)

    # X_ed = X_set[:, :, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
    # print("X_ED array Shape ", X_ed.shape)
    # print("X_ED - Min {} - Max {}".format(np.min(X_ed), np.max(X_ed)))
    #
    #
    # X_es = X_set[:, :, :, [9, 10, 11, 12, 13, 14, 15, 16, 17]]
    # print("X_ES array Shape ", X_es.shape)
    # print("X_ES - Min {} - Max {}".format(np.min(X_es), np.max(X_ed)))

    return X_set


# Adding the channelbefore and after the middle one
# We want an array (600, 150, 150, 3)
# Where each patient has 2x3 inputs, 2 for the frame (ES and ED) and 3 for the each slice.
# Final channel corresponds to the structures.
# Lets apply the cv2 resizing
def get_dataset_long(search_dir):
    '''Format corresponds to the end wanted
    0 = ED  . 1 = ES ,  2 = Both  '''
    shape = (150, 150)

    dataset = Dataset(search_dir)

    final_array2 = []
    for x in range(len(dataset.patient_ids)):
        #         final_array = []
        print("Patient  ----------------------------------- ", x + 1)
        f_ed = nib.load(dataset.search_string_fr_ed[x])

        array = np.array(f_ed.dataobj)
        num_channels = array.shape[2]
        sel_c = round(num_channels / 2)
        print("Selected Channels", sel_c)
        array_1 = array[:, :, [sel_c - 1, sel_c, sel_c + 1]]

        array_11 = cv2.resize(array_1, shape, interpolation=cv2.INTER_CUBIC)  # Frame ED
        print(array_11.shape)

        #         for x in range(array_11.shape[2]):
        #             plt.imshow(array_11[:,:,x])
        #             plt.show()

        msk_ed = nib.load(dataset.search_string_msk_ed[x])

        array2 = np.array(msk_ed.dataobj)
        array_2 = array2[:, :, [sel_c - 1, sel_c, sel_c + 1]]
        #         array_2 = MinMaxScaled(array_2)
        array_22 = cv2.resize(array_2, shape, interpolation=cv2.INTER_CUBIC)  # Frame ES

        #         for x in range(array_2.shape[2]):
        #             plt.imshow(array_2[:,:,x])
        #             plt.show()

        #         ed_ = MinMaxScaled(np.multiply(array_11[:,:,0], array_22[:,:,0]))
        #         print(ed_.shape)
        #         ed_input = cv2.resize(ed_, (96,96), interpolation = cv2.INTER_CUBIC)  #Multiplication

        #         for x in range(ed_.shape[2]):
        #             plt.imshow(ed_[:,:,x])
        #             plt.show()

        # channel_1

        for i in range(array_1.shape[2]):
            final_array = []
            print("Channel ", i)
            lved_ = MinMaxScaled(np.multiply(array_11[:, :, i], np.equal(array_22[:, :, i], 1)))
            print("Patient: ", x, "Equal 1", "Max ", np.max(lved_), "Min ", np.min(lved_))
            plt.imshow(lved_)
            plt.show()
            final_array.append(lved_)

            myoed = MinMaxScaled(np.multiply(array_11[:, :, i], np.equal(array_22[:, :, i], 2)))
            print("Patient: ", x, "Equal 2", "Max ", np.max(myoed), "Min ", np.min(myoed))
            plt.imshow(myoed)
            plt.show()
            final_array.append(myoed)

            rved = MinMaxScaled(np.multiply(array_11[:, :, i], np.equal(array_22[:, :, i], 3)))
            print("Patient: ", x, "Equal 3", "Max ", np.max(rved), "Min ", np.min(rved))
            plt.imshow(rved)
            plt.show()
            final_array.append(rved)
            print(i)

            inter_array = np.dstack(final_array)

            print('inter_array shape', inter_array.shape)

            final_array2.append(inter_array)

        f_es = nib.load(dataset.search_string_fr_es[x])

        array3 = np.array(f_es.dataobj)
        array_3 = array3[:, :, [sel_c - 1, sel_c, sel_c + 1]]
        #         array_3 = MinMaxScaled(array_3)
        array_33 = cv2.resize(array_3, shape, interpolation=cv2.INTER_CUBIC)

        msk_es = nib.load(dataset.search_string_msk_ed[x])

        array4 = msk_es.get_fdata()
        array4 = np.array(msk_es.dataobj)
        array_4 = array4[:, :, [sel_c - 1, sel_c, sel_c + 1]]
        #         array_4 = MinMaxScaled(array_4)
        array_44 = cv2.resize(array_4, shape, interpolation=cv2.INTER_CUBIC)

        #         es_ = np.multiply(array_33, array_44)
        # #         es_input = np.resize(es_, (256, 256,1))

        for i in range(array_1.shape[2]):
            final_array = []
            lves = MinMaxScaled(np.multiply(array_33[:, :, i], np.equal(array_44[:, :, i], 1)))
            final_array.append(lves)
            myoes = MinMaxScaled(np.multiply(array_33[:, :, i], np.equal(array_44[:, :, i], 2)))
            final_array.append(myoes)
            rves = MinMaxScaled(np.multiply(array_33[:, :, i], np.equal(array_44[:, :, i], 3)))
            final_array.append(rves)

            inter_array = np.dstack(final_array)

            print('inter_array shape', inter_array.shape)

            final_array2.append(inter_array)

    X_tot = np.asanyarray(final_array2)

    print(X_tot.shape)

    return X_tot


def y_class_long(y):
    list_y = []
    for i in range(len(y)):
        for x in range(0, 6):
            list_y.append(y[i])

    return np.asanyarray(list_y)

