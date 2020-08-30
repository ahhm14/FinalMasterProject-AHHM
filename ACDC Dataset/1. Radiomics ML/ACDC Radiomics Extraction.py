
import numpy as np
import os
import pprint as pp
import pandas as pd
import glob
import pdb
import nibabel as nib
import radiomics as rm
import pprint as pp
import matplotlib.pyplot as plt
import SimpleITK as sitk
#get_ipython().run_line_magic('matplotlib', 'inline')
import csv


seg_path= r'C:\Users\alex1\Documents\Fundamentals_of_Data_Science\PFM\Datasets\ACDC Dataset\training_acdc\training'


for file in os.walk(seg_path):
    if file[0] == seg_path:
        continue
    print(file[0])
    search_string = os.path.join(seg_path, os.path.basename(file[0]),  os.path.basename(file[0]) + ".nii.gz")
    #print(file[1])
    print(os.path.basename(file[0]))
    print(search_string)
    print(file[2][2])
    print(file[2][4],  "\n")


class Dataset():

    def __init__(self, path, counter_max=0, type='4D'):

        self.path = path
        self.class_folders = [folder for folder in os.listdir(self.path) if 'class' in folder]
        self.dataset = {'img_filenames': [], 'frame_ed': [], 'msk_ed': [], 'frame_es':[], 'msk_es':[]}
        # self.es_frames = pd.read_excel(os.path.join(self.path,"ES_frames.xlsx"))
        self.patient_ids = []

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

            self.dataset['frame_ed'].append(os.path.join(file[0], file[2][2]))  #mask
            self.dataset['msk_ed'].append(os.path.join(file[0], file[2][3]))  #image #dont pay attention to the "index"
            self.dataset['frame_es'].append(os.path.join(file[0], file[2][-2]))  #mask
            self.dataset['msk_es'].append(os.path.join(file[0], file[2][-1])) 

            patient_id = os.path.basename(file[0])
            search_string = os.path.join(path,  patient_id , patient_id + file[2][2] + ".nii.gz")  #what is this for really?

            #     pdb.set_trace()
            image_location = glob.glob(search_string)

            self.patient_ids.append(patient_id)

            self.dataset['img_filenames'].append(image_location)

            print(self.dataset['frame_ed'][-1])
            print(self.dataset['msk_ed'][-1])
            print(self.dataset['frame_es'][-1])
            print(self.dataset['msk_es'][-1])
            #print(self.dataset['img_filenames'][-1])





dataset = Dataset(seg_path)


len(dataset.patient_ids)


i = 0
feature_names = []
feature_values = []
for x in range(len(dataset.patient_ids)):
    print(" \n Patient ID: ", x, "\n")
    #if x == 1:
        #break
    img_ED = nib.load(dataset.dataset['frame_ed'][x]).get_fdata()
    msk_ED = nib.load(dataset.dataset['msk_ed'][x]).get_fdata()
    
    img_ES = nib.load(dataset.dataset['frame_es'][x]).get_fdata()
    msk_ES = nib.load(dataset.dataset['msk_es'][x]).get_fdata()
    
    
    sitk_img_ED = sitk.GetImageFromArray(np.array(img_ED, dtype=np.int16))   
    sitk_msk_ED = sitk.GetImageFromArray(np.array(msk_ED, dtype=np.int16))
    
    sitk_img_ES = sitk.GetImageFromArray(np.array(img_ES, dtype=np.int16))   
    sitk_msk_ES = sitk.GetImageFromArray(np.array(msk_ES, dtype=np.int16))
    
    
    #create an radiomics extractor

    extractor = rm.featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['label'] = 1   
    #features = extractor.execute(filpath_img, filpath_msk) # from file
    features = extractor.execute(sitk_img_ED, sitk_msk_ED) # from sitk image
    
    #feature_names = []
    #feature_values = []
    all_subjects = []

    
    # LV EXTRACTION : LABEL 1
       
    for key,value in features.items():
    
#         if 'shape' in key:
        print(key,value) 
            #print(value)
        feature_names.append(key+'_LV_ED')
        feature_values.append(value)
            
    print(len(feature_values))
    extractor.settings['label'] = 3   
    features = extractor.execute(sitk_img_ED, sitk_msk_ED) # from sitk image
    
    
    # RV EXTRACTION : LABEL 3
    for key,value in features.items():
    
#         if 'shape' in key:
        print(key,value) 
        feature_names.append(key+'_RV_ED')
        feature_values.append(value)
    
    print(len(feature_values))
          
    extractor.settings['label'] = 2   
    features = extractor.execute(sitk_img_ED, sitk_msk_ED) # from sitk image
    
    # MYOCARDIUM EXTRACTION : LABEL 2
    for key,value in features.items():
    
#         if 'general' not in key:
        print(key,value) 
        feature_names.append(key+'_MYO_ED')
        feature_values.append(value)
            
    print(len(feature_values))
    df = pd.DataFrame([feature_values], columns = feature_names)


list_ED= []
count = 0
for i in df.columns:
    if i == "diagnostics_Versions_PyRadiomics_LV_ED":
        list_ED.append(df.iloc[0,count:count+387])
        count = count + (387)

print('Number of Patients Loaded: ', len(list_ED))

df_2 = pd.DataFrame(list_ED)

### Extracting End Systole

i = 0
feature_names = []
feature_values = []
for x in range(len(dataset.patient_ids)):
    print(" \n Patient ID: ", x, "\n")
    #if x == 1:
        #break
    img_ED = nib.load(dataset.dataset['frame_ed'][x]).get_fdata()
    msk_ED = nib.load(dataset.dataset['msk_ed'][x]).get_fdata()
    
    img_ES = nib.load(dataset.dataset['frame_es'][x]).get_fdata()
    msk_ES = nib.load(dataset.dataset['msk_es'][x]).get_fdata()
    
    
    sitk_img_ED = sitk.GetImageFromArray(np.array(img_ED, dtype=np.int16))   
    sitk_msk_ED = sitk.GetImageFromArray(np.array(msk_ED, dtype=np.int16))
    
    sitk_img_ES = sitk.GetImageFromArray(np.array(img_ES, dtype=np.int16))   
    sitk_msk_ES = sitk.GetImageFromArray(np.array(msk_ES, dtype=np.int16))
    
    
    
    
    #create an radiomics extractor

    extractor = rm.featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['label'] = 1   
    #features = extractor.execute(filpath_img, filpath_msk) # from file
    features_ES = extractor.execute(sitk_img_ES, sitk_msk_ES) # from sitk image
    
    #feature_names = []
    #feature_values = []
    all_subjects = []

    
    # LV EXTRACTION : LABEL 1
       
    for key,value in features_ES.items():
    
#         if 'shape' in key:
        print(key,value) 
            #print(value)
        feature_names.append(key+'_LV_ES')
        feature_values.append(value)
            
    print(len(feature_values))
    extractor.settings['label'] = 3   
    features = extractor.execute(sitk_img_ES, sitk_msk_ES) # from sitk image
    
    
    # RV EXTRACTION : LABEL 3
    for key,value in features.items():
    
#         if 'shape' in key:
        print(key,value) 
        feature_names.append(key+'_RV_ES')
        feature_values.append(value)
    
    print(len(feature_values))
          
    extractor.settings['label'] = 2   
    features = extractor.execute(sitk_img_ES, sitk_msk_ES) # from sitk image
    
    # MYOCARDIUM EXTRACTION : LABEL 2
    for key,value in features.items():
    
#         if 'general' not in key:
        print(key,value) 
        feature_names.append(key+'_MYO_ES')
        feature_values.append(value)
            
    print(len(feature_values))
    df = pd.DataFrame([feature_values], columns = feature_names)


list_ES= []
count = 0
for i in df.columns:
    if i == "diagnostics_Versions_PyRadiomics_LV_ES":
        list_ES.append(df.iloc[0,count:count+387])
        count = count + (387)


# In[19]:


print('Number of Patients Loaded', len(list_ES))


df_3 = pd.DataFrame(list_ES)

df_radiomcis = pd.concat((df_2, df_3), axis=1)

df_radiomcis


df_radiomcis.to_csv('ACDC_Radiomics_Training.csv', index = False)


mylines = []
height_list = []
weight_list = []
for file in os.walk(seg_path):
    cont = 0
    if file[0] == seg_path:
        continue
    #print(os.path.join(file[0], file[2][0]))
    search_string = os.path.join(seg_path, os.path.basename(file[0]),  os.path.basename(file[0]) + ".nii.gz")
    #print(file[1])
#     print(os.path.basename(file[0]))
#     #print(search_string)
#     print(file[2][0])
#     print(file[2][4],  "\n")
    
    
    with open ((os.path.join(file[0], file[2][0]))) as myfile:
        for myline in myfile:
            cont += 1
            if cont==3:
                classs = myline.lstrip("Group: ")
                classs = classs.rstrip("\n")
                mylines.append(classs)
            if cont == 4:
                height = myline.lstrip("Height: ")
                height = height.rstrip("\n")
                height_list.append(height)
            if cont == 6:
                weight = myline.lstrip("Weight: ")
                weight = weight.rstrip("\n")
                weight_list.append(weight)
            #print(cont)


class_df = pd.DataFrame(mylines)
height_df = pd.DataFrame(height_list)
weight_df = pd.DataFrame(weight_list)



df_radiomcis['height']= height_list
df_radiomcis['weight']= weight_list
df_radiomcis['class']= mylines



df_radiomcis.head()

df_radiomcis.to_csv('ACDC_(Radiomics+Clinical)_Training.csv', index = False)

