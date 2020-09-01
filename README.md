# Hand-crafted and Deep Learning-based CMR Radiomics for Diagnosis of Cardiovascular Diseases

Repository of documents for the final Master's thesis of the Master's in Fundamentals in Data Science at the University of Barcelona. This thesis was supervised by Polyxeni 
Gkontra and Karim Lekadir from the Artitificial Intelligence Labortary in Medicine at University of Barcelona. 

This repository corresponds to Jupyter Notebooks and Python files to develop pipeline for the extraction of Hand Crafted Radiomics and Deep Learning-based radiomics
and ultimately, the fusion of these both types of features.  We also provide the code for the extraction of the radiomics from the <em> PyRadiomics</em> library and pre-processing
of the images. 

Alejandro Hernandez Matheus

#### Abstract
Cardiovascular diseases (CVDs) are subject of interest among researchers and clin-icians  due  its  high  mortality  rate.   Cardiac  Magnetic  resonance  (CMR)  is  the  reference  for  clinicians  to  analyze  the  heart  tissues  by  visual  assessment  and  crudequantitative measures of the structures to dictate a diagnosis of the patient’s status. Radiomics is a novel image analysis technique extract a large number of quantitative features from CMR that provide insightful information of the heart structures to  support  clinicians  in  the  diagnosis  and  prognosis  of  these  diseases.   Previous studies have demonstrated the capacity of these features to obtain higher diagnosis accuracy than conventional methods. In this study, we cover in-depth Radiomicstechnique and explore two types of radiomics: Hand-Crafted Radiomics (HCR) and Deep Learning-based Radiomics (DLR). HCR computes a wide range of researcherdefined
quantitative features that measure the shape, intensity, and texture of image regions of interest. DLR arefeatures extracted based in the training of Convolutional Neural Networks (CNNs). We  address  the  methodology for  the  extraction  of  the  features  for  both  methods and analyze the performance in the CVDs classification task with Machine Learn-ing (ML) algorithm. We also develop a pipeline for the fusion of these features withthe aim of collecting complementary information of the heart structures from both mentioned methods with the aim of improving diagnostic accuracy.  We apply this methodology with two benchmark medical datasets: ACDC Challenge dataset andUK BioBank with the availability of both CMR and the segmentation of the heart structures:  Myocardium  (MYO),  Right  Ventricle  (RV)  and  Left  Ventricle  (LV).  We perform an analysis of the results, discuss challenges and elaborate on future work.

<p align="center"><img src="https://github.com/ahhm14/FinalMasterProject-AHHM/blob/master/X.%20Report/Figures/Fusion%20Pipeline%202.png" align=middle width=645.87435pt height=348.58725pt/>
</p>
<p align="center">
<em>Pipeline developed for the fusion of the two types of Radiomics.</em>
</p>


## Methods Applied

> - Radiomics
> - Machine Learning (ML)
> - Deep Learning (DL)
> - Fusion of Features 


## Contributions
Contributions are welcome! For bug reports or requests please submit an [submit an issue](https://github.com/ahhm14/FinalMasterProject-AHHM//issues).

## Contact
Feel free to contact me to discuss any issues, questions or comments.
* GitHub: [ahhm14](https://github.com/ahhm14)
* Linkedin: [Alejandro Hernandez Matheus](https://www.linkedin.com/in/alejandro-hernandez-matheus/)


## License

The content developed by Alejandro Hernandez is distributed under the following license:

    Copyright 2020 Alejandro Hernandez - UB

