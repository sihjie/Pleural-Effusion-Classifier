This study presents a system that combines ultrasound images and clinical data to classify pleural effusions as benign or malignant. Using 100 cases, the workflow includes image preprocessing, deep segmentation, ROI expansion, texture and deep feature extraction, and dimensionality reduction. Three feature combinations were tested with multiple machine-learning models, with the best performance achieved by integrating deep, texture, and clinical features.



The program uses three separate virtual environments, and the package lists for each environment have been exported. The correspondence between the environments and the code is as follows:
* .venvFeatureExtract：/Feature_extraction
* .venvPE：/Preprocessing, /Classification/ML
* .venvpytorch：/Segmentation, /Classification/DL, /Classification/DL_ML
