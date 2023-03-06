Integrated RGB-Thermal Orthomosaicking Pipeline
=======================================

This repository contains the official code implementation for the paper ...
Official documentation can be found at https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/


![Summary of integrated workflow showing advantages over thermal-only workflows, specifically the lack of gaps and swirling artifacts](images/challenge.tiff?raw=true)



## Run Locally (Windows) 
This code is implemented in Python, and uses several other repositories, which have been cloned already into the repository. These are (1) DJI Thermal SDK, (2) OpenDroneMap, (3) R version 4.2.1  
Other dependencies are installed during step 2 below using Anaconda. 

1. Clone the project, enter directory  

~~~bash  
  git clone https://github.com/rudrakshkapil/Integrated-RGB-Thermal-Orthomosaicing.git IntegratedOrtho
  cd IntegratedOrtho
~~~

2. Install dependencies through anaconda, creating a new environment called 
~~~bash  
  conda env create -f environment.yml
~~~

Note:
The implementation requires PyTorch. 
On a GeForce RTX 3090 we used CUDA version 11.1.
Depending on the GPU available (if any), the CUDA version you need to install may differ. 
Refer to https://pytorch.org/ and https://en.wikipedia.org/wiki/CUDA for resolving version compatibility issues. 


## Citation
If you use this code in your research, please consider citing our paper that proposed the integrated workflow.
~~~tex
...
~~~