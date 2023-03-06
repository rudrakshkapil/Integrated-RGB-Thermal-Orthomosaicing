Integrated RGB-Thermal Orthomosaicking Pipeline with Intensity-based Registration 
=======================================

This repository contains the official code implementation for the paper ...
Official documentation can be found at https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/

Briefly, it implements an integrated RGB-thermal orthomosaicking workflow that produces a high-quality thermal orthomosaic free from gap and thermal artifacts commonly found in thermal-only workflows. 
The workflow uses intermediate outputs of RGB orthomosaicking to initialize the thermal orthomosaicking process, bypassing the need to perform structure from motion with the thermal images.
For this to be done correctly, the RGB and thermal images are first co-registered using a intensity-based registration that is learned through a gradient descent-based optimization.
The worflow is summarized in the following image. 

![Summary of integrated workflow showing advantages over thermal-only workflows, specifically the lack of gaps and swirling artifacts](images/challenge.jpg?raw=true)

A more detailed figure of the steps involved in the workflow is shown below. 

![Summary of integrated workflow showing advantages over thermal-only workflows, specifically the lack of gaps and swirling artifacts](/images/pipeline.jpg?raw=true)


The code is usable in scenarios where RGB and thermal images are simulataneously captured by a drone, such as with the DJI H20T drone and many other commercially available alternatives. 


Official documentation can be found at https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/.




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

3. Activate the environment, run the GUI application
~~~bash
  conda activate integrated_rgb_thermal_ortho
  python ./pipeline_tool.py
~~~

Note:
The implementation requires PyTorch with CUDA enabled for GPU acceleration. 
On a GeForce RTX 3090 we used CUDA version 11.1.
Depending on the GPU available (if any), the CUDA version you need to install may differ. 
Refer to https://pytorch.org/ and https://en.wikipedia.org/wiki/CUDA for resolving version compatibility issues. 
CPU-only processing is also an option.


## Citation
If you use this code in your research, please consider citing our paper that proposed the integrated workflow.
`
...
`