Integrated RGB-Thermal Orthomosaicking Pipeline with Intensity-based Registration 
=======================================

This repository contains the official code implementation for the paper ...
Official documentation can be found at https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/

Briefly, it implements an integrated RGB-thermal orthomosaicking workflow that produces a high-quality thermal orthomosaic free from gap and thermal artifacts commonly found in thermal-only workflows. 
The workflow uses intermediate outputs of RGB orthomosaicking to initialize the thermal orthomosaicking process, bypassing the need to perform structure from motion with the thermal images.
For this to be done correctly, the RGB and thermal images are first co-registered using a intensity-based registration that is learned through a gradient descent-based optimization.

A figure summaraizing the advantage of our proposed integrated workflow compared to existing thermal-only workflows is shown below.  

![Summary of integrated workflow showing advantages over thermal-only workflows, specifically the lack of gaps and swirling artifacts](images/challenge.jpg?raw=true)

A more detailed figure of the steps involved in the workflow is shown below. 

![Summary of integrated workflow showing advantages over thermal-only workflows, specifically the lack of gaps and swirling artifacts](/images/pipeline.jpg?raw=true)


The code is usable in scenarios where RGB and thermal images are simultaneously captured by a drone, such as with the DJI H20T multi-sensor instrument and many other commercially available alternatives. 


Official documentation for our code can be found at https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/.




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

3. Download the sample dataset (or place your own) into the current directory. The sample dataset can be found as a zipped file at https://doi.org/10.5281/zenodo.7662405.
  - Refer to [documentation](https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/) for details on correctly organizing your own data.

4. Activate the environment, run the GUI application
~~~bash
  conda activate integrated_rgb_thermal_ortho
  python ./pipeline_tool.py
~~~

5. Within the GUI, specify the name of the directory in the Input panel on the left, i.e. `2022_08_30`. Also specify other settings as needed (refer to [documentation](https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/) for more details).
  - Default settings are provided in `/configs/default.yml` and pre-loaded into the GUI.
  - Current settings are saved into `/configs/combined.yml`.
  - RGB-only and thermal-only default settings are also provided in `/configs`. 

6. Click the `Save Settings` button in the GUI, followed by the `Start Processing` button. 

7. Output orthomsaics will be saved in `<project>/outputs`


Note:
The implementation requires PyTorch with CUDA enabled for GPU acceleration. 
On a GeForce RTX 3090 we used CUDA version 11.1.
Depending on the GPU available (if any), the CUDA version you need to install may differ. 
Refer to https://pytorch.org/ and https://en.wikipedia.org/wiki/CUDA for resolving version compatibility issues. 
CPU-only processing is also an option.


## Citation
If you find this code helpful for your research, please consider citing our paper that proposed the integrated workflow.
`
...
`
