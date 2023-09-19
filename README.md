Orthomosaicking Thermal Drone Images of Forests via Simultaneously Acquired RGB Images
=======================================

This repository contains the official code implementation for the paper titled, "Orthomosaicking Thermal Drone Images of Forests via Simultaneously Acquired RGB Images", published as a featured research article in the MDPI Remote Sensing Journal.

The paper can be found [at this link on the MDPI website](https://www.mdpi.com/2072-4292/15/10/2653).<br>
Official documentation can be found at https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/ (under construction... for now, feel free to reach out at rkapil@ualberta.ca for any help with setting up and running the tool, thanks for your patience!)

**Abstract:** Operational forest monitoring often requires fine-detail information in the form of an orthomosaic, created by stitching overlapping nadir images captured by aerial platforms, e.g., drones. RGB drone sensors are commonly used for low-cost, high-resolution imaging that is conducive to effective orthomosaicking, but only capture information in the visible part of the electromagnetic spectrum. Thermal sensors, on the other hand, are able to capture long-wave infrared radiation, making them useful for a variety of applications, e.g., early pest detection. However, these lower-resolution images suffer from reduced contrast and lack of descriptive features for successful orthomosaicking, leading to gaps or swirling artifacts in the orthomosaic. To tackle this issue, we propose an integrated workflow where simultaneously acquired RGB images are first used for producing the surface mesh of the site via structure from motion, while thermal images are only used to texture this mesh and yield a thermal orthomosaic. Prior to texturing, we align the pixel-wise geometry of RGB-thermal image pairs through an automated co-registration technique based on image gradients that leverages machine learning. Empirical results show that the thermal orthomosaic generated from our workflow (1) is of a high quality, (2) is geometrically aligned with the RGB orthomosaic, and (3) preserves radiometric information (i.e., absolute temperature) from the original thermal imagery. Additionally, we highlight the advantage of the resulting geometric alignment by easily accomplishing a sample downstream task -- tree crown detection from the thermal orthomosaic. Our final contribution is an open-source tool with a graphical user interface that implements our workflow to support future works.

The code is usable in scenarios where RGB and thermal images are simultaneously captured by a drone, such as with the DJI H20T multi-sensor instrument and many other commercially available alternatives. 

A figure summarizing the advantage of our proposed integrated workflow compared to existing thermal-only workflows is shown below.
![Summary of integrated workflow showing advantages over thermal-only workflows, specifically the lack of gaps and swirling artifacts](images/challenge.jpg?raw=true)

A more detailed figure of the steps involved in the workflow is shown below. 
![Summary of integrated workflow showing advantages over thermal-only workflows, specifically the lack of gaps and swirling artifacts](/images/pipeline.jpg?raw=true)



## Run Locally (Windows) 
This code is implemented in Python, and uses several other repositories/tools, which have been cloned already into the repository. These are (1) [DJI Thermal SDK](https://www.dji.com/ca/downloads/softwares/dji-thermal-sdk), (2) [OpenDroneMap](https://opendronemap.org/), and (3) [Bidirectional UTM-WGS84 converter for python](https://github.com/Turbo87/utm).  
Other required dependencies are installed during step 2 below using Anaconda. 

1. Clone the project, enter directory  

~~~bash  
  git clone https://github.com/rudrakshkapil/Integrated-RGB-Thermal-Orthomosaicing.git IntegratedOrtho
  cd IntegratedOrtho
~~~

2. Download the required virtual environment for ODM from [this link](https://drive.google.com/drive/folders/1s9TMOsA4KC155mleJuzGay14aj-xPTyD?usp=sharing) and place it into `IntegratedOrtho/ODM`.

3. Install dependencies through anaconda, creating a new environment called 
~~~bash  
  conda env create -f environment.yml
~~~

4. Download the sample dataset (or place your own) into the current directory. The sample dataset can be found as a zipped file at https://doi.org/10.5281/zenodo.7662405.
  - Refer to [documentation](https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/) for details on correctly organizing your own data.

5. Activate the environment, run the GUI application
~~~bash
  conda activate integrated_rgb_thermal_ortho
  python ./pipeline_tool.py
~~~

6. Within the GUI, specify the name of the directory in the Input panel on the left, i.e. `2022_08_30`. Also specify other settings as needed (refer to [documentation](https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/) for more details).
  - Default settings are provided in `/configs/default.yml` and pre-loaded into the GUI.
  - Current settings are saved into `/configs/combined.yml`.
  - RGB-only and thermal-only default settings are also provided in `/configs`. 

7. Click the `Save Settings` button in the GUI, followed by the `Start Processing` button. 

8. Output orthomsaics (RGB and thermal) will be saved in `<project>/output`


Note:
The implementation requires PyTorch with CUDA enabled for GPU acceleration. 
On a GeForce RTX 3090 we used CUDA version 11.1.
Depending on the GPU available (if any), the CUDA version you need to install may differ. 
Refer to https://pytorch.org/ and https://en.wikipedia.org/wiki/CUDA for resolving version compatibility issues. 
CPU-only processing is also an option.


## Citation
If you find this code helpful for your research, please consider citing our paper that proposed the integrated workflow.

**BibTex:**
~~~LaTeX
@Article{Kapil2023,
AUTHOR = {Kapil, Rudraksh and Castilla, Guillermo and Marvasti-Zadeh, Seyed Mojtaba and Goodsman, Devin and Erbilgin, Nadir and Ray, Nilanjan},  
TITLE = {Orthomosaicking Thermal Drone Images of Forests via Simultaneously Acquired RGB Images},  
JOURNAL = {Remote Sensing},  
VOLUME = {15},  
YEAR = {2023},  
NUMBER = {10},  
ARTICLE-NUMBER = {2653},  
URL = {https://www.mdpi.com/2072-4292/15/10/2653},  
ISSN = {2072-4292},  
DOI = {10.3390/rs15102653},  
}
~~~

**AMA Style:**
~~~python
Kapil R, Castilla G, Marvasti-Zadeh SM, Goodsman D, Erbilgin N, Ray N. Orthomosaicking Thermal Drone Images of Forests via Simultaneously Acquired RGB Images. Remote Sensing. 2023; 15(10):2653. https://doi.org/10.3390/rs15102653
~~~
