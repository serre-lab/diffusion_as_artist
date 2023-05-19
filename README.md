# Diffusion Models as Artists: Are we closing the Gap between Humans and Machines (Oral at ICML2023)?

Link to the article : https://arxiv.org/abs/2301.11722

## 1. Quick Summary 
<img align="left" src="image/Fig2.png" height="250">
In our article entitled "Diffusion Models as Artist: Are we closing the Gap between Humans and Machines ?", we compare humans and machines on a one-shot image generation task. To do so, we use and extend the "diversity vs. recognizability" framework [1] (see https://arxiv.org/abs/2205.10370), to include 3 types of diffusion model : the DDPM [2], the CFGDM [3] and the FSDM [4]. We use the Omniglot and QuickDraw dataset to compare the models with those presented in [1]. We observe that diffusion models produce sample that are easily recognizable and diverse in the same time (see figure on the Left).  

<br/>  
<br/>   
<br/>  

<img align="right" src="image/Fig3.png" height="250">
To deepen our analysis we introduce the generalization curves: it describes the evolution of the sample's recognizability as a function of their originality (see figure on the Right). We observe that the human generalization cruve is above all others, meaning that humans have the ability to produce recognizable, yet original samples. In addition, guiding the diffusion process using a classifier (as in the case of the CFGDM), tend to improve the sample recognizability at a cost of decreased originality. In contrary, the base diffusion network (the DDPM) suffers from a huge drop in recognizability as the samples become more originals. We confirm this phenomenon by varying the degree of guidance of the CFGDM. We observe that the higher the guidance scale ($\gamma$) the higher the sample's recognizability, but the narower the originality range. With high guidance scale ($\gamma=2$) the diffusion model almost match (locally) the human performancen. However there is still a gap to be filled: models cannot produce original samples that are as recognizable as those produced by humans.

<p align="center">
    <img src="image/Fig4.png" height="280">
  <img src="image/Fig5.png" height="300">
</p>

<br/>  
<br/>   
<br/>  
<br/>  
<br/>   
<br/>  
<br/>  
<br/>   
<br/>  
<br/>  
<br/>   
<br/>  

## 2. Train the one-shot diffusion models 
For the VAE-STN, the DAGAN and the VAS-NS, please refer to the following github repository : https://github.com/serre-lab/diversity_vs_recognizability

Prior to train dataset, any make sure to copy the QuickDraw and Omniglot datasets (available here:).

### DDPM
To train the DDPM on Omniglot, run the following command line
```
python train_guided_diffusion.py --model_name ddpm --dataset_root YOUR_DATA_PATH --dataset omniglot --augment_class --timestep 500 --n_feat 48 --device cuda:6 
```

To run the DDPM, run the following command line
```
python train_guided_diffusion.py --model_name ddpm --dataset_root YOUR_DATA_PATH --dataset quickdraw_clust --augment_class --timestep 500 --n_feat 60 --device cuda:6 
```

#### CFGDM
To train the CFGDM on Omniglot, run the following command line
```
python train_guided_diffusion.py --model_name cfgdm --dataset_root YOUR_DATA_PATH  --dataset omniglot --augment_class --timestep 500 --n_feat 48 --device cuda:6 
```

To train the CFGDM on QuickDraw, run the following command line
```
python train_guided_diffusion.py --model_name cfgdm --dataset_root YOUR_DATA_PATH --dataset quickdraw_clust --augment_class --timestep 500 --n_feat 60 --device cuda:6 
```

## References
[1] : 
[2] :
[3] :
[4] :
