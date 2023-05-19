# Diffusion Models as Artists: Are we closing the Gap between Humans and Machines (Oral at ICML2023)?

Link to the article : https://arxiv.org/abs/2301.11722

## 1. Quick Summary 
In our article entitled "Diffusion Models as Artist: Are we closing the Gap between Humans and Machines ?", we compare humans and machines on a one-shot image generation task. To do so, we use and extend the "diversity vs. recognizability" framework (see https://arxiv.org/abs/2205.10370), to include 3 types of diffusion model : the DDPM [1], the CFGDM [2] and the FSDM [3]. 
<img align="right" src="image/Fig2.png" height="250">

Diversity vs. Recognizability: Human-like generalization in one-shot generative models", we introduce a new framework to assess one-shot generative model. The evaluation framework is based on 1) the diversity of the generated samples and 2) their recognizability.The higher the diversity and the recognizability the better the generalization ability of the generative model.



<img align="right" src="image/Fig3.png" height="250">

<img align="right" src="image/Fig4.png" height="250">

<img align="right" src="image/Fig5.png" height="250">



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
