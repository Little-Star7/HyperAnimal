# HyperAnimal: Identity Hypersphere Guided Synthetic Datasets Generation for Individual Animal Identification
This repository contains source code to reproduce the following paper: HyperAnimal: Identity Hypersphere Guided Synthetic Datasets Generation for Individual Animal Identification.

# Framework
![4-2-paper](https://github.com/user-attachments/assets/df0b01fb-0c1f-4904-aaa6-80f29975a2ef)

# Datasets and pretrained models
📥 Datasets and Pretrained Models
Download links for the pretrained HyperAnimal diffusion model weights:
- [Pretrained HyperAnimal for Red panda](https://drive.google.com/drive/folders/199PT_9hB8BZe-_klc1F_0N2sGqk3SOsy?usp=sharing)
- [Pretrained HyperAnimal for Giant panda](https://drive.google.com/drive/folders/199PT_9hB8BZe-_klc1F_0N2sGqk3SOsy?usp=sharing)
- [Pretrained HyperAnimal for Amur tiger](https://drive.google.com/drive/folders/199PT_9hB8BZe-_klc1F_0N2sGqk3SOsy?usp=sharing)

Download links for the generated synthetic 2K identities x 10 images datasets from the paper:
- [Synthetic dataset for Red panda](https://drive.google.com/drive/folders/1lqPkZpIvAmY_RpX-yRKx1YN2z13jqhL2?usp=sharing)
- [Synthetic datasets for Giant panda](https://drive.google.com/drive/folders/1lqPkZpIvAmY_RpX-yRKx1YN2z13jqhL2?usp=sharing)
- [Synthetic datasets for Amur tiger](https://drive.google.com/drive/folders/1lqPkZpIvAmY_RpX-yRKx1YN2z13jqhL2?usp=sharing)

Download links for the pretrained individual animal identification models using synthetic HyperAnimal generated data:
- [Pretrained Identification Models for Red panda](https://drive.google.com/drive/folders/11Qh4jIZYmq4gKqpWRgvTwUjqgL8E6o-U?usp=sharing)
- [Pretrained Identification Models for Giant panda](https://drive.google.com/drive/folders/11Qh4jIZYmq4gKqpWRgvTwUjqgL8E6o-U?usp=sharing)
- [Pretrained Identification Models for Amur tiger](https://drive.google.com/drive/folders/11Qh4jIZYmq4gKqpWRgvTwUjqgL8E6o-U?usp=sharing)


# How to use the code? 
It includes the main scripts used for training and evaluating the HyperAnimal models. Below we take the Red Panda dataset as an example.

# Setup 
Create the conda environment using the provided environment.yml file:

cd HyperAnimal
conda env create -n hyperanimal -f environment.yml

Put the unlabelled real redpanda images under `data/redpanda/`. The training embeddings used as identity contexts during training are provided under `data/redpanda_embeddings` and can be extracted using the `extract_identity_embeddings.py` script. 

For that, the pre-trained identity extractor weights have to be downloaded from the [identity embeddings](https://drive.google.com/drive/folders/1_MQI72nr_lVCJa5LuSHyo8iWRf58dYJy?usp=sharing) and placed under `models/identification/weights/rpd43.pth`. 

The pre-trained autoencoder weights that originally come from the `fhq256` LDM from [Rombach et al.](https://github.com/CompVis/latent-diffusion/blob/main/models/ldm/ffhq256/config.yaml). Their VQModelInterface submodule has been manually extracted and split into its encoder and decoder models, since the encoder is only used during training and the decoder is only needed for sampling:
- [Encoder and decoder model weights](https://drive.google.com/drive/folders/1_2hTh0Bi0NekxtHaSfY9eFtfavclfgwQ?usp=sharing)
The resulting .pt files are then expected to be saved under `models/autoencoder/vq_f8_encoder.pt` and `models/autoencoder/vq_f8_decoder.pt`, respectively.

---
# Training HyperAnimal model
Make sure that the `dataset: redpanda_FDIE` option is set and that the paths in the corresponding subconfiguration `configs/dataset/redpanda.yaml` are pointing to the training images and pre-extracted embeddings. The model training can be initiated by executing:
    
    python main.py
     
After the model is trained, the model output directory content is under `outputs/redpanda/`.

---
# Sampling with a trained HyperAnimal model
Download the pretrained HyperAnimal models including ".hydra" folder and place it inside `outputs/redpanda/`.  

For reproducibility and consistency, the synthetic contexts are NOT generated on-the-fly during sampling. Instead, they are pre-generated and saved in `.npy` files.  Execute the `create_sample_identity_contexts.py` script, which will pre-compute synthetic uniform contexts that you can use for sampling. Then, specify the path to the trained model and the contexts file that shall be used for sampling in the `configs/sample_rp.yaml`. There you can also configure the number of identities to use from the provided contexts file and the number of images per identity context. The sampling script can be started via:
    
    python create_sample_identity_contexts.py
    python sample.py
     
Those samples will be saved under `samples/` as identity blocks, e.g. a 4x4 grid block of 512x512 images. These blocks can then be splitted using e.g. then `split_identity_blocks.py` script.    
    
    python split_identity_blocks.py
    
---
# Training individual animal identification models
With the code provided under `reid/`, the training and testing of six identification models should be started via:

    python prepare_gallery_query.py
    python prepare_train_val.py
    ./train.sh
    ./test.sh

***Important!!! Testing the values in Table 3/9/10***
Download the well-trained six identification models from the [identification models](https://drive.google.com/drive/folders/11Qh4jIZYmq4gKqpWRgvTwUjqgL8E6o-U?usp=sharing), and place them in ./reid/model/rp or ./reid/model/gp or ./reid/model/atrw.

Download the test data from the [Test data](https://drive.google.com/drive/folders/1KA-W50bNshT8s9gOy0SjNR2zNytKgPhA?usp=sharing), and place them in ./reid/test_data/redpanda-test or ./reid/test_data/iPanda-test or ./reid/test_data/atrw-test.

Execute the `test_tb3.sh` script in reid files, which will compute the mAP and CMC values provided in Table 3/9/10 of our paper. 

    cd HyperAnimal/reid
    bash test_tb3.sh

Specifically, the results can also be found in result.txt file in the [pretrained identification models](https://drive.google.com/drive/folders/11Qh4jIZYmq4gKqpWRgvTwUjqgL8E6o-U?usp=sharing).

# More information on remaining folders and scripts:
### Directories:
- `configs/` contains the configuration .yaml files
- `data/` contains the training images and training embeddings
- `diffusion/` contains the DDPM code
- `models/` contains the PyTorch modules and model structures
- `outputs/` contains pre-trained models
- `samples/` will contain the generated samples, their extracted features and the contexts used for sampling
- `utils/` contains utility modules, models and scripts
- `reid/` contains code that was used to train animal identifictaion models

### Main scripts:
- `main.py` contains the training script
- `sample.py` contains the sampling script
- `create_sample_identity_contexts.py` contains code for identity-context generation
- `split_identity_blocks.py` samples are saved as concatenated blocks per identity (can easily be modified),
              and this script can be used to split them to create identity-class folders for identification training
- `extract_identity_embeddings.py` extracts identity embeddings from the pre-trained FDIE extractor or ResNet50 identification model



