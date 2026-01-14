# HyperAnimal: Identity Hypersphere Guided Synthetic Datasets Generation for Individual Animal Identification
This repository contains source code to reproduce the following paper: HyperAnimal: Identity Hypersphere Guided Synthetic Datasets Generation for Individual Animal Identification.

## Framework
![4-2-paper](https://github.com/user-attachments/assets/df0b01fb-0c1f-4904-aaa6-80f29975a2ef)

---
## 📥 Datasets and Pretrained Models
### Synthetic Animal Datasets
Download the generated synthetic animal datasets (2K identities × 10 images) from the paper:
- [Synthetic dataset for Red panda](https://drive.google.com/drive/folders/1lqPkZpIvAmY_RpX-yRKx1YN2z13jqhL2?usp=sharing)
- [Synthetic datasets for Giant panda](https://drive.google.com/drive/folders/1lqPkZpIvAmY_RpX-yRKx1YN2z13jqhL2?usp=sharing)
- [Synthetic datasets for Amur tiger](https://drive.google.com/drive/folders/1lqPkZpIvAmY_RpX-yRKx1YN2z13jqhL2?usp=sharing)

### Pretrained HyperAnimal Diffusion Models
Download the pretrained HyperAnimal diffusion model weights for different species:
- [Pretrained HyperAnimal for Red panda](https://drive.google.com/drive/folders/199PT_9hB8BZe-_klc1F_0N2sGqk3SOsy?usp=sharing)
- [Pretrained HyperAnimal for Giant panda](https://drive.google.com/drive/folders/199PT_9hB8BZe-_klc1F_0N2sGqk3SOsy?usp=sharing)
- [Pretrained HyperAnimal for Amur tiger](https://drive.google.com/drive/folders/199PT_9hB8BZe-_klc1F_0N2sGqk3SOsy?usp=sharing)

### Pretrained Identification Models
Download the pretrained individual animal identification models trained on synthetic HyperAnimal data:
- [Pretrained Identification Models for Red panda](https://drive.google.com/drive/folders/11Qh4jIZYmq4gKqpWRgvTwUjqgL8E6o-U?usp=sharing)
- [Pretrained Identification Models for Giant panda](https://drive.google.com/drive/folders/11Qh4jIZYmq4gKqpWRgvTwUjqgL8E6o-U?usp=sharing)
- [Pretrained Identification Models for Amur tiger](https://drive.google.com/drive/folders/11Qh4jIZYmq4gKqpWRgvTwUjqgL8E6o-U?usp=sharing)

---
## 🚀 Quick Start
### Environment Setup
- Clone the repository

    ```bash
    git clone <repository-url>
    cd HyperAnimal
    ```

- Create and activate the conda environment

    ```bash
    conda env create -n hyperanimal -f environment.yml
    conda activate hyperanimal
    ```

### Data Preparation (Red Panda For Example)
- Place unlabeled real red panda images in `data/redpanda/`
- The pre-extracted training embeddings are provided in `data/redpanda_embeddings/`

### Required Pretrained Models
- Identity embeddings extractor:
  - Downloaded from [identity embeddings](https://drive.google.com/drive/folders/1_MQI72nr_lVCJa5LuSHyo8iWRf58dYJy?usp=sharing)
  - Save to `models/identification/weights/rpd43.pth` 

- Autoencoder Weights:
  - Download pre-trained encoder/decoder weights from [Encoder and decoder model weights](https://drive.google.com/drive/folders/1_2hTh0Bi0NekxtHaSfY9eFtfavclfgwQ?usp=sharing)
  - Save to `models/autoencoder/vq_f8_encoder.pt` and `models/autoencoder/vq_f8_decoder.pt`
  - *Note: The pre-trained autoencoder weights that originally come from the `fhq256` LDM from [Rombach et al.](https://github.com/CompVis/latent-diffusion/blob/main/models/ldm/ffhq256/config.yaml). Their VQModelInterface submodule has been manually extracted and split into its encoder and decoder models, since the encoder is only used during training and the decoder is only needed for sampling.*

---
## 📊 Usage Guide
### Training the HyperAnimal Model
- Make sure that the `dataset: redpanda_FDIE` option is set and that the paths in the corresponding subconfiguration `configs/dataset/redpanda_FDIE.yaml` are pointing to the training images and pre-extracted embeddings.  
- Start training:

    ```bash
    python main.py
    ```

    Trained models will be saved in `outputs/rp_f1/checkpoints/`

### Sampling with a Trained HyperAnimal Model
- Download the pretrained HyperAnimal models (including the `.hydra` folder) and place them in `outputs/rp_f1/checkpoints/`.
- Generate synthetic identity contexts, and save them in `data/contexts/syn_2000.npy`

    ```bash
    python create_sample_identity_contexts.py
    ```

- Configure sampling parameters in `configs/sample_rp.yaml`, including
  - Path to trained model
  - Path to contexts file
  - Number of identities
  - Images per identity

- Generate samples:

    ```bash
    python sample.py
    ```

    Those samples will be saved under `samples/` as identity blocks, e.g. a 4x4 grid block of 512x512 images.

- Split identity blocks into individual images:

    ```bash
    python split_identity_blocks.py
    ```
  
    Generated samples are saved in `samples/`

### Training Individual Animal Identification Models
- With the code provided under `reid/`, the training and testing of six identification models should be started via:

    ```bash
    # Prepare data splits
    python prepare_gallery_query.py
    python prepare_train_val.py
        
    # Train models
    ./train.sh
        
    # Evaluate models
    ./test.sh
    ```

***Important!!! Testing the values in Table 3/9/10***
- Download the pretrained six identification models from [identification models](https://drive.google.com/drive/folders/11Qh4jIZYmq4gKqpWRgvTwUjqgL8E6o-U?usp=sharing), and place them in
     - ./reid/model/rp
     - ./reid/model/gp
     - ./reid/model/atrw

- Download the test data from [Test data](https://drive.google.com/drive/folders/1KA-W50bNshT8s9gOy0SjNR2zNytKgPhA?usp=sharing), and place them in
   - ./reid/test_data/redpanda-test
   - ./reid/test_data/iPanda-test
   - ./reid/test_data/atrw-test

- Execute the `test_tb3.sh` script in reid files 

    ```bash
    cd HyperAnimal/reid
    bash test_tb3.sh
    ```
    
    Results (mAP and CMC values) will be saved to `result.txt`. These values correspond to Table 3, 9, and 10 in the paper. 
    Specifically, the results can also be found in `result.txt` file from [pretrained identification models](https://drive.google.com/drive/folders/11Qh4jIZYmq4gKqpWRgvTwUjqgL8E6o-U?usp=sharing).

---
## 📁 Repository Structure

```
HyperAnimal/
├── configs/                            # Configuration YAML files
├── data/                               # Training images and embeddings 
|   ├── embeddings/                     # Identity embeddings for training
|   ├── contexts/                       # Synthetic identity embeddings for sampling 
|   ├── redpanda/                       # Redpanda dataset for training
├── models/                             # PyTorch model architectures
|   ├── autoencoder/                    # Autoencoder models
|   ├── diffusion/                      # DDPM implementation
|   ├── identification/                 # Identification model weights
├── outputs/                            # Trained model checkpoints
├── samples/                            # Generated samples
├── utils/                              # Utility modules and scripts
├── reid/                               # Animal identification training code
├── main.py                             # Main training script
├── sample.py                           # Sampling script
├── create_sample_identity_contexts.py  # Context generation
├── split_identity_blocks.py            # Sample processing
├── extract_identity_embeddings.py      # Embedding extraction
└── environment.yml                     # Conda environment specification
```
