# MVDE-Net

An official implementation code for paper "A Multi-View Feature Extraction and Dual-Edge Contrastive Learning Approach for Image Forgery Detection"



## Table of Contents
- [Background](#background)
- [Usage](#Usage)
- [Citation](#citation)

## Background
With the rapid development and widespread use of advanced image editing tools such as Adobe Photoshop and Meitu, the creation and dissemination of highly realistic forged images have become increasingly prevalent, posing significant challenges to the authenticity verification of visual content across various fields including journalism, forensic analysis, and social security. Conventional image forgery detection methods predominantly formulate the task as a pixel-wise binary classification problem, which often leads to label ambiguity and conflicts, especially around the edges of tampered regions. Additionally, most existing approaches primarily focus on spatial domain features, neglecting the rich complementary information available from other perspectives such as noise and frequency domains, which can be crucial for forgery detection. 

<p align='center'>
    <img src='./imgs/fig1.png' width='850'/>
</p>

To overcome these limitations, this paper proposes a novel image forgery detection algorithm based on multi-view feature extraction combined with a dual-edge contrastive learning framework. The core idea involves redefining the detection task as an intra-image inconsistency detection problem, thereby effectively avoiding the label conflict issues inherent in traditional pixel classification schemes. To address the semantic ambiguity and blurred boundaries at tampered edges, a dual-edge contrastive learning strategy is designed, which separately extracts and contrasts features from inner and outer edge regions as well as from non-edge tampered and nontampered areas. This approach encourages the model to pay attention to challenging edge samples, thereby improving edge detection accuracy. Furthermore, the proposed method develops a dual-branch multi-view feature encoder to comprehensively capture diverse clues. The spatial domain branch employs a High-Resolution Network (HRNet) backbone to extract multi-scale spatial features, enhanced by a mixture-of-experts gating mechanism that dynamically weights features across scales and fuses residuals between adjacent scales, thus emphasizing subtle forgery traces. The noise domain branch extracts multiple noise-related features, including camera noise fingerprints, SRM filter responses, constrained Bayar convolution outputs, max pooling features, residuals from average pooling, and learnable Fourier domain features with adaptive masking. A mixture-ofexperts strategy is also utilized to assign relevance weights to these heterogeneous features dynamically, according to each input image’s specific characteristics. During training, the fused multi-view features are subjected to the dual-edge contrastive learning framework, which employs a contrastive loss to enhance the discrimination between tampered and authentic regions, especially at their edges. At the inference stage, clustering algorithms such as K-means are applied to the learned feature representations to delineate tampered regions without relying on explicit pixel labels, thus providing a more flexible detection process. 

<p align='center'>
    <img src='./imgs/fig2.png' width='850'/>
</p>

Extensive experiments are conducted on multiple widely used benchmark datasets, including NIST16, Columbia, COVERAGE, CASIA-v1, and DSO, covering various forgery types such as splicing, copy-move, object removal, and post-processing. The proposed method consistently outperforms state-of-the-art approaches, achieving average permuted F1 and IoU score improvements of 2.6% and 7.9%, respectively, over the best existing methods.

<p align='center'>
    <img src='./imgs/fig3.png' width='850'/>
</p>

<p align='center'>
    <img src='./imgs/fig4.png' width='850'/>
</p>

## Usage

- Prepare
```bash
pip install -r requriements.txt
```
We use the pretrained weights of noiseprint++ in Trufor. You can download it from [here](https://github.com/grip-unina/TruFor/blob/main/TruFor_train_test/pretrained_models/noiseprint%2B%2B/noiseprint%2B%2B.th) and put it in the `models/` directory.

- Train
```bash
sh train.sh
```

- Test
```bash
sh test.sh
```
MVDE-Net will detect the images in the `demo/input/` and save results in the `demo/output/` directory.
Also, if you provide corresponding gt masks in the `demo/gt/`, pF1 and pIoU will be printed out.

- Datasets
```bash
sh generate_flist.sh
```

**Note: We have uploaded our model's weight on Google Drive. You can download from [here](https://drive.google.com/drive/folders/1uH485rBVyuvbypEbirczwqLJmCLn-mqo?usp=sharing) and put it in the `weights/` directory.**

## Citation

If you found this code helpful, please citing the reference:
```
@article{MVDE-Net,
    title={A Multi-View Feature Extraction and Dual-Edge Contrastive Learning Approach for Image Forgery Detection},
    author={Z. Xu},
    year={2025}

}
```

This work is based on the code of [FOCAL](https://github.com/HighwayWu/FOCAL). Thanks for their excellent work.