Code for Hierarchical Multi-modal Fusion Tracker in CVPR2022 paper [Visible-Thermal UAV Tracking: A Large-Scale Benchmark and New Baseline](https://arxiv.org/pdf/2204.04120.pdf), which is a strong baseline for RGB-T tracking.

## Framework
![alt text](https://github.com/zhang-pengyu/HMFT/blob/master/framework.png)
*Three complementary modules are introduced for multi-modal fusion*
 * **Complementary Image Fusion(CIF)**: CIF aims to use a shared backbone to extract complementary information and Kullback–Leibler divergence loss is introduced to unify the feature distribution.
 * **Discriminative Feature Fusion(DFF)**: DFF aims to build individual representations for both modalities and learns a channel-wise modality weight to fuse them.
 * **Adaptive Decision Fusion(ADF)**: ADF is to adaptively provide the final response by considering the results of two branches and the modality confidence. 
## Model Zoo
The pretrained model are available at [[GoogleDrive]](https://drive.google.com/file/d/1vnof9qMFsfwmn8xk-UKaFhHTYM1F85j2/view?usp=sharing) and [[BaiduDisk]](https://pan.baidu.com/s/1561M-cvx5wUXm_AXRVDiVw?pwd=cegf)

## Get Started
### Set up Anaconda environment
```
conda create -n HMFT python=3.6
conda activate HMFT
cd $Path_to_HMFT$
bash install.sh
```
### Run demo sequence
```
cd $Path_to_HMFT$/mfDiMP/pytracking
python demo.py
```
### Test on current benchmarks
```
cd $Path_to_HMFT$/mfDiMP/pytracking
python run_VTUAV.py
python run_GTOT.py
python run_RGBT210.py
python run_RGBT234.py
```
## Training

## Results
![alt text](https://github.com/zhang-pengyu/HMFT/blob/master/results.png)
The results can be found at [[GoogleDrive]](https://drive.google.com/file/d/1IKWNaKscdw8A5vZlzN5Iyyh_liFhtnHZ/view?usp=sharing) and [[BaiduDisk]](https://pan.baidu.com/s/1-g_LbPuwAGCl0DvwwXMesg?pwd=ot32)
## Citation
If you find our work useful, please cite

@InProceedings{Zhang_CVPR22_VTUAV,
author = {Zhang Pengyu and Jie Zhao and Dong Wang and Huchuan Lu and Xiang Ruan},
title = {Visible-Thermal UAV Tracking: A Large-Scale Benchmark and New Baseline},
booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
year = {2022}
}
