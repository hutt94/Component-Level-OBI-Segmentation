# Component-Level OBI Segmentation
Source code for AAAI'25 paper "[Component-Level Segmentation for Oracle Bone Inscription Decipherment]()"

## 0. To Do List
1. -[ ] Task Definition
2. -[x] Code
3. -[x] Dataset

## 1. Task Definition

## 2. Code
### Preparation
Installing the MMlab as follow:
```
pip install mmcv-full==1.4.7 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmsegmentation==0.24.0
```
You can download other mmlab version for your cuda and torch via 
```
https://download.openmmlab.com/mmcv/dist/{your_cuda_version}/{your_torch_version}/index.html
```
Then, you need download [pvt-medium](https://github.com/whai362/PVT) as backbone.

### Train
```
python train.py --data_path your_data_path
```
### Test
```
python test.py --data_path your_data_path
```
### Visualization
```
python visual.py --data_path your_data_path
```

## 3. Dataset
Please refer to [Component-Level Oracle Bone Inscription Retrieval](https://github.com/hutt94/Component-Level_OBI_Retrieval/tree/main) to apply for the dataset and [here](https://github.com/hutt94/Component-Level_OBI_Retrieval/tree/main/OBI_Component_20) for more details about the dataset.

## 4. Citation
```
@inproceedings{hu2025component,
  title={Component-Level Segmentation for Oracle Bone Inscription Decipherment},
  author={Hu, Zhikai and Cheung, Yiu-ming and Zhang, Yonggang and Zhang, Peiying and Tang, Pui-ling},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={},
  number={},
  pages={},
  year={2025}
}
@inproceedings{hu2024component,
  title={Component-Level Oracle Bone Inscription Retrieval},
  author={Hu, Zhikai and Cheung, Yiu-ming and Zhang, Yonggang and Zhang, Peiying and Tang, Pui-ling},
  booktitle={Proceedings of the 2024 International Conference on Multimedia Retrieval},
  pages={647--656},
  year={2024}
}
```

## Acknowledgments
We would like to thank [小學堂](https://xiaoxue.iis.sinica.edu.tw/) for sharing the public OBI data. We are also grateful to [Mr. Changxing Li](https://github.com/li1changxing) for his assistance with the data collection and code implementation.
