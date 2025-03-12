# Component-Level OBI Segmentation
Source code for AAAI'25 paper "[Component-Level Segmentation for Oracle Bone Inscription Decipherment](https://drive.google.com/file/d/1dCciWu2y5CTTa0a_LCfuV-l-dwonYwFB/view?usp=drive_link)"

## 0. To Do List
1. -[x] Task Definition
2. -[x] Code
3. -[x] Dataset

## 1. Task Definition
### Background
[Prof. Huang Tianshu](https://www.ctwx.tsinghua.edu.cn/info/1054/2558.htm) pointed out that OBIs that exhibit direct correspondences with modern Chinese characters were basically deciphered by the late 20th century. **_The remaining undeciphered OBIs are exceedingly difficult to decode SOLELY through glyph analysis._**

Take the [case](http://www.fdgwz.org.cn/Web/Show/4472) where [Dr. Jiang Yubin](http://www.fdgwz.org.cn/Web/DetailStaff/44) successfully diciphered the OBI ![Capture](https://github.com/user-attachments/assets/aaab13a8-40b4-4b19-9c20-31e44f7b07a6) (denoted as _A_) for example, the deciphering process roughly includes three steps:
- Step 1: identifying OBIs with similar glyph to _A_ to make an initial semantic hypothesis;
- Step 2: seeking evidence from other OBIs to refine and support this hypothesis;
- Step 3: crossreferencing pre-Qin literature to find corresponding corpus that can further validate the hypothesis from the Step 2.

The final deciphered meaning 蠢（蠢动，骚动） differs significantly from _A_ in terms of glyph structure, indicating that accurate interpretation cannot be achieved through glyph evolution alone.

![Capture](https://github.com/user-attachments/assets/b93077c4-406e-407d-8720-24196acaa87e)

### Why this task?
One of the key contributions of Dr. Jiang Yubin lies in revising the erroneous component segmentation of OBI _B_ from previous study during this step, thereby enabling the textual evidence in Step 3 to perfectly align with the corrected segmentation, as shown in the above figure.
Thus, accurately segmenting OBIs to extract the target components is crucial. We termed this task **Component-Lvel OBI Segmentation**.


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
