# 第三届计图人工智能挑战赛：大规模无监督语义分割赛题（赛题二）baseline


## 简介
![image](https://user-images.githubusercontent.com/20515144/196449430-5ac6a88c-24ea-4a82-8a45-cd244aeb0b3b.png)


PASS是一种新的大规模无监督语义分割方法，包含四个步骤。1） 通过对借口任务的自我监督（即非对比像素到像素的表示对齐和深度到浅层监督）来训练随机初始化的模型，以学习形状和类别表示。在表示学习之后，获得了所有训练图像的特征集。2） 然后，应用基于像素注意力的聚类方案来获得伪类别，并将生成的类别分配给每个图像像素。3） 用生成的伪标签对预训练的模型进行微调，以提高分割质量。4） 在推理过程中，LUSS模型将生成的标签分配给图像的每个像素，与监督模型相同。


有关LUSS任务和[ImageNet-S dataset](https://github.com/LUSSeg/ImageNet-S)的更多详细信息在[项目页面](https://LUSSeg.github.io/)和[文章](https://arxiv.org/abs/2106.03149)。


## 用法
我们在[用法文档](USAGE_ZH.md)中给出了训练和推理的详细信息


## 引用
```
@article{gao2022luss,
  title={Large-scale Unsupervised Semantic Segmentation},
  author={Gao, Shanghua and Li, Zhong-Yu and Yang, Ming-Hsuan and Cheng, Ming-Ming and Han, Junwei and Torr, Philip},
  journal=TPAMI,
  year={2022}
}
```


