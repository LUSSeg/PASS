# PASS method for LUSS task （Jittor version）
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/large-scale-unsupervised-semantic/unsupervised-semantic-segmentation-on-4)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on-4?p=large-scale-unsupervised-semantic)

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/large-scale-unsupervised-semantic/unsupervised-semantic-segmentation-on-5)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on-5?p=large-scale-unsupervised-semantic)

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/large-scale-unsupervised-semantic/unsupervised-semantic-segmentation-on-6)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on-6?p=large-scale-unsupervised-semantic)



## Introduction
![image](https://user-images.githubusercontent.com/20515144/196449430-5ac6a88c-24ea-4a82-8a45-cd244aeb0b3b.png)

We propose a new method for LUSS, namely PASS, containing four steps. 1) A randomly initialized model is trained with self-supervision of pretext tasks (i.e. our proposed Non-contrastive pixel-to-pixel representation alignment and Deep-to-shallow supervision) to learn shape and category representations. After representation learning, we obtain the features set for all training images. 2) We then apply a pixel-attention-based clustering scheme to obtain pseudo categories and assign generated categories to each image pixel. 3) We fine-tune the pre-trained model with the generated pseudo labels to improve the segmentation quality. 4) During inference, the LUSS model assigns generated labels to each pixel of images, same to the supervised model. 

More details about the LUSS task and [ImageNet-S dataset](https://github.com/LUSSeg/ImageNet-S) are in [project page](https://LUSSeg.github.io/) and [paper link](https://arxiv.org/abs/2106.03149).



## Usage
We give the training and inference details in **[USAGE](USAGE.md)**.
## Results
**Fully Unsupervised Evaluation Protocol**
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Arch</th>
<th valign="bottom">Val</th>
<th valign="bottom">Test</th>
<th valign="bottom">Args</th>
<th valign="bottom">Pretrained</th>
<th valign="bottom">Pixelatt</th>
<th valign="bottom">Centroid</th>
<th valign="bottom">Finetuned</th>
<!-- TABLE BODY -->
<tr>
<td align="left">ImageNet-S</td>
<td align="center">ResNet50</td>
<td align="center">11.4</td>
<td align="center">10.3</td>
<td align="center"><a href="scripts/luss919_pass_jt.sh">bash</a></td>
<td align="center"><a href="">model</a></td>
<td align="center"><a href="">model</a></td>
<td align="center"><a href="">centroid</a></td>
<td align="center"><a href="">model</a></td>
</tr>
<td align="left">ImageNet-S 300</td>
<td align="center">ResNet50</td>
<td align="center">17.8</td>
<td align="center">17.4</td>
<td align="center"><a href="scripts/luss300_pass_jt.sh">bash</a></td>
<td align="center"><a href="">model</a></td>
<td align="center"><a href="">model</a></td>
<td align="center"><a href="">centroid</a></td>
<td align="center"><a href="">model</a></td>
</tr>
</tr>
<td align="left">ImageNet-S 50</td>
<td align="center">ResNet50</td>
<td align="center">29.2</td>
<td align="center">29.8</td>
<td align="center"><a href="scripts/luss50_pass_jt.sh">bash</a></td>
<td align="center"><a href="">model</a></td>
<td align="center"><a href="">model</a></td>
<td align="center"><a href="">centroid</a></td>
<td align="center"><a href="">model</a></td>
</tr>
</tbody></table>

## Citation
```
@article{gao2022luss,
  title={Large-scale Unsupervised Semantic Segmentation},
  author={Gao, Shanghua and Li, Zhong-Yu and Yang, Ming-Hsuan and Cheng, Ming-Ming and Han, Junwei and Torr, Philip},
  journal=TPAMI,
  year={2022}
}
```

## Acknowledgement

This codebase is build based on the [SwAV codebase](https://github.com/facebookresearch/swav).

If you have any other question, open an issue or email us via shgao@live.com


