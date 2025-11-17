---
title: Convolutional Neural Networks for actual Computer Vision
theme: evo
highlightTheme: vs
separator: <!--s-->
verticalSeparator: <!--v-->
revealOptions:
    transition: 'fade'
    transitionSpeed: 'default'
    controls: true
    slideNumber: true
    width: '100%'
    height: '100%'
---

# Deep Learning for Computer Vision
## from image classification to actual computer vision

**ISAE-SUPAERO, SDD, 18 Nov. 2025**

Florient CHOUTEAU

<!--v-->

Slides : https://fchouteau.github.io/isae-intro-to-cnns

Notebooks : https://github.com/SupaeroDataScience/deep-learning/tree/main/vision

<!--s-->

# Use cases of CNNs

<!--v-->

We've done image classification that we applied in a sliding window fashion on larger images

![sliding](static/img/sliding_window.gif)

<!--v-->

We can solve other types of tasks with ConvNets

![tasks](static/img/computervision_tasks.png)  <!-- .element: style="width: 60%; height: 40%"--> 

<!--v-->

Keypoint Detection

![keypoint](static/img/yolov7-mediapipe-human-pose-detection-feature-1.gif)  <!-- .element: style="width: 60%; height: 40%"-->

<!--v-->

3D (VGGT)

<video data-autoplay src="https://vgg-t.github.io/resources/teaser_video_v3_compressed_short.mp4"></video> <!-- .element: style="width: 60%; height: 40%"-->

<!--v-->

Image/Video Restoration-Upsampling

![denoising](static/img/denoising.png)  <!-- .element: style="width: 60%; height: 40%"-->

<!--v-->

Image/Video Restoration-Upsampling

![DLSS](static/img/maxresdefault.jpg) <!-- .element: style="width: 60%; height: 40%"-->

<!--v-->

And of course image and video generation

<video data-autoplay src="https://openaiassets.blob.core.windows.net/$web/nf2/blog-final/golden/6eda9a57-5d6d-4890-90ee-61f89e999719/20250922_1557_an%20astronaut%20golden%20retriever%20named%20_Sora_%20levitates%20around%20an%20intergalactic%20pup-themed%20space%20statio_simple_compose_01k5sta9hmegr9axrpbty5sxva%20(1).mp4"></video>

<!--s-->

# How ?

It's just about architectures and loss functions

<!--v-->

Image Segmentation

![segmentation](static/img/encoderdecoder2.png)   <!-- .element: style="width: 60%; height: 40%"-->

<!--v-->

Object Detection (intuition)

![objdet](static/img/1_REPHY47zAyzgbNKC6zlvBQ.png)  <!-- .element: style="width: 60%; height: 40%"-->

<!--v-->

Object Detection (in practice)

![objectdetection](https://www.labellerr.com/blog/content/images/2023/01/yolo-algorithm-1.webp)   <!-- .element: style="width: 60%; height: 40%"-->

<!--v-->

Instance Segmentation

![instseg](https://cdn.prod.website-files.com/680a070c3b99253410dd3df5/68ee302b0bd7d85155faad5d_684d854d2575826aacdbf29e_67ed516491886a7a596e2fd7_67dd6dc00c8345cf27922d88_rcnn_fig2.webp)  <!-- .element: style="width: 60%; height: 40%"-->

<!--v-->

To learn more about this, see [this cs231n class](http://cs231n.stanford.edu/slides/2022/lecture_9_jiajun.pdf)

<!--s-->

# Encoder Decoder and U-nets

<!--v-->

How to solve a dense task with CNNs ? The problem : Context

![fcnn](static/img/fcnn.png) <!-- .element: style="width: 60%; height: 40%"-->

<!--v-->

The Encoder Decoder Concept

![encoder-decoder](https://towardsdatascience.com/wp-content/uploads/2022/11/1XeUglwXyh7967mlMOF20Zw.png)

<!--v-->

![fcnn](static/img/fcnn2.png)) <!-- .element: style="width: 60%; height: 40%"-->

<!--v-->

How to upsample ? Static upsampling

![upsampling](https://mriquestions.com/uploads/3/4/5/7/34572113/uppooling-methods_orig.png) <!-- .element: style="width: 50%; height: 50%"-->

<!--v-->

How to upsample ? Learnable upsampling

![upsampling](https://towardsdatascience.com/wp-content/uploads/2022/06/1kv5m8-VXHZ5RzHu70Jt_BQ.png) <!-- .element: style="width: 50%; height: 50%"-->

<!--v-->

How to upsample ? Learnable upsampling

![tconv](https://i.redd.it/oooaoghhfaj91.png) <!-- .element: style="width: 50%; height: 50%"-->

<!--v-->

Is it enough ? How to keep spatial information ?

![unet](https://towardsdatascience.com/wp-content/uploads/2022/11/1LH_JiIJngSllUZ0F8JYcwQ.png) <!-- .element: style="width: 50%; height: 50%"-->

<!--v-->

They are called [U-Net (2015)](https://arxiv.org/abs/1505.04597)s

![unet](static/img/unet.png) <!-- .element: style="width: 50%; height: 50%"-->

<!--v-->

Imagen & co are based on unets

![unet](static/img/imagen.png)


<!--s-->

# Hands-on 2 : From image classification to image segmentation

<!--v-->

### Dataset description

- 12800 train images (6400 cloudy, 6400 clear), size 64x64
- 640 test images (320 cloudy, 320 clear), size 64x64

![toy_dataset](static/img/cloud_samples_segmentation.png) <!-- .element height="40%" width="40%" -->

<!--v-->

### Dataset description

- 64 big test images (256x256)

![toy_dataset](static/img/cloud_tiles.png) <!-- .element height="40%" width="40%" -->

<!--v-->

### Let's go ! 

- 3_classification_to_segmentation.ipynb : From image classification to dense tasks
- 4_sliding_window_unet.ipynb : How to apply our detector to a big satellite image ?
- 
<!--s-->

Extra : Artificial Intelligence in practice in a space systems context

[ISAE SPAPS 2023-2024 class](https://docs.google.com/presentation/d/10zd65eg0X_aqydggKRvC3s20AXSu3WI9RWWUKJVvfAI/edit?usp=sharing)

<!--s-->

Extra : Other keywords that are important for CNN in Computer Vision

- [Self-Supervised Learning](http://cs231n.stanford.edu/slides/2023/lecture_13.pdf)
- [Vision Transformers](https://www.v7labs.com/blog/vision-transformer-guide)
- [Generative Adversarial Networks](https://github.com/SupaeroDataScience/deep-learning/tree/main/GAN)
- [Diffusion Models](http://cs231n.stanford.edu/slides/2023/lecture_15.pdf) (StableDiffusion, MidJourney ...)
