# [Local Group Bias Detection](https://drive.google.com/file/d/19QIaHnWqGzo-WWRAmxXG7H1Z7kYY5MfC/view?usp=sharing)

This is our github project repository for UCLA CS 263 - Natural Language Processing course taught by Prof. Kai Wei Chang.

Authors: Anubhav Mittal, Emma Peterson, Eddie Huang

## Abstract

Machine learning systems have achieved widespread success across a plethora of domains but have recently been shown to inherit societal biases in data. As these systems are increasingly being used in important decision making systems, they can propagate and even amplify these biases. In this work, we explore different ways to identify, evaluate, and mitigate the effects of unintended bias in datasets and word embeddings. We examine the differences in model performance among different
identity groups present in the dataset (like race or gender). We start by establishing a framework for quantifying bias which distinguishes between global and local group bias. We move on to implement a recently published bias detection algorithm called [LOGAN](https://arxiv.org/abs/2010.02867) , which highlights the extent of this local group bias, and use it to detect local biases in models trained on two popular classification datasets. Finally, we review the effectiveness of different bias reduction techniques in mitigating local group bias and try to analyze their behaviour.


## Using our code

Both toxicity classification and MS-COCO classification using Pytorch. In addition, we use the [hugging face transformer library](https://github.com/huggingface/transformers "hugging face transformers") for getting the pretrained 'BERT-base-cased' model.