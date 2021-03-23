# [Local Group Bias Detection](https://drive.google.com/file/d/19QIaHnWqGzo-WWRAmxXG7H1Z7kYY5MfC/view?usp=sharing)

This is our github project repository for UCLA CS 263 - Natural Language Processing course taught by Prof. Kai Wei Chang.

Authors: Anubhav Mittal, Emma Peterson, Eddie Huang

## Abstract

Machine learning systems have achieved widespread success across a plethora of domains but have recently been shown to inherit societal biases in data. As these systems are increasingly being used in important decision making systems, they can propagate and even amplify these biases. In this work, we explore different ways to identify, evaluate, and mitigate the effects of unintended bias in datasets and word embeddings. We examine the differences in model performance among different
identity groups present in the dataset (like race or gender). We start by establishing a framework for quantifying bias which distinguishes between global and local group bias. We move on to implement a recently published bias detection algorithm called [LOGAN](https://arxiv.org/abs/2010.02867), which highlights the extent of this local group bias, and use it to detect local biases in models trained on two popular classification datasets. Finally, we review the effectiveness of different bias reduction techniques in mitigating local group bias and try to analyze their behaviour.


## Using our code

Both toxicity classification and MS-COCO classification have been implemented using Pytorch and Scikit-Learn. In addition, we use the [hugging face transformer library](https://github.com/huggingface/transformers "hugging face transformers") for getting the pretrained 'BERT-base-cased' model.

Links for downloading the datasets, trained vanilla and debiased word vectors, saved model checkpoints and predictions are present in [Files.md](https://github.com/anubhavm11/Local-biases-in-models/blob/master/Files.md) .

### Toxicity Classification

#### Main paper results

For training a BERT model on the toxicity classification dataset, place the data files in the `data_dir` folder and run the following:

    python "./Toxicity classification/train.py" --train --data_dir "./data/"
Or if you only want to get predictions for the test set, then also place the model checkpoint in `save_dir` and run the following:

    python "./Toxicity classification/train.py" --data_dir "./data/" --save_dir "./saved_models/"

#### Performance of different pre-trained word vectors (word2vec, hard-debiased word2vec, GloVe, GN-GloVe) 

For training a BiLSTM classifier on the toxicity classification dataset using a pre-trained word embedding, place the data and word vector files in the `data_dir` folder (named appropriately), and for the chosen `word_vector` (can be one of `word2vec`, `hd-word2vec`, `glove`, `gn-glove`), run the following:

    python "./Toxicity classification/train_lstm.py" --train --data_dir "./data/" --word_vector "glove"
Or if you only want to get predictions for the test set, then also place the model checkpoint in `save_dir` and run the following:

    python "./Toxicity classification/train_lstm.py" --data_dir "./data/" --save_dir "./saved_models/" --word_vector "glove"

### MS-COCO Object Classification

For training a classifier on top of a ResNet-50 for the MS-COCO object classification dataset, place the image in the `image_dir` folder, annotation files (provided [here](https://github.com/anubhavm11/Local-biases-in-models/tree/master/MS-COCO/data) ) in the `annotation_dir` folder, and run the following:

    python "./MS-COCO/train.py" --train --image_dir "./data/" --annotation_dir "./data/" --num_epochs 100
Or if you only want to get predictions for the test set, then also place the model checkpoint in `save_dir` and run the following:

    python "./MS-COCO/train.py" --image_dir "./data/" --annotation_dir "./data/" --save_dir "./saved_models/"--num_epochs 100

### K-means and LOGAN Clustering
For this part, you first have to copy the contents of the `./LOGAN/cluster`directory to the `cluster` directory of your `scikit-learn` installation (described in more detail [here](https://github.com/uclanlp/clusters#about-our-code)). 

#### Toxicity Classification
You have to place the test prediction and embeddings file in `data_dir` , specify the hyperparameter `lamb`and `demographic`(can be either `race` or  `gender`) and run the following:

    python "./LOGAN/logan.py" --data_dir "./data/" --test_pred "pred_test_bw_withpredscores.csv" --test_embed "second2last_mean.npy" --demographic "gender" --lamb 5  

#### MS-COCO Object Classification
You have to place the file containing both the embeddings and predictions in `data_dir` , specify the hyperparameter `lamb`and if you want to save the LOGAN cluster labels (using the `save_clusters` parameter) and run the following:

    python "./LOGAN/logan_ms_coco.py" --data_dir "./data/" --test_pred "val.data" --lamb 5 --save_clusters  
