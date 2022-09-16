# Neuro myKakuritsu Research Code with PyTorch

This repo is a fork of https://github.com/UEFI-code/Neuro_myKakuritsu, but NO LFS will upload because we have No Money to pay for LFS quato...

However, we will try to upload the Archieved pth files to [Google Drive](https://drive.google.com/drive/folders/1J2_FkFKFnkagXT4x3rEZagRy-eK4HX8w?usp=sharing).

## Idea

You may heard of [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) which is A Simple Way to Prevent Neural Networks from Overfitting.

While, the author thought the neuro cells died in brain is useful, because it make neuros have ability to random cooperation and can prevent overfitting during learning.

We think the random death of neuros in fully-connected network cause serious memory lossing because one cell's death effect all cell's input behind, makes convergence harder. Especialy, the 50% of death possibly that Dropout's proposer suggested during training, which seams like training an Alzheimer's disease / Cerebral infarction one. 

Miracly, the performance of Network resumed obviously when turn off Dropout during evaluating. Out of our curiosity, if you train the Alzheimer's disease / Cerebral infarction one something, will they mastered the knowledges & skills even better than normals one when they got illness recovered?

Well, we are not sure if Alzheimer's disease / Cerebral infarction one's brain dropout the neuro cells like 'Dropout', which have reborn chance or permanency death, so still needs medical science research. And how about the Sleepy One?

Anyway, if you keep Dropout during evaluating, you will get bad result!

To avoid kill neuro cells like Dropout, we propose Kakuritsu Activation method. Kakuritsu means probability in Japanese, we just let each synapse activation with a probability (suggested p = 50%).

Currently experiment on ImageNet 2012 showed our benefits, especially keep p = 50% in evaluation our performance 30.744% improvement than Dropout at Acc1, and 43.010% improvement at Acc5! That reflected our method improved the neuro cells random cooperation or de-dependence ability better than Dropout do.

To see experiment details, go to [The Archieve](/Archieve) Page.

Still Need more Experment to prove this guess.

## Usage

There are two version of experiment code.

Imagenet\_TrainFromZero.py contains NO pretrained ResNet152's weight, keeps its convolutional layers and removed its linear for experiment.

Imagenet\_TrainForExp.py keeps the pretrained ResNet152's weight, and model structure same as above.

```bash
python3 Imagenet_TrainForYOULIKE.py [args] [Dataset_Dir]

positional arguments:
  DIR                   path to dataset (default: imagenet)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: Kakuritsu and Dropout, with ResNet152
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  -sw, --switch         Switch Dropout or myKakuritsu during Validation
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single
                        node or multi node data parallel training
  --dummy               use fake data to benchmark
```

## Archieve

Archieved code, pth files, experiment results can be found [here](Archieve/)

## Credit

SuperHacker UEFI (Shizhuo Zhang)

Cookie (Yue Fang)

Research supported by Automation School, BISTU; Microsoft The Practice Space (ai-edu)
