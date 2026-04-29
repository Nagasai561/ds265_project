# DS265 - Deep Learning for Computer Vision Course Project

As part of course project, I explored whether using clustering to identify false negatives and construct minibatches with more diverse cluster assignments can improve convergence and final accuracy in SimCLR training on CIFAR-10.

For training take a look at `run.py` and `train_script.sh`. 
For getting evaluation metrics, see `eval_script.sh`.
For visualization of results, see `visualize.ipynb`.

Following python packages are required to run the code:
+ torch 
+ torchvision
+ tqdm
+ pandas
+ seaborn