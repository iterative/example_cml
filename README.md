# Classifying Toy Data with CML & DVC


## Overview

This is a fully automated Machine Learning pipeline example that successfully trains a model on AWS to classify data using our local dataset. It then preserves a version of the fully trained model weights on AWS, as well as pulls a version for us to store locally. We'll use the Continous Machine Learning (CML) library to implement continuous integration & delivery (CI/CD). We'll also use Data Version Control (DVC) to track our ML model & dataset so that it's shareable & reproducable. CML & DVC help us version control our entire ML pipeline & let us use simple push/pull commands to move models, data, and code across machines easily. 

## Dependencies
- sklearn 
- numpy 
- matplotlib 
- json

## Instructions

This repository contains code and data for a simple classification problem. To get the dataset, please run `python get_data.py`.


## Credits

Credits go to Iterative.ai! 


