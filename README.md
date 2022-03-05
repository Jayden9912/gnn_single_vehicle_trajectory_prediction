# GNN_single_vehicle_trajectory_prediction

This repository is about my learning journey on the whole pipeline of a machine learning project. It is done based on this research paper. For more information,
please refer to that [paper](https://arxiv.org/pdf/2107.03663.pdf) and [github repository](https://github.com/Xiaoyu006/GNN-RNN-Based-Trajectory-Prediction-ITSC2021).

The model used in this project is GNN-RNN for single vehicle trajectory prediction.

# Data preprocessing
Refer to this [github repository](https://github.com/Xiaoyu006/GNN-RNN-Based-Trajectory-Prediction-ITSC2021).

# Train
```
python training.py
```

# Visualisation
![msg312903414-29718](https://user-images.githubusercontent.com/85933053/156882496-25915722-1265-4167-a3f0-2bff03a3a3cf.jpg)
![msg312903414-29721](https://user-images.githubusercontent.com/85933053/156882506-b3f72329-d069-49a3-b89e-9054b2ccc9d1.jpg)
![msg312903414-29719](https://user-images.githubusercontent.com/85933053/156882507-498acc4f-b198-42b4-9134-99d5f2a4a76b.jpg)
Legend:  
grey solid line: history trajectory of surrounding vehicle (3s)  
red solid line: history trajectory of ego vehicle (3s)  
blue dotted line: ground truth of future trajectory of ego vehicle (5s)  
green solid line: predicted future trajectory of ego vehicle (5s)
## Painful lesson
Remember, garbage in, garbage out. At first, I only prepare the dataset with 700+ examples, which is definitely not enough. The model is overfit after training and 
the error never goes down steadily. I am not aware of this and spend weeks to debug until I realised there is no bugs in my code.
