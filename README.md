## DelvMap: Completing Residential Roads in Maps Based on Couriers' Trajectories and Satellite Imagery

### Stage 1: Dual Signal Fusion-based Map Completion

Training
--------
    python Dual_Signal_Fusion_based_Map_Completion/train.py --name Your_Task_Name --dataroot Your_dataset --lam 0.2 --batch_size     8 --train_pattern DSFNet --net_trans DSFNet --model DSFNet

Inference
--------
    python Dual_Signal_Fusion_based_Map_Completion/test.py --name Your_Task_Name --dataroot Your_dataset --lam 0.2 train_pattern     DSFNet --net_trans DSFNet --model DSFNet --epoch XX

### Stage 2: Adaptive Map Completion

