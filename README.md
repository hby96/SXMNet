# Multi-label X-ray Imagery Classification via Bottom-up Attention and Meta Fusion 

This repository is the official PyTorch implementation of paper [Multi-label X-ray Imagery Classification via Bottom-up Attention and Meta Fusion](https://hby96.github.io/_pages/pdfs/Multi-label_X-ray_Imagery_Classification_via_Bottom-up_Attention_and_Meta_Fusion_ACCV2020.pdf?raw=true). (The work has been accepted by ACCV 2020)

## Main Requirements

+ Python3
+ torch == 1.3.1
+ torchvision == 0.4.2



## Pretrain model

 [Baidu Cloud](https://pan.baidu.com/s/1KrhhagaNeonQ6tLa5Qj05Q)      password: wf3d

download the pretrain model and put it into the `test_offline` folder.



## Usage

### Train

1. get into the main folder:

   ```shell
   cd main/
   ```

2. train the attention task (**Stage I**) as follows:

   ```shell
   sh run_train_bbox_insight.sh
   ```

3. train the multi-label classification task (**Stage II**) as follws:

   ```shell
   sh run_train_attention_multi_label.sh
   ```


### Inference

1. get into the test_offline folder:

   ```shell
   cd ../test_offline/
   ```

2. get inference results:

   ```shell
   python3 test_on_cpu_total_merge.py 
   ```

3. Calculate the mAP & FP & FN & TP & TN:

   ```shell
   python3 calc_pre_recall_merge.py
   ```

   

## Citation

If you find this work is helpful, please cite this paper.

```
@inproceedings{Hu2020Xray,
  author    = {Benyi Hu and Chi Zhang and Le Wang and
               Qilin Zhang and Yuehu Liu},
  title     = {Multi-label X-ray Imagery Classification via Bottom-up Attention and Meta Fusion},
  booktitle = {15th Asian Conference on Computer Vision, ({ACCV})},
  year      = {2020},
}

```

## 