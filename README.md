# Multi-label X-ray Imagery Classification via Bottom-up Attention and Meta Fusion 



## Main Requirements

+ Python3
+ torch == 1.3.1
+ torchvision == 0.4.2



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
   python3 test_on_cpu_total.py 
   ```

3. Calculate the mAP & FP & FN & TP & TN:

   ```shell
   python3 calc_pre_recall.py
   ```

   