# NextIP

Modify from [Implement of NextIP](https://csse.szu.edu.cn/staff/panwk/publications/Code-NextIP.zip).

# Requirements

- Python 3.6
   - TensorFlow 1.15.3
   - pandas 0.25.3
   - numpy 1.17.3

# Usage

> Datasets:
> [Tmall](https://tianchi.aliyun.com/dataset/dataDetail?dataId=42)
> [UB](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)

``` bash
# 1. Download the dataset to data_process/{DATASET_NAME}/data/
# 2. Preprocess the dataset
# 3. Run the following command to train the model
sh train.sh
```

# Reference

``` bibtex
@inproceedings{luo2022dual,
  title={Dual-Task Learning for Multi-Behavior Sequential Recommendation},
  author={Luo, Jinwei and He, Mingkai and Lin, Xiaolin and Pan, Weike and Ming, Zhong},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={1379--1388},
  year={2022}
}
```