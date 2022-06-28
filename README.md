# 基于MSCOCO数据集的图文生成器

## 项目简介
此项目是一个基于MSCOCO数据集训练的根据输入图像生成图像描述语句的神经网络

~~效果不是很好，不要太过于期待，仅供学习参考！~~

在模型架构方面使用了以下模型：

- Inception-V3 (Image-net预训练)
- GLoVe词表征工具
- LSTM循环神经网络

语句生成算法使用了以下两种：

- 贪心搜索
- 集束搜索

使用Microsoft COCO图像描述评估代码进行评估，在BLEU-4和CIDEr上的得分与基线模型对比如下：

| 模型       | BLEU-4 |CIDEr|
|----------|--------|---|
| m-RNN(2) | 0.302  |0.886|
| MSR      | 0.291  |0.912|
| m-RNN    | 0.299  |0.917|
| MSR-C    | 0.308  |0.931|
| NICv2    | 0.309  |0.943|
| 此模型      | 0.053  |0.285|
| 人类表现     | 0.217  | 0.854  |

总之就是非常的不尽人意

## 模型训练

1. 在MSCOCO官网 https://cocodataset.org/ 下载以下两个数据集：
   - [2017 Train images [118K/18GB]](http://images.cocodataset.org/zips/train2017.zip)
   - [2017 Test images [41K/6GB]](http://images.cocodataset.org/zips/test2017.zip)
2. 下载完成后解压到 *MSCOCO/* 路径下
3. 在 *main.py* 中设置选择训练图片的张数和epoch，运行即可开始训练<br>
    实测RTX3060在20000张图片3000epoch的设置下需要15小时左右，图片数量再往上，训练时间指数级增加
4. 模型每epoch将按照时间戳和训练次数保存为 *models/\*finish-time\*/model\_\*current-epoch\*.h5* 
5. 最终模型将保存在 *models/model_\*num\*\_\*epoch\*\_\*finish\-time\*.h5*
6. 训练过程中的 *loss* 将保存在 *history/histort_\*finish-time\*.txt* 中，可使用以下代码进行读取
    ```python
    import pickle
    import matplotlib.pyplot as plt
    
    with open('history/history_20220408_1115.txt', 'rb') as f:
        history = pickle.load(f)
    plt.plot(history['accuracy'])
    plt.show()
   ```
   
## 模型语句生成

1. 在 *val.py* 中根据不同的需求，选择为 一个/一组/全部 图片生成描述语句
2. 生成语句时可以通过修改函数本身以及函数参数来进行测试
3. 在 *gui.py* 中还提供了简单的可交互的图像生成界面，直接运行即可

## 模型评估

1. 在 *val.py* 中选择为全部图片生成描述语句，将会保存在 *MSCOCO/annotations/captions_val2017_fakecap_results.json* 之中
2. 运行 *coco_eval_example.py* 来进行打分<br>
    源代码请见 https://github.com/wykang/pycocoevalcap 

## 参考资料
- [Common Objects in Context](https://cocodataset.org/)
- [用 Keras 创建自己的图像标题生成器](https://blog.csdn.net/BF02jgtRS00XKtCx/article/details/113903655)
- [Python中图像标题生成的注意机制实战教程](https://blog.csdn.net/Together_CZ/article/details/113727106)



