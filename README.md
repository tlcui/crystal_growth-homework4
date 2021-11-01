# 有限差分求解枝晶固化生长的相场模拟-太极图形课作业4
这是我学习第4课热传播的求解后完成的作业：用taichi实现晶体固化生长模型（经典的Kobayashi模型）的求解与模拟。
本作业有两份，分别是explicit求解(crystal_growth.py)和implicit求解(crystal_growth_implicit.py)，其中implicit求解的版本是将温度t进行implicit求解，而对phase的求解仍然是explicit的，这与原始论文[2]中的方法保持了一致。

## 背景简介
枝晶固化生长即由液态变为固态的过程。其他具体内容可见参考资料的最后一部分

## 成功效果展示
一个很长的gif片段
![image](https://github.com/tlcui/crystal_growth-homework4/blob/master/crystal_growth.gif)

## 参考资料
[1] https://zhuanlan.zhihu.com/p/411798670  
[2] R. Kobayashi, Modeling and numerical simulations of dendritic crystal growth, Physica D: Nonlinear Phenomena. 63 (1993) 410–423. https://doi.org/10.1016/0167-2789(93)90120-P
