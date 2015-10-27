#Summary

Read the iPython Notebook online (recommended)

http://nbviewer.jupyter.org/github/roryhr/taxi-trajectories/blob/master/taxi-data-notebook.ipynb

# T-Drive trajectory data

Microsoft has made available GPS data from 10,357 taxis in Beijing. Each taxi's location is sampled every 177 seconds on average and we're given a week's worth of data.

http://research.microsoft.com/apps/pubs/?id=152883

# Installation
A minimal data file is stored in `01.zip` along with `User_guide_T-drive.pdf`
which describes the data set in more detail.

To reproduce the plots in the User Guide, you need to download all the data
folders available at the T-Drive link above. 

and place the folders in the `/data` folder. Otherwise, the program will run
without complaining but your figures will differ from mine and those in the
User Guide.


## Reference Publications

1. Jing Yuan, Yu Zheng, Xing Xie, and Guangzhong Sun. Driving with knowledge
from the physical world. In The 17th ACM SIGKDD international conference on
Knowledge Discovery and Data mining, KDD'11, New York, NY, USA, 2011. ACM.

2. Jing Yuan, Yu Zheng, Chengyang Zhang, Wenlei Xie, Xing Xie, Guangzhong Sun, and Yan Huang. T-drive: driving directions based on taxi trajectories. In
Proceedings of the 18th SIGSPATIAL International Conference on Advances in
Geographic Information Systems, GIS '10, pages 99-108, New York, NY, USA,2010.
