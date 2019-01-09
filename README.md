# vec2link:

This repository provides a Python implementation of *vec2link*

The *vec2link* is a novel link prediction framework, by jointly modeling both user's social network relationship and spatiotemporal check-in information

Before to execute *vec2link*, it is necessary to install the following packages:
<br/>
``pip install futures``
<br/>
``pip install fastdtw``
<br/>
``pip install gensim``

## Requirements

-  numpy==1.13.1
-  networkx==2.0
-  scipy==0.19.1
-  tensorflow==1.3.0
-  gensim==3.0.1
-  scikit-learn==0.19.0

### Data Set
- --In /data/:You can download [gowalla data set](http://snap.stanford.edu/data/loc-gowalla.html) here.

### Basic Usage

- --run /data_process/ by the following order:data_proper_time.py,data_filter.py and data_split.py
- --run /NE/ [Open NE](https://github.com/thunlp/OpenNE) or /struc2vec/ [struc2vec](https://github.com/leoribeiro/struc2vec) to obtain the network representation.
- --run /poissionmf/ by the following order:data_checkins.py and create_matrix to obtain the spatiotemporal check-in information representation
- --run /evaluate/ to estimate the performance of vec2link. The four_methods.py and model_mlp.py will generate four contrastive experiments. The lsh_joint.py and model_scalp.py is used to evaluate the *vec2link*

#### Options

- --You can choose the location range of check_ins or the check-in time by modifying /data_process/data_proper_time.py
- --You can choose *n* user records, *m* POI IDs, and the number of user nodes(the sub-graph) by modifying /data_process/data_filter.py
- --You can adjust the proportion of the training set and the test set by modifying /data_process/data_split.py
- --You can choose the method of network embedding in /NE/ which includes:node2vec,line,deepWalk,grarep or /struc2vec/ which includes:struc2vec to generate the network representations

### Miscellaneous

*Note:* This is only a reference implementation of *vec2link*.
