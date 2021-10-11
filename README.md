# BcarCS
Semantic Representation Learning for Code Search with Boosted Co-Attention Mechanism

`BcarCS` is our model proposed in this paper.

`DeepCS` `UNIF` `TabCS` are replication packages of baselines.


## Dependency
> Tested in Ubuntu 16.04
* Python 2.7-3.6
* Keras 2.1.3 or newer
* Tensorflow-gpu 1.7.0


## Usage

   ### DataSets
  You can find the datasets we used at https://pan.baidu.com/s/1unTcrWqULSKT0f-Ll7_Obw . The extraction code is ‘wfjp’
  
  
  And the `/data` folder need be included by `/keras`. 
  
   ### Trained model of BcarCS
   You can find the trained model of BcarCS at https://pan.baidu.com/s/1unTcrWqULSKT0f-Ll7_Obw . The extraction code is ‘wfjp’
   
   ### Configuration
   
   Edit hyper-parameters and settings in `config.py`
   
   ### Train and Evaluate
   
   ```bash
   python main.py --mode train
   
   ```bash
   python main.py --mode eval
