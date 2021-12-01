# Block Modeling-Guided Graph Convolutional Neural Networks
This repository contains the demo code of the paper:
>[Block Modeling-Guided Graph Convolutional Neural Networks]()

which has been accepted by *AAAI2022*.
## Dependencies
* Python3.7
* NumPy
* SciPy
* PyTorch
* TensorFlow.keras
## Example Usages
Before running the code， please unzip the *data_geom.zip* and make a directory named *checkpoint*.

* `python main.py --dataset cora --enhance 3.0 --self_loop 1.5`
* `python main.py --dataset citeseer --enhance 4.0 --self_loop 2.0`
* `python main.py --dataset pubmed --enhance 2.0 --self_loop 3.0`
* `python main.py --dataset squirrel --enhance 2.0 --self_loop 0.0`
* `python main.py --dataset chameleon --enhance 0.8 --self_loop 0.0`
* `python main.py --dataset texas --num_gcn_layers 2 --enhance 1.0 --self_loop 0.0`

Please refer to the *args.py* for more parameters.
## Reference
    @inproceedings{BM-GCN,
     title={Block Modeling-Guided Graph Convolutional Neural Networks},
     author={Dongxiao He and Chundong Liang and Huixin Liu and Mingxiang Wen and Pengfei Jiao and Zhiyong Feng},
     booktitle = {AAAI},
     year={2022}
    }
## License
Tianjin University
