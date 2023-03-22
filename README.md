# Thesis Project: Domain shift-aware Visual Place Recognition

This repository contains the code of the Msc thesis work '[Domain shift-aware Visual Place Recognition](http://resolver.tudelft.nl/uuid:bdda0da4-69fc-4007-912c-23258d751bea)'.

## Installation

This repository was created to work with two other repositories. 

* [The Deep Visual Geo-Localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)
* [The corresponding datasets repository](https://github.com/gmberton/datasets_vg)

After downloading all three repositories the folder structure should look as follows:

.\
├── vg_bench\
├── datasets_vg\
└── thesis_project

A working Anaconda environment can be created by navigating to the _thesis_project_ folder and creating an environment from the .yml file:\
`conda env create --name envname --file=environment.yml`

Finally, the techniques used in the ensemble can also be downloaded [here](https://github.com/gmberton/deep-visual-geo-localization-benchmark) (the model zoo), and should be saved in the _pre-trained_VPR_networks_ folder.

## Usage

The code in this repository can be used to test the ensemble-methods proposed in the thesis as well as the baselines both quantitatively and qualitatively. 

### Quantitative Testing


In order to run quantative tests, the _ensemble_testing.py_ script is used. 

Testing the ensemble with the KNN(n=1) discriminative method on the st_lucia dataset is done as follows:\

`python ensemble_testing.py --dataset_name=st_lucia`

More hyperparameters can be given as arguments. Testing the GMM with a set of different values for _n_ on the same dataset, for example:

`python ensemble_testing.py --dataset_name=st_lucia --weight_function=GMM --tweakpara_list=1,2,3`

Testing a technique (3) individually:

`python ensemble_testing.py --dataset_name=st_lucia --ds_aware=False --fuse_type=individual --indiv_tech=3`

Finally, testing the average voting baseline:

`python ensemble_testing.py --dataset_name=st_lucia --ds_aware=False --fuse_type=avg_voting`


### Qualitative testing




## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
