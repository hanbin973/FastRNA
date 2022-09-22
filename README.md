[![Documentation Status](https://readthedocs.org/projects/fastrna/badge/?version=latest)](https://fastrna.readthedocs.io/en/latest/?badge=latest)

# FastRNA
FastRNA is a scalable framework for single-cell RNA sequencing (scRNA-seq) analysis.

# Dependencies and Installation
Dependencies can be installed using the following command.

```
conda install -c conda-forge numpy scipy mkl mkl-include cython pandas
```

Currently, there is an issue with `setuptools>=60.0.0` so please install an older version.
For example,

```
conda install -c conda-forge setuptools=58.0.4
```

The package can be installed by
```
pip install git+https://github.com/hanbin973/FastRNA.git
```

# Getting started
FastRNA requires two inputs: a gene x cell matrix `mtx` and a numpy array containing batch labels `batch_labels`.
Note that both `mtx` and `batch_labels` should be sorted in an ascending order according to `batch_labels`.
Also, `batch_label` should be in an integer format.
This can be done by the following commands.

```
batch_label = pd.factorize(batch_label)[0] # convert batch label to integer
idx_sorted = batch_label.argsort() # sort index in an ascending order

mtx = mtx[:,idx_sorted] # reorder mtx according to sorted index
batch_label = batch_label[idx_sorted] # reorder mtx according to sorted index
```

# Functions
Two functions `fastrna_hvg` and `fastrna_pca` performs feature selection and principal component analysis (PCA), respectively.
For feature selection,
	```
		gene_var = fastrna_hvg(mtx, batch_label)
	```
will return an array of length `n_gene` which is the number of genes (= equals the number of rows of `mtx`) that contains the variance of genes.
These variances can be used for feature selection (e.g. top 1000 genes with highest variance).

For PCA,
	```
		eig_vec, eig_val, cov_mat, pca_coord = fastrna_pca(mtx, numi, batch_label)
	```
will return four objects: eigenvalues, eigenvectors, covariance matrix and PCA coordinates.
`numi` is the user-specified size factor.
A typical choice would be the sum over all UMI counts inside a cell, therefore, `numi = np.asarray(mtx.sum(axis=0)).ravel()`.

# Use example 
Create the `fastrna` folder inside the your project folder and download the `.so` in this repository.
A usage example can be found [here](https://github.com/hanbin973/FastRNA_paper).

# Caution
Current `scipy` sparse matrix does not enforce index sorting.
Therefore, whenever one takes a row subset using `mtx[some_index,:]`, run
	```
	mtx.sort_indices()
	```
before using the functions of FastRNA.
It will take less than a second even for very large matrices.

# License
The FastRNA Software is freely available for non-commercial academic research use. For other usage, one must contact Buhm Han (BH) at buhm.han@snu.ac.kr (patent pending). WE (Hanbin Lee and BH) MAKE NO REPRESENTATIONS OR WARRANTIES WHATSOEVER, EITHER EXPRESS OR IMPLIED, WITH RESPECT TO THE CODE PROVIDED HERE UNDER. IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE WITH RESPECT TO CODE ARE EXPRESSLY DISCLAIMED. THE CODE IS FURNISHED "AS IS" AND "WITH ALL FAULTS" AND DOWNLOADING OR USING THE CODE IS UNDERTAKEN AT YOUR OWN RISK. TO THE FULLEST EXTENT ALLOWED BY APPLICABLE LAW, IN NO EVENT SHALL WE BE LIABLE, WHETHER IN CONTRACT, TORT, WARRANTY, OR UNDER ANY STATUTE OR ON ANY OTHER BASIS FOR SPECIAL, INCIDENTAL, INDIRECT, PUNITIVE, MULTIPLE OR CONSEQUENTIAL DAMAGES SUSTAINED BY YOU OR ANY OTHER PERSON OR ENTITY ON ACCOUNT OF USE OR POSSESSION OF THE CODE, WHETHER OR NOT FORESEEABLE AND WHETHER OR NOT WE HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES, INCLUDING WITHOUT LIMITATION DAMAGES ARISING FROM OR RELATED TO LOSS OF USE, LOSS OF DATA, DOWNTIME, OR FOR LOSS OF REVENUE, PROFITS, GOODWILL, BUSINESS OR OTHER FINANCIAL LOSS.

# Commercialization
To commercialize this software/algorithm, please contact Genealogy Inc. Use of this algorithm for commercial purposes, including implementing and recoding the same algorithm yourself to circumvent license protection, without permission is prohibited as the algorithm is patented. In addition, it is forbidden to insert this code or algorithm into other software packages without permission.

# Citation
This work has been accepted at the American Journal of Human Genetics.
Please cite as

    > H Lee, and B Han (2022). FastRNA: an efficient solution for PCA of single-cell RNA sequencing data based on a batch-accounting count model. Am J Hum Genet, _in press_
