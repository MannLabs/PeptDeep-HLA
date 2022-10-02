# PeptDeep-HLA

A deep learning-model that predicts if a HLA peptide is present or not.

This is a sub-module of [AlphaPeptDeep](https://github.com/mannlabs/alphapeptdeep), and see our [publication](https://www.biorxiv.org/content/10.1101/2022.07.14.499992) for details.

## Installation

After installing anaconda, please clone and install this package using commands below:

```bash
cd path/to/place/this/package
git clone https://github.com/MannLabs/PeptDeep-HLA.git
cd PeptDeep-HLA
pip install .
```

## CLI

After installation, we can use command line interface (CLI) to train sample-specific HLA models and predict HLA peptides either from fasta files or from peptide tables, please type the command below to check usages:

```bash
peptdeep_hla class1 -h
```

## Notebook

See [HLA1_Classifier.ipynb](nbs/HLA1_Classifier.ipynb).

## Citations

Wen-Feng Zeng, Xie-Xuan Zhou, Sander Willems, Constantin Ammar, Maria
Wahle, Isabell Bludau, Eugenia Voytik, Maximillian T. Strauss, Matthias
Mann.
[BioRxiv](https://www.biorxiv.org/content/10.1101/2022.07.14.499992).