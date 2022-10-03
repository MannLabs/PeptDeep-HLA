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

After installation, we can use command line interface (CLI) to train sample-specific HLA models and predict HLA peptides either from fasta files or from peptide tables. Type the command below will show usage messages.

```bash
peptdeep_hla class1 -h
```

Here are the details of the CLI parameters/options:

- --prediction-save-as TEXT: File to save the predicted HLA peptides  [required]
- --fasta TEXT: The input fasta files for training and prediction, multiple fasta files are supported, such as: `--fasta 1.fasta --fasta 2.fasta ...`. If `--peptides-to-predict` is provided, These fasta files will be ignored in prediction.
- --peptide-file-to-predict TEXT: Peptide file for prediction. It is an txt/tsv/csv file which contains peptide sequences in `sequence` column to be predicted. If not provided, this program will predict peptides from fasta files. **Optional**, default is empty.
- --pretrained-model TEXT: The input model for transfer learning or prediction. **Optional**, default is the built-in pretrained model.
- --prob-threshold FLOAT: Predicted probability threshold to discriminate HLA peptides. **Optional**, default=0.7.
- --peptide-file-to-train TEXT: Peptide file for transfer learning. It is an txt/tsv/csv file which contains true HLA peptide sequences in `sequence` column for training**Optional**, default is empty.
- --model-save-as TEXT: File to save the transfer learned model. **Optional**, applicable if `--peptide-file-to-train` is provided.
- --predicting-batch-size INTEGER: The larger the better, but it depends on the GPU/CPU RAM. **Optional**, default=4096.
- --training-batch-size INTEGER: **Optional**, default=1024.

- --training-epoch INTEGER: **Optional**, default=40.

- --training-warmup-epoch: INTEGER **Optional**, default=10.

- --min-peptide-length INTEGER: **Optional**, default=8.

- --max-peptide-length INTEGER: **Optional**, default=14.

- -h, --help                      Show this message and exit.

For example, use the following command to predict from fasta without trainfer learning:

```bash
peptdeep_hla class1 --fasta /Users/zengwenfeng/Workspace/Data/fasta/irtfusion.fasta --prediction-save-as /Users/zengwenfeng/Workspace/Data/fasta/irt_hla.tsv
```

## Notebook

[HLA1_Classifier.ipynb](nbs/HLA1_Classifier.ipynb). We used this notebook to train the pretrained models:

- HLA1_IEDB.pt: the LSTM model trained with HLA1 sequeces from IEDB. This is the default pretrained model in peptdeep_hla.
- HLA1_94.pt: the LSTM model trained with 94 allele types.

[HLA1_transfer.ipynb](nbs/HLA1_transfer.ipynb). A simple example of transfer learning for sample-specific models.

## Citations

Wen-Feng Zeng, Xie-Xuan Zhou, Sander Willems, Constantin Ammar, Maria
Wahle, Isabell Bludau, Eugenia Voytik, Maximillian T. Strauss, Matthias
Mann.
[BioRxiv](https://www.biorxiv.org/content/10.1101/2022.07.14.499992).