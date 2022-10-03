import click
import os
import pandas as pd

from peptdeep.pipeline_api import _get_delimiter

import peptdeep_hla

from peptdeep_hla.HLA_class_I import (
    HLA_Class_I_Classifier, 
    HLA_Class_I_LSTM, HLA_Class_I_Bert,
    pretrained_HLA1
)

def read_peptide_files_as_df(peptide_files):
    df_list = []
    for fname in peptide_files:
        sep = _get_delimiter(fname)
        df_list.append(pd.read_csv(fname, sep=sep))
    seq_df = pd.concat(df_list, ignore_index=True)
    seq_df.drop_duplicates(
        'sequence', inplace=True, 
        ignore_index=True
    )
    return seq_df

@click.group(
    context_settings=dict(
        help_option_names=['-h', '--help'],
    ),
    invoke_without_command=True
)
@click.pass_context
@click.version_option(peptdeep_hla.__version__, "-v", "--version")
def run(ctx, **kwargs):
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.command("class1", 
    help="Predict peptides of HLA Class I "
    "with transfer learning (if training peptides are provided)"
)
@click.option("--prediction-save-as", required=True,
    help="File to save the predicted HLA peptides"
)
@click.option("--fasta", multiple=True,
    help="The input fasta files for training and prediction, "
    "multiple fasta files are supported, such as: "
    "`--fasta 1.fasta --fasta 2.fasta ...`. "
    "If `--peptides-to-predict` is provided, These fasta files "
    "will be ignored in prediction."
)
@click.option("--peptide-file-to-predict", multiple=True,
    help="Peptide file for prediction. "
    "It is an txt/tsv/csv file which contains "
    "peptide sequences in `sequence` column to be predicted. "
    "If not provided, this program will "
    "predict peptides from fasta files."
    " **Optional**, default is empty."
)
@click.option("--pretrained-model", 
    default=pretrained_HLA1, type=str,
    help="The input model for transfer learning or prediction. "
    "**Optional**, default is the built-in pretrained model."
)
@click.option("--prob-threshold", 
    default=0.7, type=float,
    help="Predicted probability threshold to discriminate HLA peptides. "
    "**Optional**, default=0.7."
)
@click.option("--peptide-file-to-train", multiple=True,
    help="Peptide file for transfer learning. "
    "It is an txt/tsv/csv file which contains "
    "true HLA peptide sequences in `sequence` column for training"
    "**Optional**, default is empty."
)
@click.option("--model-save-as",
    help="File to save the transfer learned model. "
    "**Optional**, applicable if `--peptide-file-to-train` is provided."
)
@click.option("--predicting-batch-size", 
    default=4096, type=int,
    help="The larger the better, but it depends on the GPU/CPU RAM. "
    "**Optional**, default=4096."
)
@click.option("--training-batch-size", 
    default=1024, type=int,
    help="**Optional**, default=1024."
)
@click.option("--training-epoch", 
    default=40, type=int,
    help="**Optional**, default=40."
)
@click.option("--training-warmup-epoch", 
    default=10, type=int,
    help="**Optional**, default=10."
)
@click.option("--min-peptide-length", 
    default=8, type=int,
    help="**Optional**, default=8."
)
@click.option("--max-peptide-length", 
    default=14, type=int,
    help="**Optional**, default=14."
)
def run_class1(
    prediction_save_as, 
    fasta:list = [], 
    peptide_file_to_predict:list = [],
    prob_threshold:float=0.7,
    pretrained_model:str=pretrained_HLA1, 
    peptide_file_to_train:list = [], 
    model_save_as:str = '',
    predicting_batch_size:int=4096,
    training_batch_size:int=1024,
    min_peptide_length:int=8,
    max_peptide_length:int=14,
    training_epoch:int=40,
    training_warmup_epoch:int=10,
):
    model = HLA_Class_I_Classifier(
        fasta, 
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length
    )
    model.predict_batch_size = predicting_batch_size

    if os.path.isfile(pretrained_model):
        model.load(pretrained_model)

    if peptide_file_to_train:
        seq_df = read_peptide_files_as_df(peptide_file_to_train)
        model.train(
            seq_df,
            batch_size=training_batch_size,
            epoch=training_epoch, 
            warmup_epoch=training_warmup_epoch,
            verbose=True
        )
        if model_save_as:
            model.save(model_save_as)

    if peptide_file_to_predict:
        seq_df = read_peptide_files_as_df(peptide_file_to_predict)
        seq_df = model.predict_from_peptide_df(
            seq_df, prob_threshold=prob_threshold
        )
    else:
        seq_df = model.predict_from_proteins(
            prob_threshold=prob_threshold
        )
    
    dirname = os.path.dirname(prediction_save_as)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    seq_df.to_csv(prediction_save_as, sep='\t', index=False)
    print(f"Predicted HLA-I peptides were saved in '{prediction_save_as}'.")
    
