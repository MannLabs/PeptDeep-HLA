import click
import os
import pandas as pd

from peptdeep.utils import _get_delimiter

import peptdeep_hla

from peptdeep_hla.HLA_class_I import (
    HLA_Class_I_Classifier, 
    HLA_Class_I_LSTM, HLA_Class_I_Bert,
    pretrained_HLA1
)
from peptdeep_hla.utils import check_is_file

from alphabase.psm_reader import psm_reader_provider

def read_peptide_files_as_df(
    peptide_files, 
    file_type:str='',
    raw_names_to_keep_seqs:list=[],
):
    try:
        reader = psm_reader_provider.get_reader(file_type)
    except KeyError:
        reader = None
    df_list = []
    for fname in peptide_files:
        if check_is_file(fname):
            if reader is None:
                sep = _get_delimiter(fname)
                df_list.append(pd.read_csv(fname, sep=sep))
            else:
                df = reader._load_file(fname)
                reader._translate_columns(df)
                df_list.append(reader._psm_df)
    seq_df = pd.concat(df_list, ignore_index=True)
    if raw_names_to_keep_seqs and "raw_name" in seq_df.columns:
        seq_df = seq_df[seq_df.raw_name.isin(raw_names_to_keep_seqs)]
    return seq_df.drop_duplicates(
        'sequence', ignore_index=True
    )

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
@click.option("--prediction_save_as", required=True,
    help="File to save the predicted HLA peptides"
)
@click.option("--fasta", multiple=True, required=True,
    help="The input fasta files for training and prediction, "
    "multiple fasta files are supported, such as: "
    "`--fasta 1.fasta --fasta 2.fasta ...`. "
    "If `--peptides-to-predict` is provided, These fasta files "
    "will be ignored in prediction."
)
@click.option("--peptide_file_to_predict", multiple=True,
    help="Peptide file for prediction. "
    "It is an txt/tsv/csv file which contains "
    "peptide sequences in `sequence` column to be predicted. "
    "If not provided, this program will "
    "predict peptides from fasta files."
    " **Optional**, default is empty."
)
@click.option("--pretrained_model", 
    default=pretrained_HLA1, type=str,
    help="The input model for transfer learning or prediction. "
    "**Optional**, default is the built-in pretrained model."
)
@click.option("--prob_threshold", 
    default=0.7, type=float,
    help="Predicted probability threshold to discriminate HLA peptides. "
    "**Optional**, default=0.7."
)
@click.option("--peptide_file_to_train", multiple=True,
    help="Peptide file for transfer learning. "
    "It is an txt/tsv/csv file which contains "
    "true HLA peptide sequences in `sequence` column for training"
    "**Optional**, default is empty."
)
@click.option("--training_file_type", 
    default="", type=str,
    help='The input file type for transfer learning, '
    'including "" and all supported formats in `alphabase.psm_reader.psm_reader_provider`. '
    '**Optional**, default is empty "".'
)
@click.option("--training_raw_names", multiple=True,
    help="Only keep sequences from these raw_names for transfer learning"
    "**Optional**, default is empty."
)
@click.option("--model_save_as",
    help="File to save the transfer learned model. "
    "**Optional**, applicable if `--peptide-file-to-train` is provided."
)
@click.option("--training_batch_size", 
    default=1024, type=int,
    help="**Optional**, default=1024."
)
@click.option("--training_epoch", 
    default=40, type=int,
    help="**Optional**, default=40."
)
@click.option("--training_warmup_epoch", 
    default=10, type=int,
    help="**Optional**, default=10."
)
@click.option("--min_peptide_length", 
    default=8, type=int,
    help="**Optional**, default=8."
)
@click.option("--max_peptide_length", 
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
    training_file_type:str = "",
    training_raw_names:list = [],
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

    if check_is_file(pretrained_model):
        model.load(pretrained_model)

    if len(peptide_file_to_train) > 0:
        seq_df = read_peptide_files_as_df(
            peptide_file_to_train,
            file_type=training_file_type,
            raw_names_to_keep_seqs=training_raw_names,
        )
        model.train(
            seq_df,
            batch_size=training_batch_size,
            epoch=training_epoch, 
            warmup_epoch=training_warmup_epoch,
            verbose=True
        )
        if model_save_as:
            model.save(model_save_as)

    if len(peptide_file_to_predict) > 0:
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
    
