import pandas as pd
import numpy as np
import numba

from typing import Union

from alphabase.protein.lcp_digest import get_substring_indices

from alphabase.protein.fasta import load_all_proteins

def load_prot_df(
    protein_data:Union[str,list,dict], 
):
    if isinstance(protein_data, (list,tuple,set)):
        protein_dict = load_all_proteins(protein_data)
    elif isinstance(protein_data, str):
        protein_dict = load_all_proteins([protein_data])
    elif isinstance(protein_data, dict):
        protein_dict = protein_data
    else:
        return pd.DataFrame()
    prot_df = pd.DataFrame().from_dict(protein_dict, orient='index')
    prot_df['nAA'] = prot_df.sequence.str.len()
    return prot_df

def cat_proteins(prot_df, sep='$'):
    return sep+sep.join(prot_df.sequence.values)+sep

def digest_cat_proteins(cat_prot, min_len, max_len):
    pos_starts, pos_ends = get_substring_indices(cat_prot, min_len, max_len)
    digest_df = pd.DataFrame(dict(start_pos=pos_starts, end_pos=pos_ends))
    digest_df["nAA"] = digest_df.end_pos-digest_df.start_pos
    digest_df.sort_values('nAA', inplace=True)
    digest_df.reset_index(inplace=True, drop=True)
    return digest_df

def _get_rnd_subseq(x, pep_len):
    sequence, prot_len = x
    if prot_len <= pep_len: 
        return ''.join(
            [sequence]*(pep_len//prot_len)
        ) + sequence[:pep_len%prot_len]
    start = np.random.randint(0,prot_len-pep_len)
    return sequence[start:start+pep_len]

def get_random_sequences(prot_df:pd.DataFrame, n, pep_len):
    return prot_df.sample(
        n, replace=True, weights='nAA'
    )[['sequence','nAA']].apply(
        _get_rnd_subseq, pep_len=pep_len, axis=1
    ).values.astype('U')

@numba.njit
def check_sty(seq):
    for aa in seq:
        if aa in "STY": return True
    return False

def get_seq(x, cat_prot):
    return cat_prot[slice(x[0],x[1])]

def get_seq_series(df, cat_prot):
    return df[["start_pos","end_pos"]].apply(
        get_seq, axis=1, cat_prot=cat_prot
    )