import pandas as pd
import numpy as np
import os
from absl import app, flags, logging
import datetime

#--------------- FLAGS ---------------------#
flags.DEFINE_string('fasta', None, 'name of the fasta file')
flags.DEFINE_string('csv', None, 'AlphaFold CSV File')
flags.DEFINE_string('protein', None, 'EP300 or CREBBP')
FLAGS = flags.FLAGS

#--------------- Definitions ---------------------#
def parse_fasta(file_path):
    """Parses the FASTA file and returns a dictionary of variant_id to sequence."""
    sequences = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith(">"):
                variant_id = lines[i].strip()[1:]  
                sequence = ""
                i += 1
                while i < len(lines) and not lines[i].startswith(">"):
                    sequence += lines[i].strip()  
                    i += 1
                sequences[variant_id] = sequence
            else:
                i += 1
    return sequences

#--------------- Main ---------------------#
def main(argv):
    fasta_file = FLAGS.fasta
    exisitng_csv = FLAGS.csv
    protein = FLAGS.protein
#--------------- Preparing DataFrames ---------------------#
    sequences = parse_fasta(fasta_file)
    df_csv=pd.read_csv(exisitng_csv)

#--------------- Matching Job with Specific Sequence ---------------------#
    df_csv['Variant_ID']=df_csv["jobs"].str.extract(r'_and_(.+)')
    
    for index, row in df_csv.iterrows():
        transcription_factor = row['Variant_ID']
        if transcription_factor in sequences:
            sequence = sequences[transcription_factor]
            df_csv.at[index, 'Sequence'] = sequence
        else:
            logging.warning(f"Transcription factor {transcription_factor} not found in FASTA file.")

#--------------- Matching Job with Domain ---------------------#
    df_csv['Domain'] = df_csv['jobs'].str.extract(r'^(KIX|TAZ1|TAZ2|NCBD)_')

    if protein == 'EP300':
        df_csv.loc[df_csv['Domain'] == 'KIX', 'domain_sequence'] = 'GIRKQWHEDITQDLRNHLVHKLVQAIFPTPDPAALKDRRMENLVAYARKVEGDMYESANNRAEYYHLLAEKIYKIQKELE'
        df_csv.loc[df_csv['Domain'] == 'NCBD', 'domain_sequence'] = 'LKPGTVSQQALQNLLRTLRSPSSPLQQQQVLSILHANPQLLAAFIKQRAAKYANSNPQPIPGQPGMPQGQPGLQPPTMPGQQGVHSNPAMQNMNPMQAGVQRAGLPQQQPQQQLQPPMGGMSPQAQQMNMNHNTMPSQFRDILRRQQMMQQQQQQGAGPGIGPGMANHNQFQQPQGVGYPPQQQQRMQHHMQQMQQGNMG'
        df_csv.loc[df_csv['Domain'] == 'TAZ1', 'domain_sequence'] = 'DPEKRKLIQQQLVLLLHAHKCQRREQANGEVRQCNLPHCRTMKNVLNHMTHCQSGKSCQVAHCASSRQIISHWKNCTRHDCPVCLPL'
        df_csv.loc[df_csv['Domain'] == 'TAZ2', 'domain_sequence'] = 'GDSRRLSIQRCIQSLVHACQCRNANCSLPSCQKMKRVVQHTKGCKRKTNGGCPICKQLIALCCYHAKHCQENKCPVPFCLNI'
    elif protein == 'CREBBP':
        df_csv.loc[df_csv['Domain'] == 'KIX', 'domain_sequence'] = 'GVRKGWHEHVTQDLRSHLVHKLVQAIFPTPDPAALKDRRMENLVAYAKKVEGDMYESANSRDEYYHLLAEKIYKIQKELE'
        df_csv.loc[df_csv['Domain'] == 'NCBD', 'domain_sequence'] = 'QPGMQPQPGLQSQPGMQPQPGMHQQPSLQNLNAMQAGVPRPGVPPQQQAMGGLNPQGQALNIMNPGHNPNMASMNPQYREMLRRQLLQQQQQQQQQQQQQQQQQQGSAGMAGGMAGHGQFQQPQGPGGYPPAMQQQQRMQQHLPLQGSSMGQ'
        df_csv.loc[df_csv['Domain'] == 'TAZ1', 'domain_sequence'] = 'DPEKRKLIQQQLVLLLHAHKCQRREQANGEVRACSLPHCRTMKNVLNHMTHCQAGKACQVAHCASSRQIISHWKNCTRHDCPVCLPL'
        df_csv.loc[df_csv['Domain'] == 'TAZ2', 'domain_sequence'] = 'QESRRLSIQRCIQSLVHACQCRNANCSLPSCQKMKRVVQHTKGCKRKTNGGCPVCKQLIALCCYHAKHCQENKCPVPFCLNI'
    else:
        logging.warning(f"Protein {protein} not recognized.")
    
    current_directory = os.getcwd()
    current_datetime = datetime.datetime.now()

    current_date = current_datetime.date()
    df_csv.to_csv(os.path.join(current_directory, f'materials_to_predict_{current_date}.csv'))
    
if __name__ == '__main__':
    app.run(main)
