import pandas as pd 
import numpy as np
import os
from absl import flags, app, logging
import json
import pickle
import gzip
from Bio.PDB import PDBParser

flags.DEFINE_string('output_dir', None, 'directory where predicted models are stored')
FLAGS = flags.FLAGS
#--------------------- Definitions ---------------------#
def calculate_lia_score(pae_mtx, protein_b_len) -> int:
    pae_cutoff = 12.0
    thresholded_pae = np.where(pae_mtx < pae_cutoff, 1, 0)

    # Interaction between A (first part) and B (second part)
    interface_ab = thresholded_pae[:protein_b_len, protein_b_len:]
    interface_ba = thresholded_pae[protein_b_len:, :protein_b_len]

    lia_score = np.count_nonzero(interface_ab) + np.count_nonzero(interface_ba)
    return lia_score


def reverse_and_scale_matrix(pae_mtx: np.ndarray, pae_cutoff: float) -> np.ndarray:
        """
        Reverse and scale the PAE matrix.

        Args:
        - pae_mtx: Input PAE matrix.
        - pae_cutoff: Cutoff value for the PAE matrix.

        Returns:
        - np.ndarray: Scaled PAE matrix.
        """
        # Ensure values above the cutoff are set to 0
        scaled_pae = np.where(pae_mtx > pae_cutoff, 0, pae_mtx)
        return scaled_pae

def lis_score_flat(pae_mtx)->float:
    scaled_pae = reverse_and_scale_matrix(pae_mtx, pae_cutoff=12.0).flatten()
    scaled_pae_nozero = scaled_pae[scaled_pae != 0]
    if len(scaled_pae_nozero) == 0:
        return 0.0
    return np.sum(1 / scaled_pae_nozero) / len(scaled_pae)


def obtain_pae(result_subdir: str, best_model: str) -> np.array:
    """
    Obtains the PAE (Predicted Aligned Error) matrix for the given model from various file formats.

    Args:
        result_subdir (str): The directory where the model results are stored.
        best_model (str): The name of the best model.

    Returns:
        np.array: The PAE matrix.
    """
    pae_mtx = None

    try:
        with open(os.path.join(result_subdir, f"model_{best_model}.json")) as file:
            pae_results = json.load(file)  # Ensure this returns the correct structure
            if isinstance(pae_results, list) and len(pae_results) > 0:
                pae_mtx = np.array(pae_results[0].get('predicted_aligned_error'))
    except FileNotFoundError:
        logging.warning(f"model_{best_model}.json is not found at {result_subdir}")

    try:
        if pae_mtx is None:
            with open(os.path.join(result_subdir, f"result_{best_model}.pkl"), 'rb') as file:
                check_dict = pickle.load(file)
                pae_mtx = check_dict.get('predicted_aligned_error', None)
                if pae_mtx is not None:
                    return np.array(pae_mtx)
    except FileNotFoundError:
        logging.info(f"result_{best_model}.pkl does not exist. Will search for pkl.gz")
    except Exception as e:
        logging.error(f"An error occurred while loading result_{best_model}.pkl: {e}")

    try:
        if pae_mtx is None:
            with gzip.open(os.path.join(result_subdir, f"result_{best_model}.pkl.gz"), 'rb') as file:
                check_dict = pickle.load(file)
                pae_mtx = check_dict.get('predicted_aligned_error', None)
                if pae_mtx is not None:
                    return np.array(pae_mtx)
    except FileNotFoundError:
        logging.warning(f"result_{best_model}.pkl.gz is not found at {result_subdir}")
    except Exception as e:
        logging.error(f"An error occurred while loading result_{best_model}.pkl.gz: {e}")
    return pae_mtx

def average_pae(pae_mtx):
        """
        Calculate the average of the PAE score.

        Args:
        - pae_mtx (np.ndarray): The PAE score matrix.

        Returns:
        - float: The average of the PAE matrix of the whole complex.
        """
        np_pae_data = np.array(pae_mtx)  # Convert the tuple to a numpy array
        flattened_array_pae = np_pae_data.flatten()  # Flatten the array to a one-dimensional array
        average_pae = np.mean(flattened_array_pae)  # Calculate the mean
        return average_pae
#--------------------- Main ---------------------#
def main(argv):
    directory = FLAGS.output_dir
#--------------------- Grabbing Original CSV DataFrame ---------------------#
    csv_file = os.path.join(directory, 'predictions_with_good_interpae.csv')
    original_df = pd.read_csv(csv_file)

#--------------------- Making a DF of all the LIA/LIS ---------------------#
    cumulative_df = pd.DataFrame()
    jobs = os.listdir(directory)  
    count = 0
    results_list = []

    for job in jobs:
        try:
            logging.info(f"Now processing LIA/LIS for {job}")
            results_subdir = os.path.join(directory, job)
            ranking_debug_path = os.path.join(results_subdir, 'ranking_debug.json')

            if os.path.isfile(ranking_debug_path):
                # grab protein B length
                parser = PDBParser(QUIET=True)
                path_pdb = os.path.join(results_subdir, 'ranked_0.pdb')
                structure = parser.get_structure("model", path_pdb)
                chain_lengths = {
                    chain.get_id(): sum(1 for _ in chain.get_residues())
                    for model in structure
                    for chain in model
                }
                protein_b_len = chain_lengths.get('B', 0)

                with open(ranking_debug_path, 'r') as f:
                    ranking_data = json.load(f)
                    best_model = ranking_data['order'][0]
                    pae_mtx = obtain_pae(results_subdir, best_model)

                    if pae_mtx is not None:
                        lia = calculate_lia_score(pae_mtx, protein_b_len)
                        lis = lis_score_flat(pae_mtx)
                        averag_pae = average_pae(pae_mtx)

                        job_df = pd.DataFrame({
                            'job': [job],
                            'lis_score': [lis],
                            'average_pae_score': [averag_pae],
                            'lia_score': [lia]
                        })
                        results_list.append(job_df)
                    else:
                        logging.warning(f"PAE matrix not found for {job}")

        except Exception as e:
            logging.error(f"Error processing job {job}: {e}")
        finally:
            count += 1
            logging.info(f"Done with {job}. {count} of {len(jobs)} processed.")

    # Concatenate all job dataframes at the end
    if results_list:
        cumulative_df = pd.concat(results_list, ignore_index=True)
        merged_df = pd.merge(original_df, cumulative_df, how='left', left_on='jobs', right_on='job')
        merged_df.drop(columns='job', inplace=True)
        print(merged_df)
        output_csv = os.path.join(directory, 'predictions_with_lia_lis.csv')
        merged_df.to_csv(output_csv, index=False)
        print(f"✅ Saved cumulative results to {output_csv}")
    else:
        print("⚠️ No results to save.")

if __name__ == "__main__":
    app.run(main)
