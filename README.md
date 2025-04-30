# TF_binding
A deep learning model for predicting EP300-Transcription Factor interactions using integrated structural data and binding assays. 

## Abstract

EP300 is a histone acetyltransferase (HAT) enzyme that serves as a critical transcriptional co-activator for a variety of lineage-specific transcription factors (TFs). Many TFs interact with and recruit this HAT through dedicated protein-protein interactions, enabling it to acetylate histones and increase chromatin accessibility at specific genomic loci. While small molecule inhibition of EP300 has been shown to alter many specific TF-dependent gene expression programs, it remains difficult to predict what TFs are directly EP300-dependent and which may therefore be targeted therapeutically using this approach.

To address this challenge, we developed a deep learning model capable of prioritizing EP300-TF interactions for experimental testing called TF Binding. We further created individual models for each domain exploring their non-linear relationships with transcription activation and AF metrics. 

Our approach integrates recently developed high-throughput EP300-TF binding [data](https://www.biorxiv.org/content/10.1101/2024.08.19.608698v1) in combination with AlphaFold2-based structure prediction to develop a novel scoring system called HAT_score that significantly enhances the prediction of EP300-TF relative to traditional structure prediction metrics. We anticipate this model will be useful to prioritize the discovery of uncharacterized regulatory interactions, providing a link between high-throughput transcriptional activation assays and EP300 that will be potentially extensible to additional druggable transcriptional co-activator families.

## The model
This repository contains code for the paper: _________________________________

The primary goal of this repository is to give the tools and understanding for users to use our model or even train their own with new datasets. 

In this repository, there are...
1. [Trained Model](trained_model)
2. [Tools to Train a New Model](toolkit_to_train)
3. [How to create a predicitions](trained_model)
4. [Examples/Data](examples)



## AlphaPulldown Workflow
In order to explore binding of different EP300 domains with significant number of transcription factors, we implemented the [AlphaPulldown (v.0.30.7)](https://academic.oup.com/bioinformatics/article/39/1/btac749/6839971) Pipeline using a ColabFold Search v1.5.5 for multiple sequence alignments. 

### How to run AlphaPulldown on the NIH HPC Biowulf Cluster:
#### **ColabFold Search**
```bash
sinteractive --mem=128g --cpus-per-task=16 --gres=lscratch:100
module load colabfold alphapulldown/0.30.7
colabfold_search --threads $SLURM_CPUS_PER_TASK for_colabfold.fasta $COLABFOLD_DB pulldown_cf_msas
create_individual_features.py --fasta_paths=bait.fasta,candidate.fasta --output_dir=pulldown_cf_msas \
    --use_precomputed_msas=True --max_template_date=2023-01-01 \
    --use_mmseqs2=True --skip_existing=True
```

#### **AlphaPulldown**
```bash
sinteractive --mem=150g -c8 --gres=lscratch:50,gpu:a100:1 --time=3-12:00:00
module load alphapulldown/0.30.7
run_multimer_jobs.py --mode=pulldown --num_cycle=3 --num_predictions_per_model=1 \
    --output_path=pulldown_model --protein_lists=candidate.txt,bait.txt \
    --monomer_objects_dir=pulldown_cf_msas

run_get_good_pae.sh --output_dir pulldown_model --cutoff=50
```

To view example `sbatch` files, [click here]().

## Requirements


---

## Declaration of Generative AI Usage

This project utilized OpenAI's ChatGPT to assist in generating Python code, documentation, and explanatory content.

---

## References
- DelRosso, N. et al. *High-throughput affinity measurements of direct interactions between activation domains and co-activators.*
**bioRxiv,** 608698, (2024) [Paper Link](https://doi.org/10.1101/2024.08.19.608698)

- Dingquan Yu, Grzegorz Chojnowski, Maria Rosenthal, Jan Kosinski. *AlphaPulldown—a Python package for protein–protein interaction screens using AlphaFold-Multimer*, **Bioinformatics**, 39(1), 2023. [Paper Link](https://doi.org/10.1093/bioinformatics/btac749)

> This work utilized the computational resources of the [NIH HPC Biowulf cluster](https://hpc.nih.gov).
