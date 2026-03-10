import os
from utils.file_and_folder_operations import maybe_mkdir_p

# Default base paths (similar to nnU-Net)
# Users should set these environment variables or they will look in the current directory
Radiomics_raw = os.environ.get('Radiomics_raw')
Radiomics_preprocessed = os.environ.get('Radiomics_preprocessed')
Radiomics_results = os.environ.get('Radiomics_results')

if Radiomics_raw is None:
    Radiomics_raw = "./Radiomics_raw"

if Radiomics_preprocessed is None:
    Radiomics_preprocessed = "./Radiomics_preprocessed"

if Radiomics_results is None:
    Radiomics_results = "./Radiomics_results"


maybe_mkdir_p(Radiomics_raw)
maybe_mkdir_p(Radiomics_preprocessed)
maybe_mkdir_p(Radiomics_results)
