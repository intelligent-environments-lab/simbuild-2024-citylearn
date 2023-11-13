from io import BytesIO
import os
from pathlib import Path
import requests
from typing import Union
import zipfile
from citylearn.utilities import read_json
import yaml

class FileHandler:
    ROOT_DIRECTORY = os.path.join(*Path(os.path.dirname(os.path.abspath(__file__))).parts[0:-1])
    SETTINGS_FILEPATH = os.path.join(ROOT_DIRECTORY, 'settings.yaml')
    DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')
    FIGURES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'figures')
    DEFAULT_OUTPUT_DIRECTORY = os.path.join(DATA_DIRECTORY, 'simulation')
    SCHEMA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'schema')

    @staticmethod
    def read_yaml(filepath: Union[str, Path]) -> dict:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return data
    
    @staticmethod
    def get_settings() -> dict:
        return FileHandler.read_yaml(FileHandler.SETTINGS_FILEPATH)
    
    @staticmethod
    def download_schema() -> Path:
        url = FileHandler.get_settings()['dataset_url']
        response = requests.get(url)
        schema_filepath = None
        
        if response.status_code == 200:
            os.makedirs(FileHandler.DATA_DIRECTORY, exist_ok=True)

            with zipfile.ZipFile(BytesIO(response.content)) as zip_archive:
                zip_archive.extractall(FileHandler.DATA_DIRECTORY)
                schema_filepath = [f for f in zip_archive.namelist() if f.endswith('schema.json')][0]
                schema_filepath = os.path.join(FileHandler.DATA_DIRECTORY, schema_filepath)
        
        else:
            raise Exception(f'Could not download data at {url}')
        
        return schema_filepath
    
class DataHandler:
    @staticmethod
    def get_evaluation_summary(simulation_id_key: str = None, output_directory: Union[Path, str] = None):
        output_directory = FileHandler.DEFAULT_OUTPUT_DIRECTORY if output_directory is None else output_directory
        data = {}

        for f in os.listdir(output_directory):
            if f.endswith('json') and (simulation_id_key is None or simulation_id_key in f):
                data[f.split('.')[0]] = read_json(os.path.join(output_directory, f))
            else:
                pass

        return data

        
