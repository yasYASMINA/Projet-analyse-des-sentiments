import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Assurez-vous que le fichier kaggle.json est dans le répertoire .kaggle
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
kaggle_json_source = 'path_to_your_kaggle.json'  # Changez cette ligne avec le chemin vers votre fichier kaggle.json
kaggle_json_dest = os.path.expanduser('~/.kaggle/kaggle.json')
if not os.path.exists(kaggle_json_dest):
    os.rename(kaggle_json_source, kaggle_json_dest)

# Définir les permissions du fichier
os.chmod(kaggle_json_dest, 0o600)

# Télécharger le dataset
api = KaggleApi()
api.authenticate()
dataset = 'praveengovi/emotions-dataset-for-nlp'
api.dataset_download_files(dataset, path='.', unzip=False)

# Décompresser le fichier zip
zip_path = 'emotions-dataset-for-nlp.zip'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(zip_path)
