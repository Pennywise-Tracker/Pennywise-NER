import subprocess

# command = "python -m spacy train config.cfg --output ./output"
command = "python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./train.spacy"

subprocess.run(command, shell=True, check=True)

import subprocess