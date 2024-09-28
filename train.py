import subprocess

command = "python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./train.spacy --verbose"
# command = "python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy --verbose"

subprocess.run(command, shell=True, check=True)

import subprocess