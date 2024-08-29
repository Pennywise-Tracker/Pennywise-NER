## Setup Instructions

### 1. Create and Activate Conda Environment

- **Create a Conda Environment**:
   ```bash
   conda create -n spacy-env python=3.10
   ```
- **Activate Conda Environment**:
   ```bash
   conda activate spacy-env
   ```

### 2. Install Required Packages

- **Install SpaCy**:
   ```bash
   conda install -c conda-forge spacy
   ```
- **Install CuPy (for GPU support)**:
   ```bash
   conda install -c conda-forge cupy
   ```
- **Install CuPy (for GPU support)**:
   ```bash
   conda install -c conda-forge cupy
   ```
- **Download SpaCy Language Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### 3. Annotate Data

For annotating text data, use the [NER Annotator Tool](https://tecoholic.github.io/ner-annotator/) to create and label your training data. Follow the instructions on the website to generate the annotated data in JSON format.

### 4. Prepare Data for Training

- **Convert JSON Data to DocBin Format**:
    - Run the `convert_json_to_docbin.py` script to convert your annotated JSON data into SpaCy's DocBin format.
    ```bash
    python convert_json_to_docbin.py
    ```

- **Train the Model**:
    - Use the following command to train the SpaCy model with your data:
    ```bash
   python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./train.spacy 
    ```

### 5. Test the Model with Custom Text Input

- **Run the `main.py`**:
    ```bash
    main.py
    ```
This script allows you to input custom text and view recognized entities. Ensure you have your trained model saved in the output/ directory.
