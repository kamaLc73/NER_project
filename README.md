# NER Project

This project implements a Named Entity Recognition (NER) system using TensorFlow and Keras. The model identifies entities such as people, organizations, locations, etc., in text.

## Project Contents

- **`main_en.ipynb`**: Contains the code for:
  - Downloading and preparing the CoNLL 2003 dataset.
  - Data preprocessing (tokenization, encoding, padding).
  - Defining and training a bidirectional LSTM model.
  - Saving vocabularies and the trained model.
  - Example usage for predicting entities in sentences.
- **FastAPI Web Application**:
  - Allows users to input a sentence and get entity predictions dynamically through a styled web interface.
  - Displays results in a table, excluding non-entity tags (`O`).
- **`static/`**: Contains the CSS file for styling the web interface.
- **`templates/`**: Contains the HTML template for the FastAPI app.

## Installation

### Prerequisites

Ensure Python 3.8+ is installed. You also need `pip` for installing dependencies.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/kamaLc73/NER_project.git
   cd NER_project
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv env
   env\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI application:
   ```bash
   uvicorn app:app --reload
   ```

5. Open a browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```

## Using the Web Interface

- Input a sentence in the provided text box.
- Click "Predict" to view recognized entities and their types.
- Results are dynamically displayed in a table format below the form. Entities without labels (`O`) are excluded from the results.

## Model Training

- Run the cells in the `main_en.ipynb` notebook to:
  - Preprocess the data.
  - Define and train the model (Embedding, Bidirectional LSTM, TimeDistributed Dense).
  - Save the model and vocabularies.

## Project Directory Structure

```
NER_project/
├── app.py                # FastAPI application
├── README.md             # Project documentation
├── main_en.ipynb         # Training notebook in English
├── main_fr.ipynb         # Training notebook in French
├── saves/                # Directory for saved model and vocabularies
│   ├── model_pretrained.keras
│   ├── tag_vocab.json
│   └── word_vocab.json
├── static/
│   └── style.css         # CSS for web app styling
├── templates/
│   └── index.html        # HTML template for FastAPI app
├── requirements.txt      # Python dependencies
└── .gitignore            # Ignore unnecessary files
```

## Example Usage (Direct Python)

1. Load the model and vocabularies:
   ```python
   import tensorflow as tf
   import json
   from tensorflow.keras.preprocessing.sequence import pad_sequences

   model = tf.keras.models.load_model("saves/model_pretrained.keras")
   with open("saves/word_vocab.json", "r", encoding="utf-8") as f:
       word_vocab = json.load(f)
   with open("saves/tag_vocab.json", "r", encoding="utf-8") as f:
       tag_vocab = json.load(f)
   reverse_tag_vocab = {v: k for k, v in tag_vocab.items()}
   ```

2. Predict entities in a sentence:
   ```python
   def predict_entities(sentence):
       words = sentence.split()
       encoded_sentence = [word_vocab.get(word, word_vocab["<UNK>"]) for word in words]
       padded_sentence = pad_sequences([encoded_sentence], maxlen=124, padding="post")
       predictions = model.predict(padded_sentence)
       predicted_tags = [reverse_tag_vocab[np.argmax(pred)] for pred in predictions[0]]
       return [(word, tag) for word, tag in zip(words, predicted_tags) if tag != "O"]

   sentence = "James Bond works at Google INC in New York."
   print(predict_entities(sentence))
   ```

## Dataset

The project uses the [CoNLL 2003 dataset](http://lnsigo.mipt.ru/export/datasets/conll2003.tar.gz), which is automatically downloaded and extracted during preprocessing.
