# NER Project

This project implements a Named Entity Recognition (NER) system using TensorFlow and Keras. The model identifies entities such as people, organizations, locations, etc., in text.

## Project Contents

- **`main.ipynb`**: Contains all the code for:
  - Downloading and preparing the CoNLL 2003 dataset.
  - Data preprocessing (tokenization, encoding, padding).
  - Defining and training a bidirectional LSTM model.
  - Saving vocabularies and the trained model.
  - Example usage for predicting entities in sentences.

- **`.gitignore`**: Excludes files such as datasets, model files, and training example notebooks from version control.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kamaLc73/NER_project.git
   cd NER_project
   ```

2. Install dependencies:

    Make sure to install numpy and tensorflow.

3. Download and prepare the dataset:
   Running the notebook will automatically download the CoNLL 2003 dataset and prepare it for training.

## Model Training

- Run the cells in the `main.ipynb` notebook:
  - Preprocess the data.
  - Define the model (Embedding, Bidirectional LSTM, TimeDistributed Dense).
  - Train the model and evaluate its performance on the validation and test sets.

## Saved Outputs

- **Model**: The trained model is saved as `model_pretrained.keras`.
- **Vocabularies**: The word and tag vocabularies are saved as `word_vocab.json` and `tag_vocab.json`, respectively.

## Example Usage

To use the trained model for predicting named entities:

1. Load the model and vocabularies:
   ```python
   model = tf.keras.models.load_model("model_pretrained.keras")
   with open("word_vocab.json", "r", encoding="utf-8") as f:
       word_vocab = json.load(f)
   with open("tag_vocab.json", "r", encoding="utf-8") as f:
       tag_vocab = json.load(f)
   ```

2. Predict entities in a sentence:
   ```python
   def predict_entities(sentence):
       words = sentence.split()
       encoded_sentence = [word_vocab.get(word, word_vocab["<UNK>"]) for word in words]
       padded_sentence = pad_sequences([encoded_sentence], maxlen=124, padding="post")
       predictions = model.predict(padded_sentence)
       predicted_tags = [np.argmax(pred) for pred in predictions[0]]
       return list(zip(words, [reverse_tag_vocab[tag] for tag in predicted_tags]))

   sentence = "James Bond works at Google INC in New York."
   entities = predict_entities(sentence)
   print(entities)
   ```

## Dataset

The project uses the [CoNLL 2003 dataset](http://lnsigo.mipt.ru/export/datasets/conll2003.tar.gz), which is automatically downloaded and extracted during preprocessing.

## Repository Link

Find the project repository here: [NER Project on GitHub](https://github.com/kamaLc73/NER_project.git)
