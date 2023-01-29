import multiprocessing
import pandas as pd

EMBEDDING_MODEL = """
`doc2vec` is a neural network-based model for representing documents as fixed-length vectors. It is trained on a dataset of text documents and can be used for tasks such as document similarity comparison and document classification.

The `universal-sentence-encoder` and its variants (`universal-sentence-encoder-large`, `universal-sentence-encoder-multilingual`, `universal-sentence-encoder-multilingual-large`) are models that generate embeddings for individual sentences or short paragraphs. These embeddings can be used for tasks such as semantic similarity comparison, text classification, and machine translation. The main difference between these models is their size and the language they are trained on. For example, `universal-sentence-encoder-multilingual` is trained on a diverse set of languages, while `universal-sentence-encoder-large` is a larger version of the model with more parameters.

`distiluse-base-multilingual-cased` is a smaller, faster version of the `universal-sentence-encoder-multilingual` model. This distill version has similar performance with the original version with less computation required.

`all-MiniLM-L6-v2` and `paraphrase-multilingual-MiniLM-L12-v2` are both variants of MiniLM models, which are small language models that can be fine-tuned for a wide range of natural language understanding tasks. The main difference between these two models is the size of the model and the tasks they are trained on. The `all-MiniLM-L6-v2` is a smaller model, while the `paraphrase-multilingual-MiniLM-L12-v2` is larger, and specifically trained on paraphrase identification tasks.
""".strip()

SPEED = """
This parameter is only used when using `doc2vec` as embedding_model.

It will determine how fast the model takes to train. The fast-learn option is the fastest and will generate the lowest quality vectors. The learn option will learn better quality vectors but take a longer time to train. The deep-learn option will learn the best quality vectors but will take significant time to train. 
""".strip()


def get_num_cpu_cores():
    """Get the number of CPU cores."""
    return multiprocessing.cpu_count()


def read_data_csv(path):
    """Read data from a CSV file."""
    try:
        return pd.read_csv(path, engine="python")
    except:
        raise Exception(
            "Could not read CSV file. Please check the encoding of the file."
        )
