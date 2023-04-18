import embeddings
import torch

import pandas as pd

from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    """
    Baseclass for available embeddings models, i.e. models which can be used to generate meaningful
    embeddings for terms with.
    """
    embedding_fn = embeddings.default_embedding_fn
    """The embedding function to be used when embedding terms."""

    def __init__(self, embedding_fn):
        """
        Constructs an embedding model.

        :param embedding_fn: The embedding function to use when embedding terms. Can be changed later.
        """
        self.embedding_fn = embedding_fn

    def embed_term(self, term):
        """
        Embeds the given term using the model and the current embedding function set in embedding_fn.

        :param term: The term to embed
        :type term: str
        :return: The generated embedding vector
        :rtype: numpy.ndarray
        """
        return None

    def embed_with_synonyms(self, cell_type):
        """
        Embeds a cell type specified by a 'Name' and optionally 'Definition' and optionally synonym columns starting
        at index 2 onwards into the embedding. The name, definition and synonyms will all be embedded separately.

        :param cell_type: A pandas.Series containing information about the cell type to embed
        :type cell_type: pandas.Series
        :return: Two lists, the first containing a list of all terms and synonyms which got embedded, the second
                 containing the generated embedding vectors.
        :rtype: ([str], [numpy.ndarray])
        """
        return None


class BertModel(EmbeddingModel):
    """
    An embedding model which is based on the BERT model.
    """

    _model = None
    _tokenizer = None

    def __init__(self, model_version, do_lower_case=False, embedding_fn=embeddings.default_embedding_fn):
        """
        Constructs a new BERT model which can be downloaded through the transformers 'from_pretrained' API.

        :param model_version: The model's identification string (passed to 'from_pretrained')
        :type model_version: str
        :param do_lower_case: Whether or not terms should be converted to lower case when tokenizing
        :type do_lower_case: bool
        :param embedding_fn: The embedding function to use when embedding terms
        :type embedding_fn: callable
        """

        super().__init__(embedding_fn)

        self._model = AutoModel.from_pretrained(model_version, output_hidden_states=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
        self._model.eval()

    def embed_term(self, term):
        input_ids = torch.tensor(self._tokenizer.encode(term)).unsqueeze(0)
        output = self._model(input_ids)
        if self.embedding_fn:
            return self.embedding_fn(output).squeeze().detach().numpy()
        return output

    def embed_with_synonyms(self, cell_type):
        name = cell_type['Name']
        definition = None
        if not pd.isna(cell_type['Definition']):
            definition = cell_type['Definition']

        synonyms = [name]
        vectors = [self.embed_term(name)]
        if definition:
            synonyms.append(definition)
            vectors.append(self.embed_term(definition))

        for i in range(2, cell_type.shape[0]):
            if pd.isna(cell_type.iloc[i]):
                break

            synonym = cell_type.iloc[i]
            synonyms.append(synonym)
            vectors.append(self.embed_term(synonym))

        return synonyms, vectors


class BiobertModel(BertModel):
    """
    Embedding model using BioBERT v1.2.
    """
    def __init__(self, embedding_fn=embeddings.default_embedding_fn):
        super().__init__('dmis-lab/biobert-base-cased-v1.2', do_lower_case=False, embedding_fn=embedding_fn)
        self.name = 'biobert'


class ScibertModel(BertModel):
    """
    Embedding model using SciBERT.
    """
    def __init__(self, embedding_fn=embeddings.default_embedding_fn):
        super().__init__('allenai/scibert_scivocab_uncased', do_lower_case=True, embedding_fn=embedding_fn)
        self.name = 'scibert'


def create_model_from_string(name, embedding_fn=embeddings.default_embedding_fn):
    """
    Constructs an embedding model given its string representation.

    :param name: The model's string representation ({scibert, biobert})
    :type name: str
    :param embedding_fn: The embedding function to use when embedding terms.
    :type embedding_fn: callable
    :return: The created model or None if no model could be created
    :rtype: EmbeddingModel|None
    """
    if name == 'scibert':
        return ScibertModel(embedding_fn=embedding_fn)
    elif name == 'biobert':
        return BiobertModel(embedding_fn=embedding_fn)
    else:
        return None
