import embeddings
import models

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class StoredMapper:
    """Simple helper class used for pickling a generated mapper alongside all required metadata."""

    embedding_words = None
    embedding_vecs = None
    embedding_fn = None
    mapper_type = None
    mapper_model = None
    bert_model = None


class Mapper:
    """
    Baseclass for any available mapping models. Their underlying models get fit/trained on a previously
    generated embedding of terms. Afterwards, new or unknown terms can be embedded into the same BERT model
    and this mapper's model can be used to predict which of the terms in the pre-generated embedding fits
    the new term the closest.
    """
    model = None
    embedding_words = None
    embedding_vecs = None

    def __init__(self, model, embedding_words, embedding_vecs):
        """
        Constructs a new mapper model.

        :param model: The BERT model used to generate the embedding
        :type model: embeddings.EmbeddingModel
        :param embedding_words: A data frame containing a column 'CellName' of terms in the original embedding
        :type embedding_words: pandas.DataFrame
        :param embedding_vecs: A numpy array containing the embedding vectors of the terms in
                               embedding_words['CellName']
        :type embedding_vecs: numpy.ndarray
        """
        self.model = model
        self.embedding_words = embedding_words
        self.embedding_vecs = embedding_vecs
        self.name = None

    def store(self):
        """
        Converts the mapper into a StoredMapper instance for pickling

        :return: The generated StoredMapper instance
        :rtype: StoredMapper
        """
        storage = StoredMapper()
        storage.embedding_words = self.embedding_words
        storage.embedding_vecs = self.embedding_vecs
        storage.embedding_fn = self.model.embedding_fn.__name__
        storage.mapper_type = self.name
        storage.bert_model = self.model.name
        return storage

    def predict(self, text, n_guesses=5):
        """
        Produces n_guesses predictions for potential mappings for the given input text.

        :param text: The text to predict a mapping for
        :type text: str
        :param n_guesses: The number of predictions to generate
        :type n_guesses: int
        :return: Two arrays of length n_guesses; the first containing the predicted mappings,
                 the second an model specific confidence estimate
        :rtype: tuple([str], [float])
        """
        vec = self.model.embed_term(text)
        return self.predict_embedded(vec, n_guesses=n_guesses)

    def predict_embedded(self, vec, n_guesses=5):
        """
        Same as predict but takes in an existing embedding vector instead of generating one.

        :param vec: The embedding vector of the term to generated mapping predictions for
        :type vec: numpy.ndarray
        :param n_guesses: The number of predictions to generate
        :type n_guesses: int
        :return: See predict
        :rtype: tuple([str], [float])
        """
        return None

    def get_estimator(self):
        """
        Returns the underlying sklearn classification model. Referred to as estimator to avoid
        confusion with BERT models used for term embedding.

        :return: The underlying model
        """
        return None


class NearestNeighborMapper(Mapper):
    """
    A mapping model which simply performs a nearest-neighbor search on the embedding space
    (K-nearest-neighbors with K=1).
    """
    _neighbors = None

    def __init__(self, model, embedding_words, embedding_vecs, internal=None):
        super().__init__(model, embedding_words, embedding_vecs)
        self.name = 'nearest_neighbor'
        if internal:
            self._neighbors = internal
        else:
            # n_neighbors = 1 effectively makes this a simple NearestNeighbor classifier
            self._neighbors = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
            self._neighbors.fit(self.embedding_vecs, self.embedding_words['CellName'])

    def predict_embedded(self, vec, n_guesses=5):
        distances, indices = self._neighbors.kneighbors([vec], n_neighbors=n_guesses, return_distance=True)
        mappings = [self.embedding_words['CellName'].iloc[i] for i in indices[0]]
        return mappings, distances[0]

    def get_estimator(self):
        return self._neighbors

    def store(self):
        storage = super().store()
        storage.mapper_model = self._neighbors
        return storage


class SupportVectorMachineMapper(Mapper):
    """
    A mapping model which fits a support vector machine to the pre-generated embedding.
    """
    _svc = None

    def __init__(self, model, embedding_words, embedding_vecs, internal=None):
        super().__init__(model, embedding_words, embedding_vecs)
        self.name = 'svm'
        if internal:
            self._svc = internal
        else:
            self._svc = SVC(probability=True)
            self._svc.fit(self.embedding_vecs, embedding_words['CellName'].tolist())

    def predict_embedded(self, vec, n_guesses=5):
        prediction = self._svc.predict([vec])[0]
        prediction_idx = np.where(self._svc.classes_ == prediction)[0][0]
        probs = self._svc.predict_proba([vec])[0]
        probs_order = np.argsort(probs)

        mappings = [prediction]
        extra = [probs[prediction_idx]]

        cursor = 0

        while len(mappings) < n_guesses and len(mappings) < len(self.embedding_words):
            index = probs_order[cursor]
            cursor += 1

            if index == prediction_idx:
                continue

            mappings.append(self._svc.classes_[index])
            extra.append(probs[index])

        return mappings, extra

    def get_estimator(self):
        return self._svc

    def store(self):
        storage = super().store()
        storage.mapper_model = self._svc
        return storage


def load_mapper_from_storage(storage):
    """
    :param storage: The storage from which to load the original mapper
    :type storage: StoredMapper
    :return: The re-constructed mapper
    :rtype: Mapper
    """
    if not hasattr(embeddings, storage.embedding_fn):
        return None

    embedding_fn = getattr(embeddings, storage.embedding_fn)
    bert_model = models.create_model_from_string(storage.bert_model, embedding_fn=embedding_fn)

    return create_mapper_from_string(storage.mapper_type, bert_model, storage.embedding_words, storage.embedding_vecs,
                                     internal=storage.mapper_model)


def create_mapper_from_string(name, model, embedding_words, embedding_vecs, internal=None):
    """
    Constructs a mapper given the name of it's underlying method (nearest_neighbor, svm).
    See Mapper.__init__ for a detailed description of all parameters.

    :param name: 'nearest_neighbor' or 'svm' (for the respective mapping models)
    :type name: str
    :param model: see Mapper.__init__
    :param embedding_words: see Mapper.__init__
    :param embedding_vecs: see Mapper.__init__
    :param internal: The mapper's internal estimator
    :return: The generated mapper or None if no mapper could be instantiated
    :rtype: Mapper|None
    """
    if name == 'nearest_neighbor':
        return NearestNeighborMapper(model, embedding_words, embedding_vecs, internal=internal)
    elif name == 'svm':
        return SupportVectorMachineMapper(model, embedding_words, embedding_vecs, internal=internal)
    else:
        return None
