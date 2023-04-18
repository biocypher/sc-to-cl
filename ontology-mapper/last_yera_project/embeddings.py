"""embeddings.py

Contains available embedding functions, i.e. functions used to extract an embedding from the hidden states
of a BERT model after forwarding a term through it. Most of the functions are taken from an article by Jay
Alammar: "The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)".

See `here <http://jalammar.github.io/illustrated-bert/>`_ (last checked: 06/28/2022).
"""

import torch


def last_cls(outputs):
    """
    Embedding function which only takes the first token's state of the last hidden layer.


    :param outputs: The full outputs of the BERT model's hidden layers
    :type outputs: torch.Tensor
    :return: A tensor representing the generated embedding
    :rtype: torch.Tensor
    """

    # last_hidden_states has the form
    #   (batch_size, sequence_length, hidden_size)
    # batch_size is always 1 since we only put a single string through
    # sequence_length depends on the text but is always > 0 (index 0 = [CLS])
    # hidden_size is 768 due to BERT's internal layering
    last_hidden_states = outputs[0]
    # The [CLS] token is always the first in any BERT tokenization
    # Its hidden states contains an embedding for basically "the whole sentence"
    # yet given our "sentences" are simply terms for a specific cell type which
    # may or may not consist of more than one word, we simply use its hidden state
    # in the last layer as an embedding
    return last_hidden_states[0][[0]]


def last_hidden_mean(outputs):
    """
    Embedding function which takes the mean over the states of the last hidden layer for all tokens in the input
    sequence.


    :param outputs: The full outputs of the BERT model's hidden layers
    :type outputs: torch.Tensor
    :return: A tensor representing the generated embedding
    :rtype: torch.Tensor
    """

    last_hidden_states = outputs[0]
    # Build the mean over the hidden states for all sequence tokens
    # in the last hidden layer
    return last_hidden_states.mean(1)


def second_to_last_hidden_mean(outputs):
    """
    Embedding function which takes the mean over the states of the second to last hidden layer for all tokens in the
    input sequence.


    :param outputs: The full outputs of the BERT model's hidden layers
    :type outputs: torch.Tensor
    :return: A tensor representing the generated embedding
    :rtype: torch.Tensor
    """

    # Build the mean over the hidden states for all sequence tokens
    # in the second to last hidden layer
    return outputs[2][1:][-2].mean(1)


def sum_four_hidden_mean(outputs):
    """
    Embedding function which sums up the states of the last four hidden layers for each token, then calculates
    the mean of the states of all tokens.


    :param outputs: The full outputs of the BERT model's hidden layers
    :type outputs: torch.Tensor
    :return: A tensor representing the generated embedding
    :rtype: torch.Tensor
    """

    # Sum all hidden states in the last four hidden layers and build the
    # mean from there
    hidden_states = outputs[2][1:]

    layer_9 = hidden_states[-4]
    layer_10 = hidden_states[-3]
    layer_11 = hidden_states[-2]
    layer_12 = hidden_states[-1]

    sum = torch.stack(list([layer_9, layer_10, layer_11, layer_12]), dim=0).sum(dim=0)
    return sum.mean(1)


def sum_all_hidden_mean(outputs):
    """
    Embedding function which sums up the states of all hidden layers for each token, then calculates
    the mean of the states of all tokens.


    :param outputs: The full outputs of the BERT model's hidden layers
    :type outputs: torch.Tensor
    :return: A tensor representing the generated embedding
    :rtype: torch.Tensor
    """

    # Sum all hidden states in all twelve hidden layers and build the mean from
    # there
    hidden_states = outputs[2][1:]

    sum = torch.stack(list(hidden_states), dim=0).sum(dim=0)
    return sum.mean(1)


def concat_four_hidden_mean(outputs):
    """
    Embedding function which concatenates the states of the last four hidden layers for all tokens, then
    calculates the mean of the states for all tokens.


    :param outputs: The full outputs of the BERT model's hidden layers
    :type outputs: torch.Tensor
    :return: A tensor representing the generated embedding
    :rtype: torch.Tensor
    """

    # Concatenate the states for all sequence tokens
    # from the last four hidden layers then return the mean of the
    # widened vectors
    hidden_states = outputs[2][1:]

    layer_9 = hidden_states[-4]
    layer_10 = hidden_states[-3]
    layer_11 = hidden_states[-2]
    layer_12 = hidden_states[-1]

    concatenated_states = torch.cat((layer_12, layer_11, layer_10, layer_9), 2)
    return concatenated_states.mean(1)


default_embedding_fn = concat_four_hidden_mean
"""The embedding function which should be used per default."""
