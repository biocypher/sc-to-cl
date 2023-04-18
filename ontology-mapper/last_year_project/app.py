"""app.py

This script serves as the entry point for generating mappings of single cell types to the CL ontology.
It can be invoked to either generate a new embedding model (support vector machines or nearest neighbor
search) on top of an embedding into a BERT model previously generated using generator.py. Alternatively,
such a model may be loaded from file. When generating a new model, the embedding_words, embedding_file,
embedding_fn, mapper and bert parameters are required (optionally, save can specify a file to which to
save the generated model). When loading a model from file only the load option is obligatory.

The script can be executed in one of two modes. The first one is the live mode which allows to explore
the generated model and embedding by inputting custom terms and getting the closest mappings.

A more advanced mode can be entered by specified a wordlist. This should be a CSV file using '#' as its
separator and must at least contain a column 'Term' for the list of terms to be mapped. It can optionally
include columns for manually verified solutions ('Solution') a previously assigned mapping ('Mapping') and
whether or not a term has been checked for validity ('Valid'). In wordlist editing mode the console turns
interactive and allows to explore the wordlist and all generated mappings as well as mark them as valid
and/or invalid. Additionally, statistics and the mapping can be exported (and imported again). For a list
of commands press 'H' when entering the editor.

--- Example usages

To generate a new model:
python app.py
    --embedding_words embeddings/scibert_ncs_sum_four_hidden_mean_synonyms.csv
    --embedding_file embeddings/scibert_ncs_sum_four_hidden_mean.npy
    --embedding_fn sum_four_hidden_mean
    --mapper nearest_neighbor
    --bert scibert
    --wordlist workingdir/scA_list.csv
    --save mymodel.mdl

To load an existing model:
python app.py
    --load mymodel.mdl
    --wordlist workingdir/scA_list.csv
"""

import argparse
import pickle

import numpy as np
import pandas as pd

import embeddings
import mapping
import models
from wordlisteditor import WordlistEditor


def main():
    parser = argparse.ArgumentParser(
        description="""Using a pre-generated embeddings file, the same embedding function and a list of words, 
                       try and map new terms onto the ones from the pre-generated embedding """
    )

    new_model = parser.add_argument_group("New mapping model",
                                          "Options which must be present if a new model should be trained")
    new_model.add_argument("--embedding_words", action='store',
                           help="The file from which to load the word list and synonym mapping")
    new_model.add_argument("--embedding_file", action='store',
                           help="The file from which to load pre-generated embedding vectors")
    new_model.add_argument("--embedding_fn", action='store',
                           help="The name of the embedding function which was used to generate the embedding file")
    new_model.add_argument("--mapper", action='store', help="The type of mapper to be trained")
    parser.add_argument("--bert", action='store', choices=['scibert', 'biobert'],
                        help="Specify which BERT model should "
                             "be used")

    existing_model = parser.add_argument_group("Existing mapping model",
                                               "Options which must be present if an existing mapping"
                                               "model should be used")
    existing_model.add_argument("--load", action='store', metavar="MODELFILE",
                                help="The file in which the pre-trained model is stored")

    parser.add_argument("-w", "--wordlist", action='store', nargs='?',
                        help="Optionally, a list of words for which to generate a mapping onto the target embedding")
    parser.add_argument("-s", "--save", action='store', nargs='?',
                        help="When training a new model, save it to the specified file")

    args = parser.parse_args()

    if not (args.load or (args.embedding_words and args.embedding_file and args.embedding_fn and args.mapper)):
        parser.error("Please specify all required arguments to train a new model or load an existing model")

    bert_model = None
    mapper = None
    if args.load:
        with open(args.load, 'rb') as f:
            storage = pickle.load(f)
            mapper = mapping.load_mapper_from_storage(storage)
    else:
        embedding_fn = getattr(embeddings, args.embedding_fn, None)
        if not embedding_fn:
            parser.error("Unknown embedding function")

        # Read embedding data
        embedding_words = pd.read_csv(args.embedding_words, sep='#')
        embedding_vectors = np.load(args.embedding_file, allow_pickle=False)

        # Construct the requested bert_model
        bert_model = models.create_model_from_string(args.bert, embedding_fn=embedding_fn)
        mapper = mapping.create_mapper_from_string(args.mapper, bert_model, embedding_words, embedding_vectors)

        if args.save:
            with open(args.save, 'wb') as f:
                pickle.dump(mapper.store(), f, protocol=pickle.HIGHEST_PROTOCOL)

    # If specified, import wordlist
    wordlist = None
    if args.wordlist:
        wordlist = pd.read_csv(args.wordlist, sep='#')

    if wordlist is not None:
        if len(wordlist) <= 0:
            print("Error: Wordlist is empty")
            return

        solutions = wordlist['Solution'].tolist() if 'Solution' in wordlist else None
        validations = wordlist['Valid'].tolist() if 'Valid' in wordlist else None
        mappings = wordlist['Mapping'].tolist() if 'Mapping' in wordlist else None
        editor = WordlistEditor(mapper, wordlist['Term'].tolist(), solutions=solutions, validations=validations, mappings=mappings)
        editor.start()

    else:
        print("Live Mapping Mode")
        print("BERT model:", mapper.model.name)
        print("Mapper model:", mapper.name)
        print("Embedding Function:", mapper.model.embedding_fn.__name__)

        while True:
            print()
            word = input("Enter word: ")
            mappings, distances = mapper.predict(word)
            print("\tMappings: ", mappings)
            print("\tDistances: ", distances)


main()
