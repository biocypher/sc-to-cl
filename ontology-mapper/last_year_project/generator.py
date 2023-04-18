"""generator.py

Embeds a list of cell types previously extracted from the CL ontology using the extract.py script
into a BERT model using one or multiple embedding functions and saves the resulting embedding vectors
and wordlists. For each cell type, both its name and description as well as potential synonyms will be
included in the embedding.

The result can be exported to either Tensorboard for visualization, as a numpy file or both.
When choosing a numpy export, a .npy file containing the raw vectors of the embedding and a .csv file containing
a list of terms and what cell type they originated from will be generated.

--- Example usages

python generator.py
    --numpy
    --tensorboard
    --logdir runs/scibert_ncs
    --embeddings last_cls last_hidden_mean second_to_last_hidden_mean sum_four_hidden_mean sum_all_hidden_mean
        concat_four_hidden_mean
    --prefix embeddings/scibert_ncs_
    --bert scibert
    embeddings/native_cells.csv
"""

import argparse

import embeddings
import models

import numpy as np

import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(
    description="""Generate embeddings of a wordlists for display in TensorBoard or export them to a numpy file"""
)

export_group = parser.add_argument_group("Exports", "Available options for exporting the generated embeddings")
export_group.add_argument('-t', '--tensorboard', action='store_true', dest='tensorboard', help="Enable exporting a TensorBoard experiment")
export_group.add_argument('-n', '--numpy', action='store_true', dest='numpy', help="Enable exporting the generated embedding to a .npy file")
parser.add_argument('--logdir', action='store', help="The log directory when exporting a TensorBoard experiment")
parser.add_argument('--prefix', action='store', default="", help="A prefix to be added to an exported .npy file's filename")
parser.add_argument('--embeddings', action='store', nargs='+', default=[embeddings.default_embedding_fn.__name__], metavar="EMBEDDING", help="A list of embedding functions to use to extract embeddings")
parser.add_argument('--bert', action='store', choices=['scibert', 'biobert'], required=True, help="Which BERT model to use")
parser.add_argument('wordlist', action='store', help="The file from which to read the cell types")

args = parser.parse_args()

if not (args.tensorboard or args.numpy):
    parser.error("No export requested, add at least one of the export specifiers")

embedding_fns = []
if args.embeddings:
    for embedding_name in args.embeddings:
        if not hasattr(embeddings, embedding_name):
            parser.error("Embedding function not found")
        embedding_fns.append(getattr(embeddings, embedding_name))


cell_types = pd.read_csv(args.wordlist, sep='#')


writer = None
if args.tensorboard:
    writer = SummaryWriter(log_dir=args.logdir)

bert_model = models.create_model_from_string(args.bert)

total_embeddings = len(embedding_fns)
total_words = len(cell_types.index)

current_words = 0

print("Embedding words into BERT model...")
bert_model.embedding_fn = None
raw_names = []
raw_synonyms = []
raw_embeddings = []
for _, cell_type in cell_types.iterrows():
    if current_words % 100 == 0:
        print("\tEvaluated", "{:.2%}".format(current_words / total_words), "% of words (", current_words, " / ",
              total_words, ")")

    synonym, embedding = bert_model.embed_with_synonyms(cell_type)
    raw_names = raw_names + [cell_type['Name']] * len(synonym)
    raw_synonyms = raw_synonyms + synonym
    raw_embeddings = raw_embeddings + embedding
    current_words += 1

print("\tEvaluated", current_words, "words with a total of", len(raw_synonyms), "synonyms")

current_embedding = 1

for embedding_fn in embedding_fns:
    bert_model.embedding_fn = embedding_fn
    embedding_name = embedding_fn.__name__

    print("Generating embedding", current_embedding, " (", embedding_name, ") out of", total_embeddings)

    result_embeddings = [embedding_fn(x).squeeze().detach().numpy() for x in raw_embeddings]

    if args.tensorboard:
        print("\tWriting to Tensorboard...")
        matrix = torch.from_numpy(np.stack(result_embeddings))
        writer.add_embedding(matrix, metadata=raw_synonyms, tag=embedding_name)

    if args.numpy:
        print("\tSaving numpy vectors...")
        np.save(args.prefix + embedding_name + ".npy", result_embeddings, allow_pickle=False)
        mapping = pd.DataFrame({'Synonym': raw_synonyms, 'CellName': raw_names})
        mapping.to_csv(args.prefix + embedding_name + "_synonyms.csv", sep='#')

    current_embedding += 1

if args.tensorboard:
    writer.close()
