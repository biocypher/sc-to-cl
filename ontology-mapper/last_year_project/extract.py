"""extract.py

Simple script which attempts to download the CL ontology from obolibrary.org and extracts a list of
cell types including their descriptions and potential synonyms. Requires a single positional argument
specifying the file to which to write the extracted list in CSV format.

Note: this script is usually the first step in setting up embeddings. Because the descriptions of cell types
at times use include ',' and ';' neither could be used as a separator for the generated CSV file. Instead,
'#' will be used - for consistency it will be used and expected in all CSV files for further steps as well.
"""

import argparse

from owlready2 import *

SEPARATOR = '#'

parser = argparse.ArgumentParser(
    description="Extract the list of cells found under the 'native cell' root node from the CL ontology"
)
parser.add_argument("outfile", action='store', default="native_cells.csv",
                    help="The file into which to write the list of results")
args = parser.parse_args()

target_namespace = "cell"

cl = get_ontology("http://purl.obolibrary.org/obo/cl.owl")
cl.load()

obo = cl.get_namespace("'http://purl.obolibrary.org/obo'")
native_cell_class = cl.search_one(iri="http://purl.obolibrary.org/obo/CL_0000003")

oboInOwl = cl.get_namespace("http://www.geneontology.org/formats/oboInOwl")
has_obo_namespace_annotation_prop = oboInOwl.hasOBONamespace
definition_prop = cl.search_one(iri="http://purl.obolibrary.org/obo/IAO_0000115")

cell_results = []
max_synonyms = 0

for clazz in cl.classes():
    if not hasattr(clazz, "label") or len(clazz.label) < 1:
        pass

    # if target_namespace in clazz.hasOBONamespace:
    # This entity is in fact a cell ; narrow it down further to only native cells for now:
    ancestors = list(clazz.ancestors())
    if native_cell_class in clazz.ancestors():
        cell_name = clazz.label.first()
        cell_definition = definition_prop[clazz].first() or ""

        cell_data = [cell_name, cell_definition]


        if hasattr(clazz, "hasExactSynonym"):
            synonym_count = 0
            for synonym in clazz.hasExactSynonym:
                cell_data.append(synonym)
                synonym_count += 1
            max_synonyms = max(max_synonyms, synonym_count)

        print(";".join(cell_data))

        cell_results.append(cell_data)

with open(args.outfile, 'w', encoding='utf-8') as f:
    header = SEPARATOR.join(["Name", "Definition"])
    for i in range(max_synonyms):
        header += "{}Synonym_{}".format(SEPARATOR, i)
    header += "\n"

    f.write(header)

    for cell_data in cell_results:
        cell_name = cell_data[0]
        # only first sentence of definition is required (BERT can only take two)
        definition = cell_data[1].split('.')[0] + '.'
        synonyms = cell_data[2:]

        row = "{}{}{}".format(cell_name, SEPARATOR, definition)
        if len(synonyms) > 0:
            row += SEPARATOR + SEPARATOR.join(synonyms)
        row += SEPARATOR * (max_synonyms - len(synonyms))
        row += "\n"

        f.write(row)
