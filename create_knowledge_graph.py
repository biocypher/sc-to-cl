from biocypher import BioCypher
from create_yaml_config import Creat_Yaml_Config
from sc2cl.cell_adapters import Cell, CellAdapter
from Marker2Cell.marker2cell import Marker2Cell

from sc2cl.adapters.example_adapter import (
    ExampleAdapter,
    ExampleAdapterNodeType,
    ExampleAdapterEdgeType,
    ExampleAdapterProteinField,
    ExampleAdapterDiseaseField,
)

# Please specifiy a path to 

# Create the conifg. yaml from a ChatGSE output. Please provided a 
# path your ChatGSE output otherwise a toy data set is used.
config_yaml = "config/schema.yaml"
data_path   = "data/"             # Path were the analyzed single cell data is stored. E.g. output of the seurat script.
Creat_Yaml_Config(config_yaml)

# Instantiate the BioCypher interface
# You can use `config/biocypher_config.yaml` to configure the framework or
# supply settings via parameters below
bc = BioCypher(
    biocypher_config_path="config/biocypher_config.yaml",
)
bc.show_ontology_structure()

# Choose node types to include in the knowledge graph.
# These are defined in the adapter (`adapter.py`).
node_types = [
    ExampleAdapterNodeType.PROTEIN,
    ExampleAdapterNodeType.DISEASE,
]

# Choose protein adapter fields to include in the knowledge graph.
# These are defined in the adapter (`adapter.py`).
node_fields = [
    # Proteins
    ExampleAdapterProteinField.ID,
    ExampleAdapterProteinField.SEQUENCE,
    ExampleAdapterProteinField.DESCRIPTION,
    ExampleAdapterProteinField.TAXON,
    # Diseases
    ExampleAdapterDiseaseField.ID,
    ExampleAdapterDiseaseField.NAME,
    ExampleAdapterDiseaseField.DESCRIPTION,
]

edge_types = [
    ExampleAdapterEdgeType.PROTEIN_PROTEIN_INTERACTION,
    ExampleAdapterEdgeType.PROTEIN_DISEASE_ASSOCIATION,
]

# Create a protein adapter instance
adapter = ExampleAdapter(
    node_types=node_types,
    node_fields=node_fields,
    edge_types=edge_types,
    # we can leave edge fields empty, defaulting to all fields in the adapter
)

# Create an instance of Marker2Cell.
marker2cell  = Marker2Cell(download_data=True)
cell_adapter = CellAdapter(marker2cell, data_path)

# Create a knowledge graph from the adapter
bc.write_nodes(cell_adapter.get_nodes())
#bc.write_edges(adapter.get_edges())

# Write admin import statement and check structure / output
bc.summary()
