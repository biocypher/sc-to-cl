import text2term
import pandas as pd
import string


#TODO Required: pip install text2term
#TODO: Update File for proper use

#cache the ontology
text2term.cache_ontology("http://purl.obolibrary.org/obo/cl.owl", "CL")


#opening the csv file
df_csv = pd.read_csv('ontology-mapper/cell_annotations_in_sc_heart_atlases - cell_annotations.csv')

#drop nans
df_csv = df_csv.dropna(subset=['celltypes'])

#getting celltypes, removing the punctuation and convert it to a list
celltypes = df_csv['celltypes'].str.replace('[{}]'.format(string.punctuation), '').tolist()


#map the two terms
df = text2term.map_terms(celltypes, "CL", output_file="ontology-mapper/ccb/mapping.txt", save_mappings=True, use_cache=True)


#clear the cache
text2term.clear_cache("CL")