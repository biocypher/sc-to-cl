import pandas
import yaml

def Creat_Yaml_Config(yaml_file_path:str, input_file:str = 'ontology-mapper/results.csv') -> None: 
    """ Creates .yaml file based on ChatGSE, mapping ontology terms to free text description.

    Reads ChatGSE output files and creates a .yaml file for setting up the Graph Database.
     Args:
            yaml_file_path: (str) 
                path and filename were the created .yaml file should be stored.
                e.g. './local/schema.yaml'
            
            input_file: (str)
                Output file of ChatGSE from which the data can be extracted.
                Default is the example ChatGSE output.
        
        
    Returns:
        No Return.
    
    """
    data = pandas.read_csv(input_file)

    # Create for each cell type description a yaml element of the form:
    # Cariomiocyte:
    #   represented_as: node
    #   label_in_input: CL_12345
    yaml_configuration = {}
    for i, mapped_term_id in enumerate(data['Mapped Term CURIE']):
        source_term = data["Source Term"][i]
        yaml_configuration[source_term] = {"represented_as": "node", "label_in_input": mapped_term_id}

    # Stores the .yaml file.
    with open(yaml_file_path, "w") as f:
        yaml.dump(yaml_configuration, f)