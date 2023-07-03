from example_adapter import Node
from os import listdir
import pandas


class Cell(Node):
    """ Provides a basic node like structure to integrate single cell data with cell ontologies.
    
    Simple biocypher node for integreating single cells into a knowlege graph.
    Inherits from Node.

    Args:
        onotloy_terms: (list of strings) 
            List of ontology terms associated with the cell.
            (e.g. ['CL:1234']).
        
        identification_info: (str)
            A string that serves as a hash input for generating the unqiue cell id.
            Any string can be passed here, however we advise to e.g. use the 
            expression data set, or something that identifies the cell.

        scoring: any
            Scoring values of other cell ontology terms e.g. marker2cell output.
    """

    def __init__(self, onotloy_terms:str, identification_info:str, scoring):
        super(Node, self).__init__()
        self.id    = self._generate_id(identification_info)
        self.label = onotloy_terms
        self.properties = {}
        self.properties['Scoring'] = scoring
    
    def _generate_id(self, x) -> None:
        """ Generates a pseudo random hash value."""
        return hash(x)


class CellAdapter:
    """ Adapter for integrating data into the Biocypher knowlege graph.
    
    Creates a instance from Cell for each single cell, present in the 
    speicifed directory.

    Args:
        cell2marker_instance: callable 
            cell2marker instance.
        
        identification_info: (str)
            Path to single cell data output.
    """

    def __init__(self, cell2marker_instance, data_directory_path:str):
        
        self.cell2marker = cell2marker_instance
        self.files       = listdir(data_directory_path)
        self.nodes       =  []

        # Loop through each file in the speicifed directory.
        for file in self.files:
            cell_expresion_data = pandas.read_csv(data_directory_path + file)
            # Create a single string of all the expression data to create a unique hash/id for each cell.
            identification_info = str(cell_expresion_data['Unnamed: 0']) + str(cell_expresion_data['x'])
            # Obtain a cell annotation using cell2marker, to associate cell ontologies with the 
            # expression profile of the single cell.
            scored_cell_ontologies, _ = self.cell2marker(identifiers = cell_expresion_data['Unnamed: 0'], 
                                                         transcript_number = cell_expresion_data['x'])
            
            # Choose the ontogogy with the hightes scoring value.
            ontology_term = scored_cell_ontologies.axes[0][0].replace('_',':')

            # Create a instance for each single cell.
            self.nodes.append(Cell(onotloy_terms = ontology_term, identification_info = identification_info))
        
    def get_nodes(self) -> tuple:

        for node in self.nodes:
            yield(node.get_id(), node.get_label(), node.get_properties())


    
     