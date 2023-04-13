import pandas
import warnings
from os import mkdir

class Marker2Cell:
    """ Marker2Cell infers cell ontology terms from cell markers.
    
    Marker2Cell computes possible cell ontologies terms from a given list of cell makers 
    based on the CellMarker2 database (http://117.50.127.228/CellMarker/CellMarker_download.html)[1].
    For each found cell ontologie term an score is calculated based on 
    the total number of markers associated with the respective term and 
    the number of provided markers that matched with the term.
    
    [1] Zhang, Xinxin, et al. "CellMarker: a manually curated resource of 
        cell markers in human and mouse." 
        Nucleic acids research 47.D1 (2019): D721-D728.
    
    Example:
        > marker2cell  = Marker2Cell() # OR marker2cell  = Marker2Cell(download_data = True)
        > cell_markers = ['CD20', 'CD4', 'CD80']
        > scored_ontologie, scored_cell_names = marker2cell(cell_markers)
        ...
        > marker2cell  = Marker2Cell('Path/to/dataset.xslx', download_data = False)
        > cell_markers = ['CD20', 'CD4', 'CD80']
        > scored_ontology_terms, scored_cell_names = marker2cell(cell_markers)
    
    TODOS:
        * Make Symbols, UNIPROTIDs avialable for identification.
        * Collect Gene Symbols for which no match could be found.
        * Use cell ontology ids to aquire exact discription of the function
        * Add Chisquare testing (with p-value adjustment) as additional statistic
        * Add summary functionality
        * Expand _download to deal with situation where the folder possibly already exsists.
    """
    
    CELL_MARKER_DATABASE_URL = 'http://117.50.127.228/CellMarker/CellMarker_download_files/file/Cell_marker_All.xlsx'
    
    def __init__(self, data_path:str or None = None, download_data:bool = True) -> None:
        """ 
        Args:
            data_path: str or None
                File path for the CellMarker2.0 Database in xlsx format or None,
                if the file should be downloaded.
            
            download_data: bool
                Indicating if the databse of CellMarker2.0 should be automaticaly 
                download.
                Default is True.
        
        """

        self.data_path = data_path
        self.download_data    = download_data
        self.cell_marker_data = None
        self.used_data = None
        
        if self.data_path:
            self.cell_marker_data = pandas.read_excel(self.data_path)
        elif download_data:
            self._download()
        else:
            warnings.warn(UserWarning("No data path was provided. Please set download_data to True or a path in data_path."))    
    
    def get_path(self) -> str:
        """Returns the path """
        return self.data_path
    
    def reload_data(self) -> None:
        self.cell_marker_data = pandas.read_excel(self.data_path)
          
    def _download(self) -> None:
        """Downloads the cell maker database and saves it localy.
        
        Downloads the cellmarker databse from 
        'http://117.50.127.228/CellMarker/CellMarker_download_files/file/Cell_marker_All.xlsx'
        and stores in the current directory under './_cell_marker_data/Cell_marker_All.xlsx', 
        creating a new directory cell_marker_data. 
        """

        path = './_cell_marker_data/Cell_marker_All.xlsx'
        self.cell_marker_data = pandas.read_excel(self.CELL_MARKER_DATABASE_URL)
        mkdir('_cell_marker_data')
        self.cell_marker_data.to_excel(path)
        self.data_path = path
    
    def _get_data(self) -> pandas.DataFrame:
        return self.cell_marker_data

    def _get_used_data(self) -> pandas.DataFrame:
        """ Retuns the data after pre-selecting the spiecies and cell_name. 
        Returns None if __call__ method has not been called yet.
        """
        return self.used_data
    
    def __call__(self, identifiers:list, cell_type:str = 'Normal cell', species:str = 'Human') -> tuple:
        """ Collects all ontology terms that are associated with the provided identifiers.
        
        Based on the provided indetifiers all ontologies terms and cell names 
        from the CellMarker2 database are collected an weighted by a
        score [1]. The score is callculated by score = n/m where n
        is the number of identifiers associated with an ontology term 
        or cell name and m the total number of markers in this ontology.
        Furthermore, a weighted score is computed by dividing the weighted score
        by the expected score. Cell ontologies terms or names with more hits than
        expected will therefore have a weighted score >1. 
        
        
        Args:
            identifiers: (list of strings) 
                List of identifiers in the form markers.
                (e.g. ['BRCA', 'CD20']).
            
            cell_type: str
                Defines which cell types should be used for the analysis.
                Accepted inputs are 'Normal cell', 'Cancer cell' and 'all'.
                Default is 'Normal cell'
            
            species: str
                Defines the species that should be used for the analysis,
                Accepted options for 'human', 'mouse' and 'all'.
                Default is 'human'.
        
        Returns:
            tuple of pandas DataFrames containing the raw, weighted, expected
            score with the respective cell ontology (or cell name) as keys sorted in 
            decreasing order by raw score starting from the highest scored 
            cell ontology in the first row
            
        
        Expcetions:
            ValueError if cell_type argument is not "Cancer cell", "Normal cell" or
                "all".
            
            ValueError if species argument is not "human", "mouse" or "all".
        
        
        """
        
        if cell_type not in ['Cancer cell', 'Normal cell', 'all']:
            raise ValueError(
                """cell_type argument must be "Cancer cell", "Normal cell" or 
                "all" not {}.""".format(cell_type))
            
        if species not in ['Human', 'Mouse', 'all']:
            raise ValueError(
                """species argument must be "human", "mouse" or "all" not 
                {}.""".format(species))
        
        # Preselect the prefereed cell type from the data.
        if cell_type != 'all':
            data = self.cell_marker_data[self.cell_marker_data['cell_type'] == cell_type]
        else:
            data = self.cell_marker_data
    
        # Preselect the preferred species from the data
        if species != 'all':
            data = data[data['species'] == species]
        else:
            pass
        
        self.used_data = data
        
        # Collect all rows with corresponding ontology terms/cell names with matching
        # identifiers.
        row_idx = []
        for identifier in identifiers:
            row_idx = row_idx + data[data['marker'] == identifier].index.to_list()
        
        # Obtain all unique row indices (cell ontologies terms/ cell names), 
        # since doubles may have been collected.
        unique_idx = list(set(row_idx))
        
        # Compute the ontology score for each cell ontology term (CL) by obtaining
        # the number of all the terms that were associated/matched with 
        # the current identifiers and respective number of markers associated
        # with each ontology term/cell name. 
        matched_ontology_terms = data.loc[unique_idx]['cellontology_id'].value_counts() 
        total_ontology_terms   = data['cellontology_id'].value_counts()           
        
        ontology_term_raw_score = matched_ontology_terms/total_ontology_terms[matched_ontology_terms.keys()]
        scored_ontology_terms   = pandas.DataFrame(ontology_term_raw_score)
        scored_ontology_terms.columns = ['raw_score']

        # The expected ontology term is computed by number of provided indetifiers A
        # and and total number of markers B multipled by the number of markers matched
        # with an ontology term N. Expected_score = A/B * N.
        n_identifers = len(identifiers)
        n_all_marker = len(set(data['marker']))
        expected_ontology_score = n_identifers/n_all_marker * total_ontology_terms[matched_ontology_terms.keys()]
        scored_ontology_terms['weighted_score'] = (matched_ontology_terms/total_ontology_terms[matched_ontology_terms.keys()])/expected_ontology_score
        scored_ontology_terms['expected_score'] = expected_ontology_score
        scored_ontology_terms = scored_ontology_terms.sort_values('raw_score', ascending = False)
        
        # Compute the score for each matched cell name (e.g. Macrophage).
        matched_cell_names = data.loc[unique_idx]['cell_name'].value_counts()
        total_cell_names   = data['cell_name'].value_counts()
    
        cell_names_raw_scores = matched_cell_names/total_cell_names[matched_cell_names.keys()]
        scored_cell_names = pandas.DataFrame(cell_names_raw_scores)
        scored_cell_names.columns = ['raw_score']

        expected_cell_names_score = n_identifers/n_all_marker * total_cell_names[matched_cell_names.keys()]
        scored_cell_names['weighted_score'] = (matched_cell_names/total_cell_names[matched_cell_names.keys()])/expected_cell_names_score
        scored_cell_names['expected_score'] = expected_cell_names_score
        scored_cell_names = scored_cell_names.sort_values('raw_score', ascending = False)
        
        return scored_ontology_terms, scored_cell_names