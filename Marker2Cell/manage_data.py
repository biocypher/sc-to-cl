import pandas
import warnings

class DataManager:
    """Base class for manging requests/api calls."""

    def __init__(self) -> None:

        self.url = 'NotImplmented'
        self.used_urls = []
    
    def get_url(self) -> str:
        return self.url
    
    def get_used_url(self) -> str:
        return self.used_urls
    
    def create_request_url(self, *args) -> str:
        return self.url.format(*args)
    
    def get(self):
        raise NotImplementedError()


class HumanProteinAtlasManager(DataManager):
    """ Requests the Human Protein Atlas API. 
    
    HumanProteinAtlasManager generates querries to request the Human Protein
    Atlas API, returning the result as pandas.DataFrame.
    The request should contain a gene identifier (e.g. MS4A1) and a the desired 
    entries. Please notice that in the current version only one gene at a time 
    can be requested. 

    Example:
    > manager = HumanProteinAtlasManager()
    > data    = manager.get('MS4A1', 'g,gs,rnatss,rnatsm')
    
    TODOS:
        * Enable bulck requests.
    """

    def __init__ (self):
        super(HumanProteinAtlasManager, self).__init__()
        self.url = 'www.proteinatlas.org/api/search_download.php?search={}&format=json&columns={}&compress=no'
    
    def get(self, *args) -> pandas.DataFrame:

        url = self.create_request_url(*args)
        self.used_urls.append(url)            # Store all past used, request urls.
        data_response = pandas.read_json(url)

        # In several situations no data might be returned, although the API-call
        # runs without generating an error. To inform the user about a such a 
        # situation with raise this warning.
        if data_response.empty:
            warnings.warn('No data was retriewed. Please check youre input.')

        return data_response


class UniProtManager(DataManager):
    """ Requests the UniProt API. """

    def __init__(self):
        super(UniProtManager, self).__init__()
        pass