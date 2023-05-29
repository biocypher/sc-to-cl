# ChatGSE user ontology mapper class
# manage the ontology mapping

import json
import os
from loguru import logger
import pandas as pd
import streamlit as st
from chatgse._llm_connect import GptConversation, BloomConversation
import text2term
import string

ss = st.session_state


# ENVIRONMENT VARIABLES


class OntologyMapper:
    def __init__(self):
        if "terms" not in ss:
            ss.terms = ""

    @staticmethod
    def _render_msg(role: str, msg: str):
        return f"`{role}`: {msg}"

    #TODO: Umschreiben um Tabellen zuzulassen???
    def _show_Mapping(self, role: str, df: pd.DataFrame):
        logger.info(f"Mapping Terms from {role}: {df}")
        st.markdown(self._render_msg(role, df))
        

    def set_ontologies(self, model_name: str):
        """
        Set the text2term ontology to use for the mapping.
        """
        #TODO: Expand the method to use more than one ontology --> May a list where you can select the ontologies you want to use
        #TODO: Implement an file import so that you can map terms in files

        logger.info("Caching the Cell Ontology")

        text2term.cache_ontology("http://purl.obolibrary.org/obo/cl.owl", "CL")



    #TODO uschreiben
    def _get_mapping(self):
        logger.info("Getting Mapping from text2term.")

        #extracting the terms and put them into a list
        #TODO: possibility to get the words from a file
        terms = ss.terms.str.replace('[{}]'.format(string.punctuation), '').tolist()

        #map the terms
        #TODO: Optional possibility to save the mappings
        result = text2term.map_terms(terms, "CL", base_iris="http://purl.obolibrary.org/obo/CL",  use_cache=True)

        self._write_and_history("💬🧬 text2term", result)

        return result
