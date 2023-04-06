# Single Cell to Cell Ontology - a BioCypher Knowledge Graph
A BioCypher-driven knowledge graph pipeline for mapping and harmonising single
cell experiments.

## ⚙️ Installation (local, for docker see below)
1. Clone this repository and install the dependencies using
[Poetry](https://python-poetry.org/). (Or feel free to use your own dependency
management system. We provide a `pyproject.toml` to define dependencies.)

```{bash}

git clone https://github.com/biocypher/sc2cl.git
cd sc2cl
poetry install

```
2. You are ready to go!

```{bash}

poetry shell
python create_knowledge_graph.py

```

## 🐳 Docker

This repo also contains a `docker compose` workflow to create the example
database using BioCypher and load it into a dockerised Neo4j instance
automatically. To run it, simply execute `docker compose up -d` in the root 
directory of the project. This will start up a single (detached) docker
container with a Neo4j instance that contains the knowledge graph built by
BioCypher as the DB `docker`, which you can connect to and browse at 
localhost:7474 (don't forget to switch the DB to `docker` instead of the 
standard `neo4j`). Authentication is set to `neo4j/neo4jpassword` by default
and can be modified in the `docker_variables.env` file.

By using the `BIOCYPHER_CONFIG` environment variable in the Dockerfile, the
`biocypher_docker_config.yaml` file is used instead of the 
`biocypher_config.yaml`. Everything else is the same as in the local setup. The
first container installs and runs the BioCypher pipeline, and the second
container installs and runs Neo4j. The files created by BioCypher in the first
container are copied and automatically imported into the DB in the second
container.
