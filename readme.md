# Protein-GraphRAG

## Install

**Install from source** (recommend)

```shell
# clone this repo first
cd protein-graphrag
pip install -e .
```

## Quick Start

If you first time to use this tool, you need to download the data and preprocess the protein-go knowledge graph.

```shell
gdown 'https://drive.google.com/uc?id=1iTC2-zbvYZCDhWM_wxRufCvV6vvPk8HR'
# ProteinKG25.zip
unzip ProteinKG25.zip
python proteinKG.py --prepare_kg_data
# Preprocessed KG will take about 500 MB with pickle.
```

Search for the relationship between a protein with ID 309507 and a GO term with ID 219.
```shell
# fill in your LLM API key in proteinKG.py first
python proteinKG.py --protein_node_id 309507 --go_node_id 219
```

The LLM will generate the relationship between the protein and the GO term.
```
False. PROTEIN P00309 does not belong to the GO:0000322 class.
```


