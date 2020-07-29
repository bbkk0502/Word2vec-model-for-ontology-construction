# Word2vec-model-for-ontology-construction 

This repo is a collection of language model development files used for master thesis 'Ontology construction for Demonstration Environments: A Text Mining Approach'. This thesis specifically used Skip-G model in Gensim Word2vec.

# Author and Affiliation
These codes are developed by Nie Wei a MPhil student at the Institute for Manufacturing, University of Cambridge. Email: wn230@cam.ac.uk

# Project Introduction
This thesis aims to develop an ontology for Demonstration Environments by using a semi-automatic approach: text mining. The developed ontology is available at https://github.com/bbkk0502/Demonstration-Environments-Ontology-Version1.0

Demonstration Environments refer to all facilities and environments for technology development including testing, prototyping and confronting technology with different usage situations (e.g. testbed, pilot plant, living lab, sandbox, etc.)

# File Introduction
 * DEOntologyTextMining is the Python project for language model development. It consists of:
    * word2vec.py: The first python script to be executed. In this script, collected texts are pre-processed to build the text corpus, the text corpus is then used to train the word2vec model. 
    * algorithm.py: The second python script to be executed. In this script, a set of algorithms are defined to use the trained word2vec model for extracting ontological components from text.
    * data and pdf folder stored samples of downloaded journal articles. Due to file size limitation, full list of articles can be requested through wn230@cam.ac.uk
     * data: It's a folder storing full-texts of journal articles downloaded using Elsevier's API, files are stored in JSON format. JSON files of journal articles can be fetched by using Elsevier's TDM API (https://github.com/ElsevierDev/elsapy), an API is needed (check https://www.elsevier.com/en-gb/about/open-science/research-data/text-and-data-mining)
     * pdf: It's a folder storing full-texts of journal articles downloaded from other publishers' websites, files are stored in txt format.
 * extractTXT.sh is a shell script for batch extracting texts from PDF files stored in a folder. For usage, put this script in the target folder, and use Terminal (MAC OS) or Command-line (Windows) to run the script.
 * ConceptCluster.xlsx is a spreadsheet storing concept hierarchy build in Algorithm 3 (see algorithm.py in DEOntologyTextMining foler.
 * skipg.model is the trained SKIP-G model that was used in the thesis. This can be loaded for direct use (see algorithm.py in DEOntologyTextMining foler for usage)

# Feedback
Please report bugs or send feedbacks to wn230@cam.ac.uk. 
