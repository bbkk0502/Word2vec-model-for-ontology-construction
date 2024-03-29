# Word2vec-model-for-ontology-construction 

This repo is a collection of language model development files used for the MPhil thesis 'Ontology construction for Demonstration Environments: A Text Mining Approach'. This thesis specifically used the Skip-G model in Gensim Word2vec.

# Project Introduction
This thesis aims to develop an ontology for Demonstration Environments(DE) by using a semi-automatic approach: text mining. The developed ontology is available at https://github.com/bbkk0502/Demonstration-Environments-Ontology-Version1.0

Demonstration Environments refer to all facilities and environments for technology development including testing, prototyping and confronting technology with different usage situations (e.g. testbed, pilot plant, living lab, sandbox, etc.)

# File Introduction
 * DEOntologyTextMining is a Python project for language model development. It consists of:
    * word2vec.py: The first Python script to be executed. In this script, collected texts are pre-processed to build the text corpus, the text corpus is then used to train the word2vec model. 
    * algorithm.py: The second Python script to be executed. In this script, a set of algorithms[1] are defined to use the trained word2vec model for extracting ontological components from the text.
    * data and pdf folder stored samples of downloaded journal articles. 
     * data: It's a folder storing full texts of journal articles downloaded using Elsevier's API, files are stored in JSON format. JSON files of journal articles can be fetched by using Elsevier's TDM API (https://github.com/ElsevierDev/elsapy), an API is needed (check https://www.elsevier.com/en-gb/about/open-science/research-data/text-and-data-mining)
     * pdf: It's a folder storing full-texts of journal articles downloaded from other publishers' websites, files are stored in txt format.
 * extractTXT.sh is a shell script for batch extracting texts from PDF files stored in a folder. For usage, put this script in the target folder, and use Terminal (MAC OS) or Command-line (Windows) to run the script.
 * skipg.model is the trained SKIP-G model that was used in the thesis. This can be loaded for direct use (see algorithm.py in DEOntologyTextMining foler for usage)
 * wod2vec_visualisation.ipynb is a Python notebook for similar words visualisation using TSNE plot.
 * ConceptCluster.xlsx is the spreadsheet storing concept hierarchy build in Algorithm 3 (see algorithm.py in DEOntologyTextMining foler.
 * list_of_journal_articles_7JUN2020.xlsx is the spreadsheet storing a full list of search records downloaded from Web of Science with a set of DE-related keywords. This record was downloaded on 7 Jun 2020. However, not all articles listed inside were included in the text corpus in this study due to reasons like failure to extract texts from downloaded pdf, the article is not a peer-reviewed journal article, etc. 
 

# Reference
[1]N. Mahmoud, H. Elbeh and H. M. Abdlkader, "Ontology Learning Based on Word Embeddings for Text Big Data Extraction," 2018 14th International Computer Engineering Conference (ICENCO), Cairo, Egypt, 2018, pp. 183-188, doi: 10.1109/ICENCO.2018.8636154.
