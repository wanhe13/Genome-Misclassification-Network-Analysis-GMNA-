# Genome Misclassification Network Analysis

## Overview
In this work, we present a novel alignment-free AI-based approach, genome misclassification network analysis (GMNA), to conduct comparative analysis among ensembles of genome sequences. The proposed framework is a generic network-generating method based on misclassification results for correlation-based network analysis. We introduce *indistinguishability*, to quantify the association between pairs of genome groups and the genetic diversity of genome ensembles. We identify the pairwise association between the target outcome and the predicted outcome, which is then utilized to design a data-driven framework that can incorporate any state-of-the-art AI models for comparative genome analysis. We showed using more than 500,000 SARS-CoV-2 genomes that by employing GMNA, associations between sequences and geographic sampling location could be uncovered with limited computation resources.
Using our framework GMNA, we showed that SARS-CoV-2 genome sequences exhibit strong geographic clustering and the results are robust and consistent under various experimental settings. Further, the genome ensemble indistinguishabilities, which could indicate genetic variation and complexity, could be correlated to the centralities of their positions in a travel network, i.e. the OAG flight network. The centrality of a region in the flight network indicates its importance in global transportation. And this result suggests that human activities impact COVID genome variation. Additionally, we considered how the average indistinguishability of the genome sequences evolve during COVID and compared it with the emergence of the variants of concerns.
To validate our proposed framework, we explored the geographic patterns of misclassified genome sequences in various classification settings (Binary and Multiclass). 

### Key Features:
- **Navie Bayes, CNN and Transformer-based COVID Genome Classifiers**
- **Configuration Models**: Compares true misclassification networks with configuration models to validate observed spatial dependencies.
- **Community Detection**: Employs Louvain algorithm to identify community structure in the misclassification networks.
- **Geographic Visualization**: Visualizes misclassification results to highlight regional genome similarities and spatial dependencies.
---

## Contributions

- **Novel Alignment-Free Framework**: Introduced the Genome Misclassification Network Analysis (GMNA), using empirical misclassification likelihoods to measure associations between genome sequence ensembles.

- **Data-Driven Approach**: Developed a framework that identifies pairwise associations between target and predicted outcomes, enabling incorporation of arbitrary AI models in comparative genome analysis.

- **Indistinguishability Metric**: Proposed a novel concept to quantify genetic diversity and complexity of genome ensembles through pairwise group associations.

- **Model Adaptability**: Created a flexible framework compatible with various AI models and genomic datasets.

- **Large-Scale Validation**: Demonstrated GMNA's effectiveness using over 500,000 SARS-CoV-2 genomes, uncovering geographic genome sequence associations with minimal computational resources.

- **Network Comparative Analysis**: Constructed a misclassification-based network and compared it with the Official Airline Guide (OAG) flights network to explore how human activities impact genome variation and evolution.

- **Leave-One-Class-Out (LOCO) Model**: Introduced a balanced approach to generate misclassified data while maintaining model credibility and high prediction accuracy.

- **Computational Efficiency**: Provided a tool for large-scale comparative genomics, offering insights into phylogenetic structure and evolutionary patterns.

## Background
Classifying genome sequences based on metadata has been an active area of research in comparative genomics for decades with many important applications across the life sciences. Established methods for classifying genomes can be broadly grouped into sequence alignment-based and alignment-free models. The more conventional alignment-based models rely on genome similarity measures calculated based on local sequence alignments or consistent ordering among sequences. However, these alignment-based methods can be quite computationally expensive when dealing with large ensembles of even moderately sized genomes. In contrast, alignment-free approaches measure genome similarity based on summary statistics in an unsupervised setting and are computationally efficient enough to analyze large datasets. However, both alignment-based and alignment-free methods typically assume fixed scoring rubrics that lack the flexibility to assign varying importance to different parts of the sequences based on prior knowledge and also that prediction errors are random with respect to the underlying data generating model. In this study, we integrate artificial intelligence and network science approaches to develop a comparative genomic analysis framework that addresses both of these limitations. Our approach, termed the Genome Misclassification Network Analysis (GMNA), simultaneously leverages misclassified instances, a learned scoring rubric, and label information to classify genomes based on associated metadata and better understand potential drivers of miss-classification. We evaluate the utility of the GMNA using Naive Bayes and convolutional neural network models, supplemented by additional experiments with transformer-based models like Enformer [Avsec et al. (2021)](https://www.nature.com/articles/s41592-021-01252-x), to construct SARS-CoV-2 sampling location classifiers using over 500,000 viral genome sequences and study the resulting network of misclassifications. We demonstrate the global health potential of the GMNA by leveraging the SARS-CoV-2 genome misclassification networks to investigate the role human mobility played in structuring geographic clustering of SARS-CoV-2 and how genomes were misclassified by our model. 


