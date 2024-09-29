# Genome Misclassification Network Analysis

## Overview
In this work, we present a novel alignment-free AI-based approach, genome misclassification network analysis (GMNA), to conduct comparative analysis among ensembles of genome sequences. The proposed framework is a generic network-generating method based on misclassification results for correlation-based network analysis. We introduce \textit{indistinguishability}, to quantify the association between pairs of genome groups and the genetic diversity of genome ensembles. We identify the pairwise association between the target outcome and the predicted outcome, which is then utilized to design a data-driven framework that can incorporate any state-of-the-art AI models for comparative genome analysis. We showed using more than 500,000 SARS-CoV-2 genomes that by employing GMNA, associations between sequences and geographic sampling location could be uncovered with limited computation resources.
Using our framework GMNA, we showed that SARS-CoV-2 genome sequences exhibit strong geographic clustering and the results are robust and consistent under various experimental settings. Further, the genome ensemble indistinguishabilities, which could indicate genetic variation and complexity, could be correlated to the centralities of their positions in a travel network, i.e. the OAG flight network. The centrality of a region in the flight network indicates its importance in global transportation. And this result suggests that human activities impact COVID genome variation. Additionally, we considered how the average indistinguishability of the genome sequences evolve during COVID and compared it with the emergence of the variants of concerns.
To validate our proposed framework, we explored the geographic patterns of misclassified genome sequences in various classification settings (Binary and Multiclass). 

### Key Features:
- **Navie Bayes and CNN COVID Genome Classifiers**
- **Configuration Models**: Compares true misclassification networks with configuration models to validate observed spatial dependencies.
- **Community Detection**: Employs Louvain algorithm to identify community structure in the misclassification networks.
- **Geographic Visualization**: Visualizes misclassification results to highlight regional genome similarities and spatial dependencies.

---
