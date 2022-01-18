# Image and graph based microbiome representation for host classification
This code is attached to the paper Image and graph based microbiome representation for host classification.
We suggest two novel methods to combine information from different bacteria and improve data representation for machine learning using  bacterial taxonomy. 
# iMic
iMic translates the values and structure of the taxonomy tree to images and then apllies CNN on the images.
For full explanation see our paper.
microbiome2matrix - translate the ASVs to images
nni_data_loader - load the dataset for the learning.
main_nni_runner_tt - perform the learning itself. The parameters it has to get are:
model: {naeive , cnn1, cnn2}
name of dataset: {IBD, Male_vs_female, Cirrhosis_no_virus, new_allergy_milk, new_allergy_peanuts, new_allergy_nut, nugent, white_vs_black_vagina}
D_mode : {"1D","IEEE","dendogram"}
tag : Most of the datasets do not need a special tag, but in IBD dataset we have several tags: {CD, IBD}





# gMic and gMic + v
gMic uses only the structure of the taxonomy tree of the samples, ignoring the the abundances of the bacterias.
The tree structure is learnt via GCN layers.

gMic + v combines the structure of the graph and the abundances of the bacterias in order to improve learning performances.
