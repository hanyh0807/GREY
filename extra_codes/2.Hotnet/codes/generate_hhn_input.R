

setwd('/data2/jhlee/project/samsung/subnetwork/20200807/hnn')

if(!require(tidyverse)) install.packages('tidyverse')
if(!require(igraph)) install.packages('igraph')
library(tidyverse)
library(igraph)


# Some useful functions
rename <- dplyr::rename
select <- dplyr::select
filter <- dplyr::filter


###############################################################################
# User-specified variables
# - edge_list_file (at least 3 columns) : Source, Target, Relation(+/-), ...
# - custom_gene_score_file (2 columns): gene_symbol, score
###############################################################################

geneset_default_score <- 1
edge_list_file <- 'GRNSN_Final.txt'
network_name <- 'GRNSN_Final_1'

# geneset_default_score <- 0.5
# edge_list_file <- 'GRNSN_Final.txt'
# network_name <- 'GRNSN_Final_0.5'

# geneset_default_score <- 1
# edge_list_file <- 'EdwinWang.txt'
# network_name <- 'EdwinWang_1'

# geneset_default_score <- 0.5
# edge_list_file <- 'EdwinWang.txt'
# network_name <- 'EdwinWang_0.5'



out_folder <- paste(network_name, '_data', sep='')

anova_result_file <- '../ANOVA/ANOVA_results.csv'
geneset_file <- 'KEGG_ERBB_SIGNALING_PATHWAY.gmt'
remove_self_edge <- TRUE




##########################
# Generate network files #
##########################

# Read edge_list file
edge_list <- read_delim(edge_list_file, delim='\t', col_names=T) %>%
  select(1:3) %>%
  rename(source=Source, sign=Relation, target=Target) %>%
  mutate(sign=ifelse(sign=='+', 1, ifelse(sign=="-", -1, 0)))

# Remove duplicated rows
edge_list %<>% distinct(source, target, sign, .keep_all=T)

# Remove self-edges
if(remove_self_edge) {
  edge_list %<>% filter(source != target)
}

# Find largest connected components
g <- graph_from_edgelist(edge_list %>% select(source, target) %>% as.matrix)
lcc_nodes <- groups(components(g, mode='weak'))[[1]]
edge_list_lcc <- edge_list %>%
  filter(source %in% lcc_nodes & target %in% lcc_nodes) %>%
  select(source, target)

# Node indices
node_names <- sort(unique(c(edge_list_lcc$source, edge_list_lcc$target)))
node_tb <- tibble(Seq=seq_along(node_names), node=node_names)

# Edge list
edge_list_lcc_indices <- edge_list_lcc %>%
  left_join(node_tb, by=c('source'='node')) %>% select(-source) %>% rename(source=Seq) %>%
  left_join(node_tb, by=c('target'='node')) %>% select(-target) %>% rename(target=Seq)



############################
# Generate node score data #
############################

# Node score from ANOVA result
# --> GENESYMBOL, ANOVA score
# Aggregete genomic_features from ANOVA result
#   - Transform FDR for each genomic_feature into ANOVA score[-log10(ANOVA_FEATURE_FDR*0.01)]
#   - Aggregate GENESYMBOL_[mut|loss|gain|up|dn] into GENESYMBOL and retaining maximun ANOVA score among them
#   - Filter-in records with genes in network
anova_scores <- read_delim(anova_result_file, delim=',', col_names=T) %>%
  mutate(gene_symbol=str_replace(FEATURE, '_mut$|_loss$|_gain$|_up$|_dn$', '')) %>%
  mutate(score=-log10(ANOVA_FEATURE_FDR*0.01)) %>%
  group_by(gene_symbol) %>%
  group_modify(~{
    tibble(score=max(.x$score))
  }) %>% ungroup %>%
  filter(gene_symbol %in% node_names)

# Node score from Geneset
# --> GENESYMBOL, geneset_default_score
geneset_scores <- read_delim(geneset_file, delim='\t', col_names=F) %>% t %>% as_tibble %>%
  rename(gene_symbol=V1) %>%
  filter(gene_symbol %in% node_names) %>%
  mutate(score=geneset_default_score)

# Merge node scores from ANOVA result and Geneset
node_scores <- anova_scores %>% bind_rows(geneset_scores) %>%
  group_by(gene_symbol) %>%
  group_modify(~{
    tibble(score=max(.x$score))
  }) %>% ungroup



######################
# Write output files #
######################
if(!dir.exists(file.path(out_folder))) {
  dir.create(file.path(out_folder))
}

write_delim(edge_list_lcc_indices, path=file.path(out_folder, paste(network_name, '_edge_list.tsv', sep='')), delim='\t', col_names=F)
write_delim(node_tb, path=file.path(out_folder, paste(network_name, '_index_gene.tsv', sep='')), delim='\t', col_names=F)
write_delim(edge_list_lcc, path=file.path(out_folder, paste(network_name, '.tsv', sep='')), delim='\t', col_names=F)
write_delim(node_scores, path=file.path(out_folder, 'scores.tsv'), delim='\t', col_names=F)

