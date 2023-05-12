

setwd('/data2/jhlee/project/samsung/subnetwork/20200807/hnn')

# Load packages
if(!require(pacman)) install.packages('pacman')
pacman::p_load(tidyverse, igraph)

# Some useful functions
rename <- dplyr::rename
select <- dplyr::select
filter <- dplyr::filter


###############################################################################
# User-specified variables
# - edge_list_file (at least 3 columns) : Source, Target, Relation(+/-), ...
# - custom_gene_score_file (2 columns): gene_symbol, score
###############################################################################

network_name <- 'network_200806'
edge_list_file <- 'GRNSN_Final.txt'
remove_self_edge <- TRUE

anova_result_file <- '../ANOVA/ANOVA_results.csv'
geneset_file <- 'KEGG_ERBB_SIGNALING_PATHWAY.gmt'
# geneset_file <- NA
geneset_default_score <- 1

out_folder <- './data'



##########################
# Generate network files #
##########################

# Read edge_list file
edge_list <- read_delim(edge_list_file, delim='\t', col_names=T) %>%
  select(1:3) %>%
  rename(source=Source, sign=Relation, target=Target) %>%
  mutate(sign=ifelse(sign=='+', 1, -1))

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
anova_scores <- read_delim(anova_result_file, delim=',', col_names=T) %>%
  mutate(gene_symbol=str_replace(FEATURE, '_mut$|_loss$|_gain$|_up$|_dn$', '')) %>%
  mutate(score=-log10(ANOVA_FEATURE_FDR*0.01)) %>%
  group_by(gene_symbol) %>%
  group_modify(~{
    tibble(score=max(.x$score))
  }) %>% ungroup %>%
  filter(gene_symbol %in% node_names)

if(!is.na(geneset_file)) {
  geneset_scores <- read_delim(geneset_file, delim='\t', col_names=F) %>% t %>% as_tibble %>%
    rename(gene_symbol=V1) %>%
    filter(gene_symbol %in% node_names) %>%
    mutate(score=geneset_default_score)
  
  node_scores <- anova_scores %>% bind_rows(geneset_scores) %>%
    group_by(gene_symbol) %>%
    group_modify(~{
      tibble(score=max(.x$score))
    }) %>% ungroup
} else {
  node_scores <- anova_scores
}



######################
# Write output files #
######################
if(!dir.exists(file.path(out_folder))) {
  dir.create(file.path(out_folder))
}

write_delim(edge_list_lcc, path=file.path(out_folder, network_name, '_edge_list.tsv'), delim='\t', col_names=F)
write_delim(node_tb, path=file.path(out_folder, network_name, '_index_gene.tsv'), delim='\t', col_names=F)
write_delim(edge_list_lcc_indices, path=file.path(out_folder, network_name, '.tsv'), delim='\t', col_names=F)
write_delim(node_scores, path=file.path(out_folder, 'scores.tsv'), delim='\t', col_names=F)

