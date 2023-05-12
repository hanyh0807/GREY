
setwd('/data2/jhlee/project/samsung/subnetwork/20200522')

pacman::p_load(tidyverse, magrettr)

# Some useful functions
rename <- dplyr::rename
select <- dplyr::select
filter <- dplyr::filter




INPUT_DRUG_NAME <- 'Afatinib'



################################
# Load cancer gene information #
################################

# NCG6-cancer-genes
ncg <- read_delim('/data2/jhlee/project/kras_ccle_2019/data/NCG6/NCG6_tsgoncogene.tsv', delim='\t')
ncg_oncotsg <- ncg %>%
  filter( (!is.na(cgc_annotation) & cgc_annotation != 'fusion') | !is.na(vogelstein_annotation) | NCG6_oncogene != 0 | NCG6_tsg != 0) %>%
  # NGC-annotation
  mutate(oncotsg=ifelse(!is.na(NCG6_oncogene) & NCG6_oncogene == 1, 1,
                        ifelse(!is.na(NCG6_oncogene) & NCG6_tsg == 1, -1, NA))) %>%
  # Vogelstein-annotation
  mutate(oncotsg=ifelse(!is.na(oncotsg), oncotsg,
                        ifelse(!is.na(vogelstein_annotation) & vogelstein_annotation == 'Oncogene', 1,
                               ifelse(!is.na(vogelstein_annotation) & vogelstein_annotation == 'TSG', -1, NA)))) %>%
  # CGC-annotation
  mutate(oncotsg=ifelse(!is.na(oncotsg), oncotsg,
                        ifelse(!is.na(cgc_annotation) & str_detect(cgc_annotation, 'oncogene'), 1,
                               ifelse(!is.na(cgc_annotation) & str_detect(cgc_annotation, 'TSG'), -1, NA)))) %>%
  # filter(!is.na(oncotsg)) %>%
  rename(gene_symbol=symbol) %>%
  select(gene_symbol, oncotsg)

# OncoKB alterations
oncokb <- read_delim('/data2/jhlee/project/kras_ccle_2019/data/OncoKB/allAnnotatedVariants.txt', delim='\t') %>%
  select(`Hugo Symbol`, Alteration, `Mutation Effect`)


# Define cancer genes
cancer_gene_list <- unique(c(ncg_oncotsg$gene_symbol, oncokb$`Hugo Symbol`))
cancer_gene_list <- cancer_gene_list[cancer_gene_list != 'Other Biomarkers']


#########################
# Load cell information #
#########################

# model_list_20200204.csv : downloaded from Cell Model Passport
model_list <- read_delim(file='/data2/jhlee/data/cell_model_passport/model_list_20200204.csv', delim=',', col_names=T, trim_ws=T)







############################
# Load genomic alterations #
############################

#################
# Load mutation #
mut_org <- read_delim(file='/data2/jhlee/data/cell_model_passport/mutations_20191101.csv', delim=',', col_names=T)

############
# Load CNA #
cna_header <- read_delim(file='/data2/jhlee/data/cell_model_passport/cnv_gistic_20191101.csv', delim=',', n_max=3, col_names=F)
cna_model_id <- cna_header[1,-c(1,2)] %>% unlist %>% unname
cna_model_name <- cna_header[2,-c(1,2)] %>% unlist %>% unname
cna_org <- read_delim(file='/data2/jhlee/data/cell_model_passport/cnv_gistic_20191101.csv', delim=',', skip=3, col_names=F) %>%
  select(-1)
colnames(cna_org) <- c('gene_symbol', cna_model_id)

###################
# Load Expression #
cell_infor <- model_list %>%
  select(COSMIC_ID, model_id, tissue) %>%
  drop_na

gene_infor <- read_delim('/data2/jhlee/data/cell_model_passport/gene_identifiers_20191101.csv', delim=',')

# from Rodriguez et al.[Ref.???] #
load('/data2/jhlee/data/saez/tf_activities/cl_voom_batchcor_dupmerged.rdata')
load('/data2/jhlee/data/saez/tf_activities/cl_annotation.rdata')
# load('/data2/jhlee/data/saez/cl_voom_batchcor_dupmerged_KDCFgenenorm.rdata')

# Expression data
rna_tb_w_org <- t(EXPmerged) %>% as_tibble(rownames = 'mat_name')
rm(EXPmerged)



################
# Process data #
################

#######################
# Preprocess mutation #
mutations <- mut_org %>%
  filter(gene_symbol %in% cancer_gene_list) %>%
  # filter(cancer_driver)
  filter(protein_mutation != '-') %>%
  left_join(model_list%>%select(model_id, COSMIC_ID), by='model_id') %>% filter(!is.na(COSMIC_ID)) %>%
  select(gene_symbol, COSMIC_ID) %>%
  mutate(value=1) %>%
  mutate(feature=paste(gene_symbol, '_mut', sep='')) %>%
  distinct(feature, COSMIC_ID, .keep_all=T) %>%
  select(-gene_symbol)

##################
# Preprocess CNA #
cnas <- cna_org %>%
  filter(gene_symbol %in% cancer_gene_list) %>%
  pivot_longer(cols=2:ncol(.), names_to = 'model_id', values_to = 'gistic_score') %>%
  filter(abs(gistic_score) == 2) %>%
  left_join(model_list%>%select(model_id, COSMIC_ID), by='model_id') %>% filter(!is.na(COSMIC_ID)) %>%
  mutate(feature=ifelse(gistic_score > 0, paste(gene_symbol, '_gain', sep=''), paste(gene_symbol, '_loss', sep=''))) %>%
  # mutate(gene_symbol=paste(gene_symbol, '_cna', sep='')) %>%
  select(feature, COSMIC_ID) %>%
  mutate(value=1) %>%
  distinct(feature, COSMIC_ID, .keep_all=T)

##################
# Preprocess RNA #

# Expression cell_infor
exp_cell_infor <- CLI %>% as_tibble %>%
  separate(expression_matrix_name, c('N1','N2','N3'), '\\.') %>%
  mutate(mat_name=paste(N2, N3, sep='.')) %>%
  select(COSMIC_ID, mat_name) %>%
  distinct %>%
  filter(mat_name %in% rna_tb_w_org$mat_name) %>%
  left_join(cell_infor %>% select(-tissue), by='COSMIC_ID') %>%
  drop_na

# Choose only samples with COSMIC_ID
rna_tb_w <- rna_tb_w_org %>%
  left_join(exp_cell_infor %>% select(mat_name, COSMIC_ID), by='mat_name') %>%
  add_column(COSMIC_ID1=.$COSMIC_ID, .before=1) %>%
  select(-COSMIC_ID, -mat_name) %>%
  rename(COSMIC_ID=COSMIC_ID1) %>%
  drop_na

rna_tb_w_l <- rna_tb_w %>%
  pivot_longer(cols=2:ncol(.), names_to='ensembl_gene_id', values_to='value') %>%
  left_join(gene_infor %>% select(ensembl_gene_id, hgnc_symbol), by='ensembl_gene_id') %>%
  filter(!is.na(hgnc_symbol)) %>%
  select(-ensembl_gene_id) %>%
  group_by(hgnc_symbol) %>%
  group_modify(~{
    # Z-score
    m <- mean(.x$value)
    s <- sd(.x$value)
    z <- (.x$value-m)/s
    # Robust Z-score
    med <- median(.x$value)
    md <- mad(.x$value)
    rz <- (.x$value-med)/md
    # Result table
    tibble(COSMIC_ID=.x$COSMIC_ID, value=.x$value, z=z, rz=rz)
  }) %>% ungroup

rnas <- rna_tb_w_l %>%
  filter(hgnc_symbol %in% cancer_gene_list) %>%
  select(hgnc_symbol, COSMIC_ID, rz) %>%
  mutate(feature=ifelse(rz > 2, paste(hgnc_symbol, '_up', sep=''),
                        ifelse(rz < -2, paste(hgnc_symbol, '_dn', sep=''), NA))) %>%
  drop_na %>%
  mutate(value=1) %>%
  select(feature, COSMIC_ID, value) %>%
  distinct(feature, COSMIC_ID, .keep_all=T)

####################
# Genomic features #
genome_features <- mutations %>% bind_rows(cnas) %>% bind_rows(rnas) %>%
  pivot_wider(names_from='feature', values_from='value') %>%
  replace(., is.na(.), 0) %>%
  left_join(model_list%>%select(COSMIC_ID,msi_status,tissue), by='COSMIC_ID') %>%
  add_column(MSI_FACTOR=.$msi_status, .after='COSMIC_ID') %>%
  mutate(MSI_FACTOR=ifelse(!is.na(MSI_FACTOR) & MSI_FACTOR=='MSI', 1, 0)) %>%
  add_column(TISSUE_FACTOR=.$tissue, .after='MSI_FACTOR') %>%
  select(-msi_status, -tissue)
  



# Read GDSC drug response data
gdsc_drug_response <- read_xlsx('/data2/jhlee/data/gdsc/GDSC2_fitted_dose_response_15Oct19.xlsx') %>%
  mutate(COSMIC_ID=as.character(COSMIC_ID))

drug_codes <- gdsc_drug_response %>%
  select(DRUG_ID, DRUG_NAME, PUTATIVE_TARGET) %>%
  distinct(DRUG_ID, DRUG_NAME, .keep_all = T) %>%
  filter(DRUG_NAME == INPUT_DRUG_NAME)
drug_IC50 <- gdsc_drug_response %>%
  select(COSMIC_ID, DRUG_ID, LN_IC50) %>%
  filter(DRUG_ID %in% drug_codes$DRUG_ID) %>%
  mutate(DRUG_ID=paste('Drug_', DRUG_ID, '_IC50', sep='')) %>%
  pivot_wider(names_from='DRUG_ID', values_from='LN_IC50')


#############
# All cells #
#############
common_COSMIC_IDs <- intersect(genome_features$COSMIC_ID, drug_IC50$COSMIC_ID)

genome_features_c <- genome_features %>% filter(COSMIC_ID %in% common_COSMIC_IDs)
drug_IC50_c <- drug_IC50 %>% filter(COSMIC_ID %in% common_COSMIC_IDs)

# Print files
dir.create(file.path('./gdsctools_input'))
write_delim(genome_features_c, path='gdsctools_input/genome_features.csv', delim=',')
write_delim(drug_codes, path='gdsctools_input/drug_decode.csv', delim=',')
write_delim(drug_IC50_c, path='gdsctools_input/drug_IC50.csv', delim=',')

