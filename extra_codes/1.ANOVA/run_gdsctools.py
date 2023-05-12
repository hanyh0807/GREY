from gdsctools import IC50, GenomicFeatures, DrugDecode, ANOVA, ANOVAReport


dd = DrugDecode('gdsctools_input/drug_decode.csv')

# All cells
ic50 = IC50('gdsctools_input/drug_IC50.csv')
gdsc = ANOVA(ic50, 'gdsctools_input/genome_features.csv', dd) #gf = GenomicFeatures('gdsctools_input/genome_features.csv')
gdsc.settings.FDR_threshold = 15
results = gdsc.anova_all()
ar = ANOVAReport(gdsc, results)
ar.df.to_csv('./data/ANOVA_results.csv', index=False)
