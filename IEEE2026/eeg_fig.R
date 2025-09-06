df <- read.csv('C:/Users/Milton/PycharmProjects/neurohumanities-lab/OfflineProcessing/PSD_TAB_norm.csv')

print(df)


png('fig2.png')
barplot(df$Power, ylim = c(-1, 3), main = 'Z-score scaling')
dev.off()
