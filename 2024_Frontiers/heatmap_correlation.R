# df_corr
library(lattice)
library(RColorBrewer)

df3 <- read.csv('C:/Users/Milton/PycharmProjects/NHLabs/datos/df_corr_3.csv', row.names='X')

print(df)

heatmap(as.matrix(df2))


png('heatmapNHLab.png', width = 480)

paleta <- colorRampPalette(brewer.pal(9, "Greys")) # terrain.colors(n), magma(n)
p <- levelplot(t(as.matrix(df3)), ylab='Lobule + Spectral band', xlab = 'Emotion',
          main = 'Mean |r| Pearson correlation, iterated through each subject',
          aspect = 'fill', col.regions=paleta)
print(p)

dev.off()