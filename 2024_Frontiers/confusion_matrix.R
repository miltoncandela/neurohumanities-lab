
library(lattice) 
library(RColorBrewer)

m1 <- matrix(c(1.13717, 1.04513, 93.7057,
               3.69581, 97.7197, 4.16667,
               95.167, 1.23515, 2.12766), nrow=3, ncol=3)

m2 <- matrix(c(0.785973, 3.86941, 95.3446,
               0.630915, 97.3712, 1.9979,
               92.6199, 4.79405, 2.58303), nrow=3, ncol=3)

m3 <- matrix(c(0.592885, 2.23979, 97.1673,
              0.477783, 98.3278, 1.19446,
              92.7114, 5.53936, 1.74927), nrow=3, ncol=3)

m <- array(c(t(m1), m2, m3), dim = c(3, 3, 3))
dimnames(m) <- list(c('Low', 'Mid', 'High'), c('High', 'Mid', 'Low'),
                    c('Valence', 'Arousal', 'Dominance'))

n <- 3
paleta <- colorRampPalette(brewer.pal(n, "Reds")) # terrain.colors(n), magma(n)

pdf('conf.pdf', onefile = FALSE, useDingbats = FALSE)
l <- levelplot(m, layout = c(3,1), ylab = 'True label', xlab = 'Predicted label',
          col.regions = paleta,
          panel = function(x, y, z, ..., subscripts=subscripts) {
               panel.levelplot(x=x, y=y, z=z, ..., subscripts=subscripts)
               panel.text(x=x[subscripts], y=y[subscripts], labels=round(z[subscripts], 2))})
print(l)
dev.off()



