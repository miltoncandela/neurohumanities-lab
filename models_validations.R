df_all <- read.csv('nhlabsmodelvalidations.csv')
df_all <- head(df_all, 16)

barplot_create <- function(o, name){
     df <- df_all[-1,c(1, 2, seq(8)*3 + o)]
     colnames(df) <- c('Model', 'VAD', 'Val-0', 'Val-1', 'Val-2', 'Val-3', 'Val-4', 'Val-5', 'Val-6', 'Val-7')
     
     m <- matrix(data=NA, nrow=15, ncol = 3)
     for (model_i in 1:length(unique(df$Model))){
          model <- unique(df$Model)[model_i]
          for (vad_i in 1:length(unique(df$VAD))){
               vad <- unique(df$VAD)[vad_i]
               
               x <- df[(df$Model == model) & (df$VAD == vad), -c(1, 2)]
               x <- as.numeric(x)
               
               m[(model_i - 1)*3 + vad_i, ] <- c(vad, mean(x), sd(x))}}
     
     models <- unique(df$Model)
     df <- data.frame(m)
     colnames(df) <- c('VAD', 'Acc', 'SD')
     df$SD <- as.numeric(df$SD)
     df$Acc <- as.numeric(df$Acc)
     
     # map_values <- function(x){
     #      if (x == 'random forest'){return('RF')}
     #      else if (x == 'extra trees'){return('ET')}
     #      else if (x == 'svc'){return('SVC')}
     #      else if (x == 'knn'){return('kNN')}
     #      else {return('XGB')}}
     
     map_values <- function(x){
          if (x == 'random forest'){return('B')}
          else if (x == 'extra trees'){return('A')}
          else if (x == 'svc'){return('E')}
          else if (x == 'knn'){return('D')}
          else {return('C')}}
     
     d <- c()
     for (model in models){d <- c(d, rep(map_values(model), 3))}
     
     df['Model'] <- d
     df['SE'] <- df$SD / sqrt(8)
     df['ulim'] <- df$Acc + df$SE
     df['llim'] <- df$Acc
     df['err'] <- ifelse(df$VAD == 'Arousal', -0.225, ifelse(df$VAD == 'Dominance', 0, 0.225))
     
     ajuste <- ifelse(o == 0, 100, 1)
     
     # # # Base approach
     library(tidyverse)
     
     m_acc <- df[,c('VAD', 'Acc', 'Model')] %>% spread(Model, Acc)
     row.names(m_acc) <- m_acc$VAD
     m_acc <- as.matrix(m_acc[,-1]) * ajuste
     
     m_sd <- df[,c('VAD', 'SE', 'Model')] %>% spread(Model, SE)
     row.names(m_sd) <- m_sd$VAD
     m_sd <- as.matrix(m_sd[,-1]) * ajuste
     
     cols <- c('gray0', 'gray50', 'gray100')
     cols <- c('lightblue3', 'gray75', 'gray100')
     
     # cols <- c('black', 'grey', 'blue')
     # cols <- c('black', 'lightblue', 'lightcoral')
     # cols <- c('darkgoldenrod2', 'dodgerblue2', 'firebrick2')
     
     bp <- barplot(m_acc, beside = TRUE, ylim = c(0.6, 1.05) * ajuste, xpd = FALSE, axes = FALSE,
                   col = cols, ylab = name, xlab = 'Model', names.arg = c('ET', 'RF', 'XGB', 'kNN', 'SVC'))
     axis(2, at = c(0.6, 0.8, 0.85, 0.9, 0.95, 1) * ajuste)
     
     arrows(bp, m_acc + 0.000001, bp, m_acc + m_sd, code = 3, angle = 90, length = 0.075, lwd = 1)
     # legend('topright', legend = c('Arousal', 'Dominance', 'Valence'), fill = cols, bty = 'n')
     #dev.off()

}

pdf('FN_Model_Perf.pdf', height = 4, width = 10)

# par(mfrow = c(1, 2), oma = c(4, 1, 1, 1), mar = c(0, 0, 0, 0))
par(mfrow = c(1, 2), oma = c(0, 0, 0, 4))
barplot_create(0, 'Accuracy (%)')
# barplot_create(1, 'Balanced Accuracy (%)')
barplot_create(2, 'F1-Score')

par(new = TRUE, fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0))
# plot(0, 0, type = 'l', bty = 'n', yaxt = 'n', xaxt = 'n')
plot.new()

legend('right', legend = c('Arousal', 'Dominance', 'Valence'), fill = c('lightblue3', 'gray75', 'gray100'), bty = 'n')
# legend('right',legend = c("Measured Force", "Predicted Force"), lwd = 2, xpd = TRUE, horiz = TRUE, lty = c(1, 2),  bty = 'n')

dev.off()

# # # Lattice approach
# library(lattice)
# 
# barchart(Acc ~ Model, df, groups = VAD, ylim = c(0.6, 1),
#          auto.key = list(space = "top", rectangles = TRUE, points = FALSE,
#                          title='VAD', cex.title= 1, columns=3),
#          ylab = 'Accuracy (%)', xlab = 'Model',
#          scales = list(y = list(at =c(0.6, 0.8, 0.85, 0.9, 0.95, 1))),
#          panel=function(x,y,..., subscripts)
#          {panel.barchart(x, y, subscripts = subscripts, ...)
#               ll = df$llim[subscripts]
#               ul = df$ulim[subscripts]
#               
#               # vertical error bars
#               panel.segments(as.numeric(x) + df$err[subscripts], ll,
#                              as.numeric(x) + df$err[subscripts], ul, col=1, lwd=1)
#            
#               # upper horizontal cap
#               panel.segments(as.numeric(x) + df$err[subscripts] - 0.1, ul, 
#                              as.numeric(x) + df$err[subscripts] + 0.1, ul, col=1, lwd=1)})
