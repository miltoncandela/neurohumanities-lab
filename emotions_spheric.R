
pdf('emo_dist.pdf')
m <- matrix(c(21.24, 13.16, 9.16, 11.24, 15.33, 29.87,
              33.80, 18.81, 7.63, 3.69, 7.08, 28.97), nrow = 6, ncol = 2)

emotions <- c('Desire', 'Admiration', 'Joy', 'Love', 'Hate', 'Sadness')
bp <- barplot(t(m), beside=TRUE, names.arg=emotions, ylab='Percentage (%)',
              col = c('grey10', 'grey90'), ylim = c(0, 40))
legend('top', legend = c('Spherical', 'Cubical'), fill = c('grey10', 'grey90'), bty = 'n')
text(x = bp, y = round(t(m), 1), labels = round(t(m), 1), pos = 3, cex = 0.8)
dev.off()



# Fear prediction window
path <- 'C:/Users/Milton/PycharmProjects/neurohumanities-lab/'
df <- read.csv(paste0(path, 'Newcode/outputs/vadmat_new_3s.csv'))

for (sub_num in unique(df$Subject)){
png(paste0('V_',sub_num, '.png'))
df <- read.csv(paste0(path, 'Newcode/outputs/vadmat_new_3s.csv'))
df <- df[df$Subject == sub_num,]
plot(df$Timestamp, df$Valence, type = 'l', lwd = 2, xlab = 'Time (s)',
     ylab = 'Fear metric', axes = FALSE, ylim = c(1, 9))
axis(1)
axis(2)

df <- read.csv(paste0(path, 'Newcode/outputs/vadmat_new_5s.csv'))
df <- df[df$Subject == sub_num,]
lines(df$Timestamp, df$Valence, type = 'l', col = 2, lwd = 2)

df <- read.csv(paste0(path, 'Newcode/outputs/vadmat_new_10s.csv'))
df <- df[df$Subject == sub_num,]
lines(df$Timestamp, df$Valence, type = 'l', col = 3, lwd = 2)
legend('topright', c('3', '5', '10'), lty = 1, lwd = 2, col = 1:3, bty = 'n')
abline(h = 0.25, col = 'grey', lty = 2)
abline(h = 0.5, col = 'grey', lty = 2)
abline(h = 0.75, col = 'grey', lty = 2)}



# Fear prediction window
path <- 'C:/Users/Milton/PycharmProjects/neurohumanities-lab/'
df <- read.csv(paste0(path, 'vadmat_new_10s.csv'))

for (sub_num in unique(df$Subject)){
     png(paste0('V2_',sub_num, '.png'))
     df <- read.csv(paste0(path, 'vadmat_new_10s.csv'))
     df <- df[df$Subject == sub_num,]
     plot(df$Timestamp, df$Valence, type = 'l', lwd = 2, xlab = 'Time (s)',
          ylab = 'Valence', axes = FALSE, ylim = c(1, 9))
     axis(1)
     axis(2)
     lines(df$Timestamp, df$Arousal, col = 2, lwd = 2)
     lines(df$Timestamp, df$Dominance, col = 3, lwd = 2)
     
     df <- read.csv(paste0(path, 'Newcode/outputs/vadmat_new_10s.csv'))
     df <- df[df$Subject == sub_num,]
     lines(df$Timestamp, df$Valence, col = 1, lwd = 2, lty = 2)
     lines(df$Timestamp, df$Arousal, col = 2, lwd = 2, lty = 2)
     lines(df$Timestamp, df$Dominance, col = 3, lwd = 2, lty = 2)
     
     legend('topright', c('V', 'A', 'D'), lty = 1, lwd = 2, col = 1:3, bty = 'n')
     legend('top', c('2','20'), lty = 1:2, lwd = 2, bty = 'n')
     
dev.off()}

