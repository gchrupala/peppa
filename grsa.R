library(dplyr)
library(tidyr)
library(knitr)
library(sensemakr)

massage <- function (dat) {
    return (dat %>% drop_na() %>% filter(glovesim != 0.0) %>%                  
               mutate(samespeaker = ifelse(samespeaker=='True', 1, -1),
                      sametype = ifelse(sametype=='True', 1, -1),
                      sameepisode = ifelse(sameepisode=='True', 1, -1),
                      glovesim=scale(glovesim),
                      distance=scale(distance),
                      durationdiff=scale(durationdiff),
                      similarity=scale(similarity)))
}


rawdata.d <- read.csv('pairwise_similarities_dialog.csv')
m.d <- lm(formula = similarity ~ glovesim + distance + durationdiff + sametype + samespeaker + sameepisode, data = massage(rawdata.d))

table.d <- cbind(coeff=summary(m.d)$coeff, partial_r2=partial_r2(m.d))
print(table.d %>% kable(format='latex', booktabs=TRUE, digits=c(3,3,3,3,3)))

rawdata.n <- read.csv('pairwise_similarities_narration.csv')
m.n <- lm(formula = similarity ~ glovesim + distance + durationdiff + sametype + sameepisode, data = massage(rawdata.n))

table.n <- cbind(coeff=summary(m.n)$coeff, partial_r2=partial_r2(m.n))
print(table.n %>% kable(format='latex', booktabs=TRUE, digits=c(3,3,3,3,3)))

