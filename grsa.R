library(dplyr)
library(tidyr)
library(knitr)
library(sensemakr)

massage <- function (dat) {
    return (dat %>% select(samespeaker, sameepisode, sametype, glovesim, distance,
            durationdiff, similarity, similarity_init) %>%
	    drop_na() %>%
	    filter(glovesim != 0.0) %>%                  
               mutate(samespeaker = ifelse(samespeaker=='True', 1, -1),
                      sametype = ifelse(sametype=='True', 1, -1),
                      sameepisode = ifelse(sameepisode=='True', 1, -1),
                      glovesim=scale(glovesim),
                      distance=scale(distance),
                      durationdiff=scale(durationdiff),
                      similarity=scale(similarity),
		      similarity_init=scale(similarity_init)))
}


rawdata.d <- read.csv('pairwise_similarities_dialog.csv')
data.d <- massage(rawdata.d)
m.d <- lm(formula = similarity ~ glovesim + distance + durationdiff + sametype + samespeaker + sameepisode, data = data.d)
print(summary(m.d))

table.d <- cbind(coeff=summary(m.d)$coeff, partial_r2=partial_r2(m.d))
print(table.d %>% kable(format='latex', booktabs=TRUE, digits=c(3,3,3,3,3)))

rawdata.n <- read.csv('pairwise_similarities_narration.csv')
data.n <- massage(rawdata.n)
m.n <- lm(formula = similarity ~ glovesim + distance + durationdiff + sametype + sameepisode, data = data.n)

table.n <- cbind(coeff=summary(m.n)$coeff, partial_r2=partial_r2(m.n))
print(table.n %>% kable(format='latex', booktabs=TRUE, digits=c(3,3,3,3,3)))

