library(dplyr)
library(tidyr)
data <- read.csv('pairwise_similarities.csv')
data <- data %>% drop_na() %>% mutate(samespeaker = ifelse(samespeaker=='True', 1, -1),
                      	     sametype = ifelse(sametype=='True', 1, -1),
             	             sameepisode = ifelse(sameepisode=='True', 1, -1),
			     speakerpair = paste(speaker1, "+", speaker2),
			     episodepair = paste(episode1, "+", episode2),
                             distance=scale(distance),
			     durationdiff=scale(durationdiff),
                             similarity=scale(similarity)) 

m <- lm(formula = similarity ~ distance + sametype + samespeaker + sameepisode + durationdiff, data = data)

summary(m)