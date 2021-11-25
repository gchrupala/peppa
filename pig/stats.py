import pandas as pd
import statsmodels.formula.api as api
from sklearn.preprocessing import scale

def sumcode(col):
    return (col * 2 - 1).astype(int)

def massage(dat):
    keep = ['samespeaker', 'sameepisode', 'sametype', 'glovesim', 'distance',
            'durationdiff', 'similarity', 'similarity_init']
    return dat[keep].dropna().query("glovesim != 0.0").assign(
        samespeaker  = lambda x: sumcode(x.samespeaker),
        sameepisode = lambda x: sumcode(x.sameepisode),
        sametype     = lambda x: sumcode(x.sametype),
        glovesim     = lambda x: scale(x.glovesim),
        distance     = lambda x: scale(x.distance),
        durationdiff = lambda x: scale(x.durationdiff),
        similarity   = lambda x: scale(x.similarity),
        similarity_init = lambda x: scale(x.similarity_init))

def partial_r2(model):
    raise NotImplementedError

# Load and process data
    
rawdata_d = pd.read_csv('pairwise_similarities_dialog.csv')
data_d = massage(rawdata_d)
rawdata_n = pd.read_csv('pairwise_similarities_narration.csv')
data_n = massage(rawdata_n)


# Fit models
m_d = api.ols(formula = 'similarity ~ glovesim + distance + durationdiff + sametype + samespeaker + sameepisode', data=data_d).fit()
m_d_init = api.ols(formula = 'similarity_init ~ glovesim + distance + durationdiff + sametype + samespeaker + sameepisode', data=data_d).fit()

print(m_d.summary2().tables[1].to_latex(float_format="%.3f"), file=open("results/lm_dialog.tex", "w"))
print(m_d_init.summary2().tables[1].to_latex(float_format="%.3f"), file=open("results/lm_init_dialog.tex", "w"))


m_n = api.ols(formula = 'similarity ~ glovesim + distance + durationdiff + sametype + sameepisode', data=data_n).fit()
m_n_init = api.ols(formula = 'similarity_init ~ glovesim + distance + durationdiff + sametype + sameepisode', data=data_n).fit()

print(m_n.summary2().tables[1].to_latex(float_format="%.3f"), file=open("results/lm_narration.tex", "w"))
print(m_n_init.summary2().tables[1].to_latex(float_format="%.3f"), file=open("results/lm_init_narration.tex", "w"))


