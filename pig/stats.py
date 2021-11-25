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

def partial_r2(model, data):
    r2 = []
    mse_full = model.fit().mse_resid
    predictors = [ name for name in model.exog_names if name != 'Intercept' ]

    # drop intercept
    
    mse_red = model.from_formula(f"{model.endog_names} ~ {' + '.join(predictors)}",
                                 drop_cols=['Intercept'],
                                 data=data).fit().mse_resid
    r2.append((mse_red - mse_full) / mse_red)
    for predictor in predictors:
        exog = ' + '.join([ name for name in predictors if name != predictor ])
        formula = f"{model.endog_names} ~ {exog}"
        mse_red = model.from_formula(formula, data).fit().mse_resid
        r2.append((mse_red - mse_full) / mse_red)
    return pd.DataFrame(index=['Intercept']+predictors, data=dict(partial_r2=r2))
        
        
def main():
    # Load and process data
    
    rawdata_d = pd.read_csv('pairwise_similarities_dialog.csv')
    data_d = massage(rawdata_d)
    rawdata_n = pd.read_csv('pairwise_similarities_narration.csv')
    data_n = massage(rawdata_n)
    

    # Print variable correlations
    print(data_d[['glovesim', 'distance', 'durationdiff', 'similarity', 'similarity_init']].corr().to_latex(float_format="%.2f"), file=open("results/cor_dialog.tex", "w"))
    print(data_n[['glovesim', 'distance', 'durationdiff', 'similarity', 'similarity_init']].corr().to_latex(float_format="%.2f"), file=open("results/cor_narration.tex", "w"))
    
    # Fit models
    m_d = api.ols(formula = 'similarity ~ glovesim + distance + durationdiff + sametype + samespeaker + sameepisode', data=data_d)
    m_d_init = api.ols(formula = 'similarity_init ~ glovesim + distance + durationdiff + sametype + samespeaker + sameepisode', data=data_d)
    
    print(m_d.fit().summary2().tables[1].join(partial_r2(m_d, data_d)).to_latex(float_format="%.3f"), file=open("results/lm_dialog.tex", "w"))
    print(m_d_init.fit().summary2().tables[1].join(partial_r2(m_d_init, data_d)).to_latex(float_format="%.3f"), file=open("results/lm_init_dialog.tex", "w"))
    
    
    m_n = api.ols(formula = 'similarity ~ glovesim + distance + durationdiff + sametype + sameepisode', data=data_n)
    m_n_init = api.ols(formula = 'similarity_init ~ glovesim + distance + durationdiff + sametype + sameepisode', data=data_n)
    
    print(m_n.fit().summary2().tables[1].join(partial_r2(m_n, data_n)).to_latex(float_format="%.3f"), file=open("results/lm_narration.tex", "w"))
    print(m_n_init.fit().summary2().tables[1].join(partial_r2(m_n_init, data_n)).to_latex(float_format="%.3f"), file=open("results/lm_init_narration.tex", "w"))


