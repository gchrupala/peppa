import pandas as pd
import statsmodels.formula.api as api
from sklearn.preprocessing import scale
from plotnine import *

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
        

def plot_coef(fit, fragment_type):
    data = fit.reset_index().rename(
        columns={'index': 'Variable', 'Coef.': 'Coefficient', '[0.025': 'Lower', '0.975]': 'Upper'})
    g = ggplot(data, aes('Variable', 'Coefficient')) + \
        geom_hline(yintercept=0, color='gray', linetype='dashed') + \
        geom_errorbar(aes(color='Trained', ymin='Lower', ymax='Upper', lwd=1, width=0.25)) + \
        geom_point(aes(color='Trained')) + \
        coord_flip() 
    ggsave(g, f"results/grsa_{fragment_type}_coef.pdf")

def load(path):
    rawdata = pd.read_csv(path)
    return massage(rawdata_d)



def main():
    # Load and process data
    
    rawdata_d = pd.read_csv('pairwise_similarities_dialog.csv')
    data_d = massage(rawdata_d)
    rawdata_n = pd.read_csv('pairwise_similarities_narration.csv')
    data_n = massage(rawdata_n)
    

    m_d = api.ols(formula = 'similarity ~ glovesim + distance + durationdiff + sametype + samespeaker + sameepisode', data=data_d)
    m_d_init = api.ols(formula = 'similarity_init ~ glovesim + distance + durationdiff + sametype + samespeaker + sameepisode', data=data_d)
    table = m_d.fit().summary2().tables[1]
    table['Trained'] = True
    table_init = m_d_init.fit().summary2().tables[1]
    table_init['Trained'] = False
    table_d = pd.concat([table, table_init]).reset_index()
    table_d.to_csv("results/coef_d.csv", index=True, header=True)
    
    plot_coef(table_d, "dialog")
    
    m_n = api.ols(formula = 'similarity ~ glovesim + distance + durationdiff + sametype + sameepisode', data=data_n)
    m_n_init = api.ols(formula = 'similarity_init ~ glovesim + distance + durationdiff + sametype + sameepisode', data=data_n)
    table = m_n.fit().summary2().tables[1]
    table['Trained'] = True
    table_init = m_n_init.fit().summary2().tables[1]
    table_init['Trained'] = False
    table_n = pd.concat([table, table_init])
    table_n.to_csv("results/coef_n.csv", index=True, header=True)
    plot_coef(table_n, "narration")
    
    



