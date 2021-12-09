import pandas as pd
import statsmodels.formula.api as api
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import RidgeCV
from plotnine import *
import torch
import numpy as np

def sumcode(col):
    return (col * 2 - 1).astype(int)

def massage(dat, scaleall=False):
    keep = ['samespeaker', 'sameepisode', 'sametype', 'glovesim', 'distance',
            'durationdiff', 'similarity', 'similarity_init']
    return dat[keep].dropna().query("glovesim != 0.0").assign(
        samespeaker  = lambda x: scale(x.samespeaker) if scaleall else sumcode(x.samespeaker),
        sameepisode = lambda x: scale(x.sameepisode) if scaleall else sumcode(x.sameepisode),
        sametype     = lambda x: scale(x.sametype) if scaleall else sumcode(x.sametype),
        glovesim     = lambda x: scale(x.glovesim),
        distance     = lambda x: scale(x.distance),
        durationdiff = lambda x: scale(x.durationdiff),
        similarity   = lambda x: scale(x.similarity),
        similarity_init = lambda x: scale(x.similarity_init))


def rer(red, full):
    return (red - full) / red

def partial_r2(model, data):
    r2 = []
    mse_full = model.fit().mse_resid
    predictors = [ name for name in model.exog_names if name != 'Intercept' ]

    # drop intercept
    
    mse_red = model.from_formula(f"{model.endog_names} ~ {' + '.join(predictors)}",
                                 drop_cols=['Intercept'],
                                 data=data).fit().mse_resid
    r2.append(rer(mse_red, mse_full))
    for predictor in predictors:
        exog = ' + '.join([ name for name in predictors if name != predictor ])
        formula = f"{model.endog_names} ~ {exog}"
        mse_red = model.from_formula(formula, data).fit().mse_resid
        r2.append(rer(mse_red, mse_full))
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


def frameit(matrix, prefix="dim"):
    return pd.DataFrame(matrix, columns=[f"{prefix}{i}" for i in range(matrix.shape[1])])


def backprobes():
    for fragment_type in ['dialog', 'narration']:
        data = torch.load(f"data/out/words_{fragment_type}.pt")
        backprobe(data['words']).to_csv(f"results/backprobe_{fragment_type}.csv",
                                        index=False,
                                        header=True)
        
def backprobe(words):
    rows = []
    embedding = frameit(scale(torch.stack([word.embedding for word in words]).cpu().numpy()),
                        prefix="emb")
    embedding_init = frameit(scale(torch.stack([word.embedding_init for word in words]).cpu().numpy()),
                             prefix="emb_init")
    glove = frameit(torch.stack([word.glove for word in words]).cpu().numpy(),
                    prefix="glove")
    speaker = pd.get_dummies([word.speaker for word in words], prefix="speaker")
    episode = pd.get_dummies([word.episode for word in words], prefix="episode")
    duration = pd.DataFrame(dict(duration=[word.duration for word in words]))

    train_ix = np.random.choice(embedding.index, int(len(embedding.index)/2), replace=False)
    val_ix   = embedding.index[~embedding.index.isin(train_ix)]

    predictors = dict(glove=glove, speaker=speaker, episode=episode, duration=duration)
    for outname, y in [('embedding', embedding), ('embedding_init', embedding_init)]:
        X = pd.concat(list(predictors.values()), axis=1)
        full = ridge(X.loc[train_ix], y.loc[train_ix], X.loc[val_ix], y.loc[val_ix])
        rows.append(dict(var='NONE', outcome=outname, **full, rer=rer(full['mse'], full['mse'])))
        for name, X in ablate(predictors):
            red = ridge(X.loc[train_ix], y.loc[train_ix], X.loc[val_ix], y.loc[val_ix])
            rows.append(dict(var=name,
                             outcome=outname,
                             **red,
                             rer=rer(red['mse'], full['mse'])))
    return pd.DataFrame.from_records(rows)
        
def ridge(X, y, X_val, y_val):
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    model = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=[ 10**n for n in range(-3, 11) ],
                                  fit_intercept=True, cv=None, scoring='r2',
                                  alpha_per_target=False
                          ))
    model.fit(X, y)
    pred = model.predict(X_val)
    return dict(mse=mean_squared_error(y_val, pred),
                r2=r2_score(y_val, pred),
                alpha=model.steps[-1][1].alpha_,
                best_cv=model.steps[-1][1].best_score_)
                                    
    
        

def ablate(variables):
    """Yield dataframe concatenating all variables, except for one each time."""
    for this in variables:
        yield this, pd.concat([ var for name, var in variables.items() if name != this ], axis=1) 

        
def main():
    # Load and process data
    
    rawdata_d = pd.read_csv('data/out/pairwise_similarities_dialog.csv')
    data_d = massage(rawdata_d, scaleall=True)
    data_d.corr().to_csv("results/rsa_dialog_correlations.csv", index=True, header=True)
    data_d.corr().to_latex(float_format="%.2f", buf="results/rsa_dialog_correlations.tex")
    
    rawdata_n = pd.read_csv('data/out/pairwise_similarities_narration.csv')
    data_n = massage(rawdata_n, scaleall=True)
    data_ncor = data_n.drop("samespeaker", axis=1).corr()
    data_ncor.to_csv("results/rsa_narration_correlations.csv", index=True, header=True)
    data_ncor.to_latex(float_format="%.2f", buf="results/rsa_narration_correlations.tex")
    
    m_d = api.ols(formula = 'similarity ~ glovesim + distance + durationdiff + sametype + samespeaker + sameepisode', data=data_d)
    m_d_init = api.ols(formula = 'similarity_init ~ glovesim + distance + durationdiff + sametype + samespeaker + sameepisode', data=data_d)
    table = m_d.fit().summary2().tables[1]
    table['Trained'] = True
    table_init = m_d_init.fit().summary2().tables[1]
    table_init['Trained'] = False
    table_d = pd.concat([table, table_init])
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
    
    



