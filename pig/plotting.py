from plotnine import *
import torch
import pandas as pd
import pig.evaluation as ev
import yaml

def score_points(data):
    metrics = ['triplet_acc', 'recall_at_10_fixed', 'recall_at_10_jitter']
    rows = []
    for row in data:
        for metric in metrics:
            for score in row[metric]:
                point = { k:v for k, v in row.items()  if k not in metrics }
                if metric == 'triplet_acc':
                    point['score'] = score.item()
                else:
                    point['score'] = score.mean().item()
                point['metric'] = metric
                rows.append(point)
    return pd.DataFrame.from_records(rows)

def plots(condition_file="conditions.yaml"):
    conditions = yaml.safe_load(open(condition_file))
    versions = flatten(conditions.values())
    data = flatten([ torch.load(f"results/full_scores_v{version}.pt")
                     for version in versions ])
    data = ev.add_condition(data)
    data = score_points(data)
    data['pretraining'] = pd.Categorical(data.apply(ev.pretraining, axis=1),
                                         categories=['None', 'V', 'A', 'AV'])
    data = data.fillna(dict(scrambled_video=False))
    for condition, versions in conditions.items():
        
        if condition == 'jitter':
            
            g = ggplot(data.query(f'version in {versions} & scrambled_video == False & metric != "triplet_acc"'),
                   aes(x=condition, y='score')) + \
                   geom_boxplot() + \
                   facet_wrap('~metric + fragment_type')
        else:
            fake1 = data.query(f'version in {versions} & scrambled_video == False & metric != "recall_at_10_jitter"')
            fake1['fragment_type'] = 'dialog'
            fake2 = data.query(f'version in {versions} & scrambled_video == False & metric != "recall_at_10_jitter"')
            fake2['fragment_type'] = 'narration'
            fake = pd.concat([fake1, fake2])
            g = ggplot(data.query(f'version in {versions} & scrambled_video == False & metric != "recall_at_10_jitter"'),
                   aes(x=condition, y='score')) + \
                   geom_boxplot() + \
                   geom_blank(data=fake) + \
                   facet_wrap('~metric + fragment_type', scales='free') 
        
        ggsave(g, f"results/ablations/{condition}.pdf")
    # scrambled

    unablated = conditions["pretraining"][0]
    fake1 = data.query(f'version == {unablated} & metric != "recall_at_10_jitter"')
    fake1['fragment_type'] = 'dialog'
    fake2 = data.query(f'version == {unablated} & metric != "recall_at_10_jitter"')
    fake2['fragment_type'] = 'narration'
    fake = pd.concat([fake1, fake2])
    g = ggplot(data.query(f'version == {unablated} & metric != "recall_at_10_jitter"'),
               aes(x='scrambled_video', y='score')) + \
               geom_boxplot() + \
               geom_blank(data=fake) + \
               facet_wrap('~metric + fragment_type', scales='free')
    ggsave(g, f"results/ablations/scrambled_video.pdf")

def flatten(X):
    return [ y for Y in X for y in Y ]
