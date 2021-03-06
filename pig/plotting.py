from plotnine import *
import torch
import pandas as pd
import pig.evaluation as ev
import yaml
import numpy as np

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

def group_runs(conditions):
    output = dict(pretraining=conditions['base'] + conditions['pretraining_v'] + \
                        conditions['pretraining_a'] + conditions['pretraining_none'],
                  freeze_wav2vec=conditions['base'] + conditions['freeze_wav2vec'],
                  jitter=conditions['base'] + conditions['jitter'],
                  static=conditions['pretraining_a'] + conditions['static'])
    return output

def plots():
    configs = yaml.safe_load(open("conditions.yaml"))
    conditions = group_runs(configs)
    versions = flatten(conditions.values())
    data = flatten([ torch.load(f"results/full_scores_v{version}.pt")
                     for version in versions ])
    data = ev.add_condition(data)
    data = score_points(data)
    data['pretraining'] = pd.Categorical(data.apply(ev.pretraining, axis=1),
                                         categories=['None', 'V', 'A', 'AV'])
    data = data.fillna(dict(scrambled_video=False))
    data['version'] = data['version'].astype(int)

    
    for condition, versions in conditions.items():
        
        if condition == 'jitter':
            
            g = ggplot(data.query(f'version in {versions} & scrambled_video == False & metric != "triplet_acc"'),
                   aes(color=condition, y='score', x='fragment_type')) + \
                   geom_boxplot(outlier_shape='') + \
                   facet_wrap('~metric') + \
                   theme(aspect_ratio=0.6,
                         strip_background_x=element_text(height=0.1),
                         legend_position="bottom",
                         legend_title_align='center',
                         legend_background=element_rect(alpha=0.0)) + \
                   labs(x=None)
            ggsave(g, f"results/ablations/{condition}.pdf")
        else:
            fake1 = data.query(f'version in {versions} & scrambled_video == False & metric != "recall_at_10_jitter"')
            fake1['fragment_type'] = 'dialog'
            fake2 = data.query(f'version in {versions} & scrambled_video == False & metric != "recall_at_10_jitter"')
            fake2['fragment_type'] = 'narration'
            fake = pd.concat([fake1, fake2])
            if condition == 'pretraining':
                mapp = aes(x=condition, line_type='factor(version)', y='score')
                g = ggplot(data.query(f'version in {versions} & scrambled_video == False & metric != "recall_at_10_jitter"'), mapp) + \
                    geom_boxplot(outlier_shape='') + \
                    geom_blank(data=fake) + \
                    facet_wrap('~metric + fragment_type', scales='free') + \
                    theme(legend_position="none")
                ggsave(g, f"results/ablations/{condition}.pdf", width=10, height=4)
            else:
                mapp = aes(color=condition, y='score', x='fragment_type')
                g = ggplot(data.query(f'version in {versions} & scrambled_video == False & metric != "recall_at_10_jitter"'), mapp) + \
                    geom_boxplot(outlier_shape='') + \
                    geom_blank(data=fake) + \
                    facet_wrap('~metric', scales='free') + \
                    theme(aspect_ratio=0.6, strip_background_x=element_text(height=0.1),
                          legend_position="bottom", legend_title_align='center', legend_background=element_rect(alpha=0.0)) + \
                    labs(x=None)
                ggsave(g, f"results/ablations/{condition}.pdf")

    # scrambled
    unablated = configs["base"]
    fake1 = data.query(f'version in {unablated} & metric != "recall_at_10_jitter"')
    fake1['fragment_type'] = 'dialog'
    fake2 = data.query(f'version in {unablated} & metric != "recall_at_10_jitter"')
    fake2['fragment_type'] = 'narration'
    fake = pd.concat([fake1, fake2])
    g = ggplot(data.query(f'version in {unablated} & metric != "recall_at_10_jitter"'),
               aes(color='scrambled_video', y='score', x='fragment_type')) + \
               geom_boxplot(outlier_shape='') + \
               geom_blank(data=fake) + \
               facet_wrap('~metric', scales='free') + \
               labs(x=None) + \
               theme(aspect_ratio=0.6, strip_background_x=element_text(height=0.1),
                     legend_position="bottom", legend_title_align='center', legend_background=element_rect(alpha=0.0))
    ggsave(g, f"results/ablations/scrambled_video.pdf")


def recall_at_1_to_n_plot():
    data = torch.load(f"results/full_test_scores.pt")
    rows = [ datum for datum in data if not datum['scrambled_video'] ]
    recall_fixed  = torch.cat([ row['recall_fixed'].mean(dim=2) for row in rows ])
    recall_jitter = torch.cat([ row['recall_jitter'].mean(dim=2) for row in rows ])
    recall_fixed = pd.DataFrame(recall_fixed.numpy()).melt(var_name='N', value_name='recall')
    recall_fixed['segmentation'] = 'fixed'
    recall_jitter = pd.DataFrame(recall_jitter.numpy()).melt(var_name='N', value_name='recall')
    recall_jitter['segmentation'] = 'jitter'

    recall = pd.concat([recall_fixed, recall_jitter], ignore_index=True).query('N > 0')
    g = ggplot(recall, aes(x='factor(N)', y='recall', color='segmentation')) + \
        geom_boxplot(outlier_shape='') + \
        xlab('N') + \
        ylab('recall@N') + \
        theme(aspect_ratio=0.5, legend_position=(0.8, 0.25), legend_title_align='center', legend_margin=10,
              legend_background=element_rect(alpha=0.0))
    ggsave(g, 'results/recall_at_1_to_n_test.pdf')


def duration_effect_plot():
    static = yaml.safe_load(open("conditions.yaml"))['static']
    duration = torch.load("results/duration_effect.pt")
    subframes = []
    for ft in duration:
        for i in range(len(ft['model_ids'])):
            df = pd.DataFrame(data=dict(fragment_type=ft['fragment_type'],
                                        version=ft['model_ids'][i],
                                        success=ft['success'][i].cpu().numpy(),
                                        duration=ft['duration'].cpu().numpy()))
            subframes.append(df)
    data = pd.concat(subframes)
    data['static'] = data['version'].map(lambda v: v in static)
    grouped = data.groupby(['static', 'duration', 'fragment_type'])['success'].\
        agg([np.mean, len]).rename(columns={'mean': 'score', 'len': 'size'})
    diff = grouped.xs(False, level='static')[['score']] - grouped.xs(True, level='static')[['score']]
    size = grouped.xs(True, level='static')[['size']]
    wdata = pd.concat([diff, size], axis=1).rename(columns={'score': 'difference'})
    g = ggplot(wdata.reset_index(), aes(x='duration', y='difference', size='size', weight='size')) + \
        geom_point(alpha=0.5) + \
        geom_smooth() + \
        facet_wrap('~ fragment_type') + \
        guides(size=None) + \
        theme(aspect_ratio=1)
    ggsave(g, "results/duration_effect.pdf")

def duration_effect_scramble_plot():
    duration = torch.load("results/duration_effect_scramble.pt")
    subframes = []
    for ft in duration:
        for i in range(len(ft['model_ids'])):
            df = pd.DataFrame(data=dict(fragment_type=ft['fragment_type'],
                                        version=ft['model_ids'][i],
                                        scrambled=ft['scrambled_video'][i],
                                        success=ft['success'][i].cpu().numpy(),
                                        duration=ft['duration'].cpu().numpy()))
            subframes.append(df)
    data = pd.concat(subframes)
    grouped = data.groupby(['scrambled', 'duration', 'fragment_type'])['success'].\
        agg([np.mean, len]).rename(columns={'mean': 'score', 'len': 'size'})
    diff = grouped.xs(False, level='scrambled')[['score']] - grouped.xs(True, level='scrambled')[['score']]
    size = grouped.xs(True, level='scrambled')[['size']]
    wdata = pd.concat([diff, size], axis=1).rename(columns={'score': 'difference'})
    g = ggplot(wdata.reset_index(), aes(x='duration', y='difference', size='size', weight='size')) + \
        geom_point(alpha=0.5) + \
        geom_smooth() + \
        facet_wrap('~ fragment_type') + \
        guides(size=None)
    ggsave(g, "results/duration_effect_scramble.pdf")
    
def flatten(X):
    return [ y for Y in X for y in Y ]
