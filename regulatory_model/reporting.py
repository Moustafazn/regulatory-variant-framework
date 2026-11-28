"""Study provenance, numerical tables, result figures and model architecture."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np


def clean_json(value):
    # Convert NumPy objects and nonfinite numbers into interoperable JSON values.
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    # Atomically write strict JSON without NaN or Infinity extensions.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False, sort_keys=True) + '\n')
    temporary.replace(path)


def sha256(path):
    # Hash input and output files for reproducibility and review binding.
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def platform_metadata():
    # Record Python and library versions on every run.
    import importlib.metadata
    versions = {}
    for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'openpyxl', 'pyBigWig', 'pyliftover', 'pyfaidx', 'matplotlib']:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'versions': versions}



PAPER_TITLE = 'Assessing uncertainty in regulatory effect transfer from Jurkat to primary CD4 T cells using NeuroJico principles'
INK, BLUE, ORANGE, TEAL, PURPLE = '#16354A', '#176B96', '#C77829', '#318B84', '#7956A8'
GRAY, GRID = '#87949D', '#E2E8ED'
METHOD_LABELS = {
    'train_mean': 'Training-set mean', 'source_copy': 'Direct source copy',
    'source_calibrated': 'Affine source', 'source_shrunk': 'Shrunk source',
    'source:ridge': 'Source | ridge', 'source:gbm': 'Source | boosting',
    'sequence:ridge': 'Molecular | ridge', 'sequence:gbm': 'Molecular | boosting',
    'source_sequence:ridge': 'Source + molecular | ridge',
    'source_sequence:gbm': 'Source + molecular | boosting',
    'source_sequence:binned_gbm': 'Binned molecular | boosting',
    'source_sequence_context:ridge': 'Source + molecular + DNase | ridge',
    'source_sequence_context:gbm': 'Source + molecular + DNase | boosting',
}
REVIEW_STYLES = {
    'source_se': ('Source SE', BLUE, '-', 'o'),
    'predicted_absolute_error': ('Learned error', ORANGE, '-', 's'),
    'vprs_soft': ('Soft rough membership', TEAL, '-', 'D'),
    'vprs_crisp': ('Crisp rough membership', PURPLE, '--', '^'),
    'random': ('Random ranking', GRAY, ':', 'x'),
}
REPORT_INPUTS = ('protocol.json', 'results.json', 'replicate_diagnostics.json',
                 'confirmation_predictions.csv.gz', 'routing_predictions.csv.gz',
                 'routing_development_partitions.csv.gz')


def report_frames(output, protocol, results):
    """Read aligned, immutable prediction artifacts before making publication outputs."""
    import pandas as pd
    output = Path(output)
    predictions = pd.read_csv(output / 'confirmation_predictions.csv.gz')
    routing = pd.read_csv(output / 'routing_predictions.csv.gz')
    partitions = pd.read_csv(output / 'routing_development_partitions.csv.gz')
    if predictions.example_id.duplicated().any() or routing.example_id.duplicated().any():
        raise ValueError('Duplicate confirmation identifiers in report inputs.')
    if not predictions[['example_id', 'split_group']].equals(routing[['example_id', 'split_group']]):
        raise ValueError('Prediction and routing rows must have identical identifiers and locus order.')
    if len(predictions) != protocol['confirmation_rows'] or predictions.split_group.nunique() != protocol['confirmation_loci']:
        raise ValueError('Confirmation counts differ from the saved protocol.')
    if len(partitions) != protocol['development_rows'] or partitions.split_group.nunique() != protocol['development_loci']:
        raise ValueError('Review development counts differ from the saved protocol.')
    if partitions.groupby('split_group').routing_partition.nunique().max() != 1:
        raise ValueError('A locus occurs in more than one review development partition.')
    if not np.isfinite(predictions[['target_effect', *results['confirmation']]].to_numpy()).all():
        raise ValueError('Nonfinite target effects or predictions in report inputs.')
    if not np.isfinite(routing.prediction).all() or routing.interval_half_width.isna().any() or (routing.interval_half_width < 0).any():
        raise ValueError('Invalid routing predictions or interval widths in report inputs.')
    for name, metrics in results['confirmation'].items():
        error = (predictions.target_effect - predictions[name]) ** 2
        observed = error.groupby(predictions.split_group).mean().mean()
        if not np.isclose(observed, metrics['equal_locus_mse'], rtol=1e-10, atol=1e-12):
            raise ValueError(f'Saved prediction MSE mismatch: {name}')
    return predictions, routing, partitions


def _contrast_record(name, contrast, reference, candidate):
    relative_ci = contrast.get('relative_ci95') or [None, None]
    return {'record_type': 'paired_contrast', 'comparison': name, 'reference': reference,
            'method': candidate, 'absolute_mse_reduction': contrast['mse_reduction'],
            'relative_mse_reduction': contrast['relative_reduction'],
            'ci95_low': contrast['ci95'][0], 'ci95_high': contrast['ci95'][1],
            'relative_ci95_low': relative_ci[0], 'relative_ci95_high': relative_ci[1],
            'loci': contrast['loci'], 'bootstrap_draws': contrast['bootstrap_draws'],
            'interval_type': '95% paired locus bootstrap; conditional on fitted models and selection'}


def paper_tables(protocol, results, replicates, predictions, routing, partitions):
    """Two main tables and three supplementary tables, with explicit record types."""
    import pandas as pd
    cfg = protocol['config']; selection = results['selection']; review = results['routing']
    cohort = []
    def row(section, item, context, variants=None, loci=None, details=''):
        cohort.append(dict(section=section, item=item, context=context, variants=variants, loci=loci, details=details))
    audit = protocol['provenance'].get('coordinates', {})
    context = 'Jurkat / activated primary CD4 T cells'
    row('Assay source', 'Ho et al., paired MPRA', context,
        details='Supplementary sheets 1 and 9; quantitative allelic Log2Skew and reported SE')
    if 'input_rows' in audit:
        row('Cohort', 'Quantitative endpoint eligible', context, audit['input_rows'],
            details='Paired finite allelic effects; positive SE; both primary-cell plasmid means >20; before coordinate QC')
        row('Cohort', 'Reference verified', context, audit['normalized_rows'],
            details=f'GRCh38 reference verified before optional sampling; {audit["dropped_rows"]} coordinate exclusions')
    row('Cohort', 'Analyzed cohort', context, protocol['development_rows'] + protocol['confirmation_rows'],
        protocol['development_loci'] + protocol['confirmation_loci'], f'Loci grouped at {cfg["gap_bp"]:,} bp plus shared identifiers')
    for part in ('development', 'confirmation'):
        row('Prediction partition', part.capitalize(), context, protocol[part + '_rows'], protocol[part + '_loci'])
    for part in ('fit', 'score_training', 'policy_selection', 'calibration'):
        subset = partitions[partitions.routing_partition.eq(part)]
        row('Review development partition', part.replace('_', ' ').capitalize(), context,
            len(subset), subset.split_group.nunique(), 'Disjoint locus partitions within development')
    row('Annotation source', 'ENCFF916JRN', 'Jurkat DNase', details='GRCh38; exact covered-base mean in 150-bp windows')
    row('Annotation source', 'ENCFF124RFI', 'CD4 DNase', details='Lineage proxy; activation-state match not established')
    if 'paired_variants' in replicates:
        row('Replicate diagnostic', 'GSE297471 normalized counts', 'Primary T cells', replicates['paired_variants'],
            details=f'{replicates["rna_replicates"]} RNA and {replicates["dna_replicates"]} DNA replicates; allele pairs before endpoint-cohort QC')
    methods = []
    for name in protocol['candidates']:
        family = name.split(':')[0]
        inputs = {'train_mean': 'Training target mean; no variant predictors', 'source_copy': 'Jurkat effect',
                  'source_calibrated': 'Jurkat effect', 'source_shrunk': 'Jurkat effect and SE',
                  'source': 'Jurkat effect, log SE and shrunk effect', 'sequence': 'Alleles and molecular annotations',
                  'source_sequence': 'Source representation plus molecular annotations',
                  'source_sequence_context': 'Source and molecular representation plus DNase'}.get(family, family)
        methods.append({'task': 'Effect prediction', 'method': name, 'inputs': inputs,
                        'fitting_data': 'No parameter fitting' if name == 'source_copy' else 'Development fitting folds; final fit on all development loci',
                        'selection_rule': f'{cfg["folds"]}-fold grouped development CV; equal-locus MSE'})
    for name, (label, _, _, _) in REVIEW_STYLES.items():
        methods.append({'task': 'Review prioritization', 'method': name,
                        'inputs': {'source_se': 'Jurkat SE', 'predicted_absolute_error': 'Source, molecular and DNase representation',
                                   'vprs_soft': 'Distance-weighted soft class memberships',
                                   'vprs_crisp': 'Distance-weighted crisp class memberships', 'random': 'Seeded random ranking'}[name],
                        'fitting_data': {'source_se': 'No score fitting', 'predicted_absolute_error': 'Score-training loci; absolute errors of fixed predictor',
                                         'vprs_soft': 'Predictor-fitting loci; sampled source/target effects',
                                         'vprs_crisp': 'Predictor-fitting loci; most frequent sampled class', 'random': 'No score fitting'}[name],
                        'selection_rule': 'Descriptive reference only' if name == 'random' else
                        f'Policy-selection loci; retained MSE at {cfg["primary_review_budget"]:.0%} budget; ordinary and rough families selected separately'})
    prediction_rows = [{'record_type': 'performance', 'method': name, **values}
                       for name, values in results['confirmation'].items()]
    prediction_rows += [
        _contrast_record('Molecular augmentation', results['primary_gain_vs_source'], selection['selected_source_baseline'], selection['selected_augmented']),
        _contrast_record('DNase increment', results['context_increment'], selection['context_parent'], selection['context_candidate']),
    ]
    review_rows = [{'record_type': 'risk_curve', **values} for values in review['curves']]
    contrast = _contrast_record('Selected rough versus ordinary score', review['challenger_gain_vs_selected_reference'],
                                review['selection']['reference'], review['selection']['challenger'])
    contrast['budget'] = review['selection']['primary_budget']
    contrast['selector'] = contrast.pop('method')
    review_rows.append(contrast)
    reliability = []
    for name, values in replicates.get('correlations', {}).items():
        low, high = values['partition_percentiles_2_5_97_5']
        reliability.append({'analysis': 'Replicate agreement', 'condition': name, 'metric': 'Median Spearman correlation',
                            'value': values['median_spearman'], 'lower': low, 'upper': high,
                            'interval_type': '2.5th-97.5th partition percentiles; not a confidence interval',
                            'denominator': replicates.get('paired_variants'), 'unit': 'Allele pairs',
                            'partitions': replicates.get('partitions'),
                            'rna_replicates_per_half': '/'.join(map(str, replicates['rna_half_sizes'])),
                            'dna_controls_per_half': '/'.join(map(str, [replicates['dna_replicates']]*2 if name == 'shared_dna' else replicates['dna_half_sizes']))})
    covered = (abs(predictions.target_effect - routing.prediction) <= routing.interval_half_width)
    whole = covered.groupby(predictions.split_group).all()
    for name, values, key in [('Row coverage', covered, 'empirical_row_coverage'),
                              ('Whole-locus coverage', whole, 'empirical_whole_locus_coverage')]:
        if not np.isclose(values.mean(), review['conformal'][key]):
            raise ValueError('Conformal coverage does not match the saved results.')
        reliability.append({'analysis': 'Conformal calibration', 'condition': name, 'metric': 'Coverage',
                            'value': float(values.mean()), 'numerator': int(values.sum()), 'denominator': len(values),
                            'unit': 'Variants' if name == 'Row coverage' else 'Loci',
                            'target': review['conformal']['target_whole_locus_coverage'] if name == 'Whole-locus coverage' else None})
    widths = 2 * routing.interval_half_width.to_numpy()
    finite = widths[np.isfinite(widths)]
    reliability.append({'analysis': 'Conformal calibration', 'condition': 'All interval widths',
                        'metric': 'Mean width', 'value': review['conformal']['mean_interval_width'],
                        'denominator': len(widths), 'unit': 'Log2Skew'})
    if len(finite):
        low, high = np.quantile(finite, [.025, .975])
        reliability.append({'analysis': 'Conformal calibration', 'condition': 'Finite interval widths',
                            'metric': 'Median width', 'value': float(np.median(finite)), 'lower': float(low), 'upper': float(high),
                            'interval_type': '2.5th-97.5th variant percentiles; not a confidence interval',
                            'denominator': len(finite), 'unit': 'Log2Skew'})
    reliability.append({'analysis': 'Conformal calibration', 'condition': 'Unbounded intervals', 'metric': 'Count',
                        'value': int((~np.isfinite(widths)).sum()), 'denominator': len(widths), 'unit': 'Variants'})
    tables = {'cohort_table.csv': pd.DataFrame(cohort), 'comparator_table.csv': pd.DataFrame(methods),
              'prediction_table.csv': pd.DataFrame(prediction_rows), 'routing_table.csv': pd.DataFrame(review_rows),
              'reliability_table.csv': pd.DataFrame(reliability)}
    for table in tables.values():
        for column in ('variants', 'loci', 'rows', 'reviewed_rows', 'retained_loci', 'denominator', 'numerator', 'partitions', 'bootstrap_draws'):
            if column in table:
                table[column] = table[column].astype('Int64')
    return tables


def _plot_style():
    return {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
            'axes.titleweight': 'bold', 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
            'axes.edgecolor': '#A8B5BF', 'axes.labelcolor': INK, 'text.color': INK,
            'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': .6,
            'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}


def _panel(ax, letter, title, subtitle):
    ax.set_title(f'{letter}  {title}', loc='left', pad=35)
    ax.text(0, 1.055, subtitle, transform=ax.transAxes, fontsize=9, color='#526573', va='bottom')
    ax.set_axisbelow(True)


def _save_plot(fig, output, stem):
    """Reject clipped text before saving the publication figure."""
    import matplotlib.pyplot as plt
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend([ax.title, ax._left_title, ax._right_title, *ax.texts, ax.xaxis.label, ax.yaxis.label])
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            texts.extend(tick.label1 for tick in axis.get_major_ticks() if low <= tick.get_loc() <= high)
    for text in texts:
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
            raise ValueError(f'{stem}: text extends beyond the canvas: {text.get_text()}')
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(Path(output) / f'{stem}.{extension}', dpi=600, facecolor='white')
    plt.close(fig)


def _forest(ax, records, labels):
    for i, (contrast, color) in enumerate(records):
        lo, hi = contrast['ci95']; point = contrast['mse_reduction']
        if lo is None or hi is None:
            ax.text(0, i, 'Interval undefined', fontsize=9)
            continue
        ax.plot([lo, hi], [i, i], color=color, lw=2)
        ax.plot([lo, lo], [i-.06, i+.06], color=color, lw=.8)
        ax.plot([hi, hi], [i-.06, i+.06], color=color, lw=.8)
        ax.scatter([point], [i], s=45, c=color, edgecolor='white', lw=.6, zorder=4)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.5, -.5); ax.tick_params(axis='y', length=0, pad=7)
    ax.spines['left'].set_visible(False); ax.grid(axis='x', color=GRID, lw=.6)
    ax.axvline(0, color=GRAY, lw=.9, ls=(0, (4, 3)))
    ax.set_xlabel('MSE reduction (positive favors candidate)', labelpad=8)
    ax.margins(x=.18)


def draw_prediction(output, protocol, results, frame):
    """Figure 2: paired predictions, locus-weighted losses, and prediction contrasts."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    with plt.rc_context(_plot_style()):
        fig = plt.figure(figsize=(11, 9))
        # A full-width lower comparison avoids squeezing method labels into a narrow panel.
        gs = fig.add_gridspec(3, 2, left=.30, right=.96, bottom=.09, top=.84,
                              height_ratios=[1.5, 1.35, .75], hspace=1.22, wspace=.45)
        a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        c, d = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
        fig.suptitle('Jurkat-to-primary-CD4 effect prediction', x=.30, y=.98, ha='left', fontsize=15, fontweight='bold')
        fig.text(.30, .942, f'{len(frame):,} confirmation variants | {protocol["confirmation_loci"]} loci', fontsize=10, color='#526573')
        selected = results['selection']; baseline, candidate = selected['selected_source_baseline'], selected['selected_augmented']
        values = frame[['target_effect', baseline, candidate]].to_numpy()
        lo, hi = float(np.min(values)), float(np.max(values)); pad = max((hi-lo)*.05, .05)
        for ax, name, letter, title, color in [(a, baseline, 'A', 'Source-only reference', BLUE), (b, candidate, 'B', 'Augmented predictor', ORANGE)]:
            ax.scatter(frame.target_effect, frame[name], s=9, alpha=.25, color=color, edgecolors='none', rasterized=True)
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color=GRAY, ls='--', lw=.8)
            ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad), xlabel='Observed target Log2Skew', ylabel='Predicted target Log2Skew')
            ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
            _panel(ax, letter, title, METHOD_LABELS.get(name, name).replace('Source + molecular', 'Molecular + source'))
        order = sorted(results['confirmation'], key=lambda n: results['confirmation'][n]['equal_locus_mse'])
        losses = np.array([results['confirmation'][n]['equal_locus_mse'] for n in order]); yy = np.arange(len(order))
        c.hlines(yy, 0, losses, color=BLUE, lw=2, alpha=.5)
        c.scatter(losses, yy, s=42, color=BLUE, edgecolor='white', lw=.6, zorder=4)
        c.set_yticks(yy, [METHOD_LABELS.get(n, n) for n in order]); c.tick_params(axis='y', length=0, pad=8)
        c.set_ylim(len(order)-.4, -.65); c.set_xlim(0, max(losses)*1.21)
        c.set_xlabel('Equal-locus MSE'); c.spines['left'].set_visible(False); c.grid(axis='x', color=GRID, lw=.6)
        for y, value in zip(yy, losses):
            c.annotate(f'{value:.5f}', (value, y), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9)
        _panel(c, 'C', 'Confirmation prediction error', 'Development-selected comparisons and prespecified references')
        _forest(d, [(results['primary_gain_vs_source'], BLUE), (results['context_increment'], ORANGE)], ['Molecular augmentation', 'DNase increment'])
        _panel(d, 'D', 'Paired prediction contrasts', '95% intervals from paired locus resampling')
        _save_plot(fig, output, 'prediction')


def draw_review(output, protocol, results):
    """Figure 3: review risk, observed loss capture, and the primary policy contrast."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    with plt.rc_context(_plot_style()):
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, left=.13, right=.96, bottom=.10, top=.73,
                              height_ratios=[1.7, 1], hspace=1.18, wspace=.42)
        a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        review = results['routing']; primary = 100*review['selection']['primary_budget']
        fig.suptitle('Uncertainty scores for expert-review prioritization', x=.13, y=.98, ha='left', fontsize=14, fontweight='bold')
        fig.text(.13, .937, f'Fixed source-shrinkage predictor | primary review budget: {primary:g}%', fontsize=10, color='#526573')
        handles = [Line2D([0], [0], label=label, color=color, ls=ls, marker=marker, markersize=4, lw=1.4)
                   for label, color, ls, marker in REVIEW_STYLES.values()]
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.12, .91), ncol=3, frameon=False, fontsize=9)
        for name, (_, color, ls, marker) in REVIEW_STYLES.items():
            rows = sorted([r for r in review['curves'] if r['selector'] == name], key=lambda r: r['budget'])
            x = [100*r['budget'] for r in rows]
            a.plot(x, [r['retained_mse'] for r in rows], color=color, ls=ls, marker=marker, markersize=5, lw=1.6)
            b.plot(x, [100*r['observed_loss_captured_fraction'] if r['observed_loss_captured_fraction'] is not None else np.nan for r in rows],
                   color=color, ls=ls, marker=marker, markersize=5, lw=1.6)
        for ax in (a, b):
            ax.set_xlabel('Review budget (% of variants)', labelpad=8)
            ax.set_xticks(sorted({100*r['budget'] for r in review['curves']}))
            ax.grid(color=GRID, lw=.6); ax.axvline(primary, color=GRAY, ls='--', lw=.7, alpha=.6)
        a.set_ylabel('Retained MSE'); b.set_ylabel('Observed squared loss captured (%)')
        _panel(a, 'A', 'Error among retained estimates', 'Original locus weights remain fixed')
        _panel(b, 'B', 'Loss captured by the review queue', 'Includes observed measurement error')
        _forest(c, [(review['challenger_gain_vs_selected_reference'], TEAL)], [''])
        chosen = review['selection']; ref = REVIEW_STYLES[chosen['reference']][0]; challenger = REVIEW_STYLES[chosen['challenger']][0]
        _panel(c, 'C', f'Selected policies at the {primary:g}% budget', f'{challenger} versus {ref} | paired locus 95% interval')
        contrast = review['challenger_gain_vs_selected_reference']
        relative = contrast['relative_reduction']
        if relative is not None:
            c.text(.98, .84, f'Relative MSE reduction: {100*relative:+.2f}%', transform=c.transAxes, ha='right', fontsize=10)
        _save_plot(fig, output, 'review')


def draw_reliability(output, results, replicates, predictions, routing):
    """Figure 4: replicate sensitivity, calibration units, and interval widths."""
    import matplotlib.pyplot as plt
    with plt.rc_context(_plot_style()):
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, left=.17, right=.96, bottom=.10, top=.81,
                              height_ratios=[1, 1.2], hspace=1.06, wspace=.50)
        a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        fig.suptitle('Measurement reliability and uncertainty calibration', x=.17, y=.98, ha='left', fontsize=14, fontweight='bold')
        fig.text(.17, .937, 'Primary-cell replicate diagnostics and confirmation prediction intervals', fontsize=10, color='#526573')
        names = [('shared_dna', 'Shared DNA', PURPLE), ('disjoint_dna', 'Disjoint DNA', ORANGE)]
        for y, (name, _, color) in enumerate(names):
            values = replicates.get('correlations', {}).get(name)
            if values is None:
                continue
            low, high = values['partition_percentiles_2_5_97_5']; point = values['median_spearman']
            a.plot([low, high], [y, y], color=color, lw=2)
            a.scatter([point], [y], color=color, s=48, edgecolor='white', lw=.6, zorder=4)
            a.annotate(f'{point:.3f}', (point, y), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=10)
        a.set_yticks([0, 1], [v[1] for v in names]); a.set_ylim(1.7, -.7); a.set_xlim(-1, 1)
        a.set_xticks([-1, -.5, 0, .5, 1]); a.set_xlabel('Spearman correlation'); a.grid(axis='x', color=GRID, lw=.6)
        a.spines['left'].set_visible(False); a.tick_params(axis='y', length=0)
        _panel(a, 'A', 'Replicate agreement', f'Median and partition percentiles | {replicates.get("partitions", 0)} splits')
        covered = abs(predictions.target_effect-routing.prediction) <= routing.interval_half_width
        whole = covered.groupby(predictions.split_group).all()
        coverage = [float(covered.mean()), float(whole.mean())]
        b.bar([0, 1], coverage, width=.5, color=[BLUE, TEAL], alpha=.85)
        for i, (value, series) in enumerate(zip(coverage, [covered, whole])):
            b.text(i, value+.025, f'{100*value:.2f}%\n{int(series.sum()):,}/{len(series):,}', ha='center', fontsize=10)
        nominal = results['routing']['conformal']['target_whole_locus_coverage']
        b.axhline(nominal, color=GRAY, ls='--', lw=1)
        b.set_xticks([0, 1], ['Variants', 'Whole loci']); b.set_ylim(0, 1.24)
        b.set_yticks([0, .25, .5, .75, 1], ['0', '25', '50', '75', '100']); b.set_ylabel('Empirical coverage (%)')
        _panel(b, 'B', 'Coverage depends on its unit', f'Dashed line: {nominal:.0%} whole-locus target')
        widths = 2*routing.interval_half_width.to_numpy(); finite = widths[np.isfinite(widths)]
        if len(finite):
            c.hist(finite, bins=min(32, max(5, int(np.sqrt(len(finite))))), color=TEAL, edgecolor='white', lw=.5)
            median = float(np.median(finite)); c.axvline(median, color=ORANGE, lw=1.5)
            c.text(.98, .88, f'Median width: {median:.2f}\nFinite intervals: {len(finite):,}/{len(widths):,}',
                   transform=c.transAxes, ha='right', va='top', fontsize=10)
        else:
            c.text(.5, .5, 'All prediction intervals are unbounded', transform=c.transAxes, ha='center')
        c.set_xlabel('Prediction interval width (Log2Skew)'); c.set_ylabel('Variants')
        c.grid(axis='y', color=GRID, lw=.6)
        _panel(c, 'C', 'Width of confirmation intervals', 'Full range of finite widths; unbounded counts reported separately')
        _save_plot(fig, output, 'reliability')


def write_study_summary(output, protocol, results, replicates):
    """Generate the four-figure, two-table manuscript and three supplementary tables."""
    import matplotlib
    matplotlib.use('Agg')
    output = Path(output)
    predictions, routing, partitions = report_frames(output, protocol, results)
    tables = paper_tables(protocol, results, replicates, predictions, routing, partitions)
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    draw_prediction(output, protocol, results, predictions)
    draw_review(output, protocol, results)
    draw_reliability(output, results, replicates, predictions, routing)
    draw_architecture(output, neighbors=protocol['config']['rough']['neighbors'])
    table_index = [
        ('Table 1', 'Data sources, cohort and evaluation partitions', 'cohort_table.csv'),
        ('Table 2', 'Predictors, review scores and selection rules', 'comparator_table.csv'),
        ('Table S1', 'Prediction performance and paired contrasts', 'prediction_table.csv'),
        ('Table S2', 'Review curves and the selected-policy contrast', 'routing_table.csv'),
        ('Table S3', 'Replicate agreement and calibration diagnostics', 'reliability_table.csv'),
    ]
    figure_index = [
        ('Figure 1', 'Prediction and expert-review architecture', 'architecture'),
        ('Figure 2', 'Jurkat-to-primary-CD4 effect prediction', 'prediction'),
        ('Figure 3', 'Uncertainty scores for review prioritization', 'review'),
        ('Figure 4', 'Measurement reliability and uncertainty calibration', 'reliability'),
    ]
    paper = {'title': PAPER_TITLE,
             'tables': [{'number': n, 'title': title, 'file': path} for n, title, path in table_index],
             'figures': [{'number': n, 'title': title, 'files': [f'{stem}.{ext}' for ext in ('pdf', 'svg', 'png')]}
                         for n, title, stem in figure_index],
             'source_sha256': {name: sha256(output/name) for name in REPORT_INPUTS},
             'reporting_code_sha256': sha256(Path(__file__))}
    write_json(output / 'paper_manifest.json', paper)
    gain = results['primary_gain_vs_source']; review = results['routing']['challenger_gain_vs_selected_reference']
    lines = [f'# {PAPER_TITLE}', '', results['conclusion'], '',
             f'- Cohort: {protocol["development_rows"]+protocol["confirmation_rows"]:,} variants; {protocol["development_loci"]} development and {protocol["confirmation_loci"]} confirmation loci.',
             f'- Selected source reference: {results["selection"]["selected_source_baseline"]}.',
             f'- Selected augmented model: {results["selection"]["selected_augmented"]}.',
             f'- Relative prediction MSE reduction: {100*gain["relative_reduction"]:.3f}%; paired absolute reduction 95% interval: {gain["ci95"]}.',
             f'- Rough-selector retained MSE reduction versus the selected ordinary score: {100*review["relative_reduction"]:.3f}%; paired absolute reduction 95% interval: {review["ci95"]}.',
             f'- Prediction usefulness criterion passed: {results["useful_prediction_improvement_verified"]}.',
             f'- Review usefulness criterion passed: {results["routing"]["useful_routing_improvement_verified"]}.', '',
             '## Main figures', '']
    for number, title, stem in figure_index:
        lines.extend([f'**{number}. {title}.**', '', f'![{title}]({stem}.png)', ''])
    lines.extend(['## Tables', '', '| Number | Contents | Data |', '|---|---|---|'])
    for number, title, name in table_index:
        lines.append(f'| {number} | {title} | [{name}]({name}) |')
    lines.extend(['', 'Prediction and review comparisons use separately fitted predictors. Review cards contain source-side evidence and predictions; expert benefit is unmeasured.',
                  '', 'Bootstrap intervals condition on fitted models and selected policies. Replicate percentile ranges describe the sampled partitions and are not confidence intervals.',
                  '', 'The fixed confirmation partition belongs to a previously explored public cohort. Interpret the results as an internal benchmark.', ''])
    (output / 'study_summary.md').write_text('\n'.join(lines))
    # Remove only superseded report artifacts after all replacement outputs succeed.
    for name in ('benchmark.svg', 'benchmark.pdf', 'benchmark.png', 'contrast_table.csv', 'replicate_table.csv'):
        (output / name).unlink(missing_ok=True)


def draw_architecture(output, neighbors=25):
    """Draw the prediction workflow and its neighborhood-uncertainty detail."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon, FancyArrowPatch
    from matplotlib.colors import to_rgb

    ink='#303A42'; source='#7956A8'; molecular='#1575B5'; chromatin='#CE7618'
    response='#405876'; review='#008675'; muted='#687781'
    def tint(color, fraction):
        return tuple(1-fraction+fraction*c for c in to_rgb(color))
    style={'font.family':'DejaVu Sans','font.size':10.5,'text.color':ink,
           'mathtext.fontset':'dejavusans','svg.fonttype':'none','pdf.fonttype':42}
    with plt.rc_context(style):
        fig=plt.figure(figsize=(14.6,9.1),facecolor='white')
        ax=fig.add_axes([.012,.015,.976,.97])
        ax.set(xlim=(0,18),ylim=(0,11.1));ax.set_aspect('equal');ax.axis('off')
        labels=[]
        def text(x,y,label,size=10.5,color=ink,weight='normal',ha='center',va='center'):
            t=ax.text(x,y,label,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,linespacing=1.4)
            labels.append(t)
            return t
        def line(points,color=ink,lw=.9,ls='-'):
            xx,yy=zip(*points);ax.plot(xx,yy,color=color,lw=lw,ls=ls,solid_capstyle='round',zorder=1)
        def arrow(points,color=ink,dashed=False):
            if len(points)>2:line(points[:-1],color,.9,'--' if dashed else '-')
            ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=9,
                         color=color,lw=.9,linestyle='--' if dashed else '-',zorder=2))
        def ball(x,y,r,color):
            ax.add_patch(Circle((x,y),r,facecolor=tint(color,.88),edgecolor=tint(color,.95),lw=.45,zorder=4))
            for rr in np.linspace(.78,.16,7):
                ax.add_patch(Circle((x-r*.21,y+r*.22),r*rr,facecolor=tint(color,.2+.55*rr),edgecolor='none',zorder=5))
        def cuboid(x,y,w,h,color,depth=.11,shade=.14):
            ax.add_patch(Polygon([(x,y+h),(x+depth,y+h+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
                                 facecolor=tint(color,.09),edgecolor=tint(color,.6),lw=.55))
            ax.add_patch(Polygon([(x+w,y),(x+w+depth,y+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
                                 facecolor=tint(color,.32),edgecolor=tint(color,.6),lw=.55))
            ax.add_patch(Rectangle((x,y),w,h,facecolor=tint(color,shade),edgecolor=color,lw=.65))
        def vector(x,y,n,color,w=.17,h=.55):
            for i in range(n):
                xx=x+(i-(n-1)/2)*.25
                cuboid(xx,y-h/2,w,h,color,.065,.20+.09*(i%4))
                for yy in np.linspace(y-h/2,y+h/2,4)[1:-1]:line([(xx,yy),(xx+w,yy)],tint(color,.5),.35)
        def cell(x,y,color):
            ax.add_patch(Circle((x,y),.32,facecolor=tint(color,.12),edgecolor=color,lw=.85))
            ax.add_patch(Circle((x,y),.28,fill=False,edgecolor=tint(color,.45),lw=.45))
            ax.add_patch(Ellipse((x-.025,y+.015),.37,.40,angle=25,facecolor=tint(color,.36),edgecolor=tint(color,.75),lw=.55))
            ball(x-.075,y+.07,.045,color)
            for dx,dy,theta in [(.20,-.10,35),(-.13,-.23,-20)]:
                ax.add_patch(Ellipse((x+dx,y+dy),.10,.04,angle=theta,facecolor=tint(color,.65),edgecolor='none'))
        def dna(x,y):
            tt=np.linspace(-.40,.40,90);ww=.12*np.sin(tt*13)
            ax.plot(x+tt,y+ww,color=molecular,lw=1.7)
            ax.plot(x+tt,y-ww,color=tint(molecular,.5),lw=1.7)
            for i in range(0,len(tt),10):line([(x+tt[i],y+ww[i]),(x+tt[i],y-ww[i])],molecular,.6)
            ball(x+.015,y,.055,chromatin)
        def track(x,y,color):
            xx=np.linspace(-.46,.46,80)
            yy=.24*np.exp(-((xx+.18)/.09)**2)+.35*np.exp(-((xx-.15)/.14)**2)
            ax.fill_between(x+xx,y,y+yy,color=tint(color,.4),lw=0)
            line(list(zip(x+xx,y+yy)),color,.85);line([(x-.48,y),(x+.48,y)],muted,.45)
        def worksheet(x,y,color):
            cuboid(x-.26,y-.35,.50,.68,color,.06,.04)
            for j in range(3):
                line([(x-.17,y+.20-j*.17),(x+.13,y+.20-j*.17)],tint(color,.65),.8)
            ball(x+.20,y-.20,.12,color)
            line([(x+.28,y-.29),(x+.42,y-.43)],color,1.3)
        def neighborhood(x,y):
            positions=[(-.48,.26,0),(-.35,-.28,1),(.05,.40,0),(.41,.26,2),(.44,-.24,3),(.02,-.38,0)]
            colors=[molecular,chromatin,source,review]
            for dx,dy,c in positions:
                line([(x,y),(x+dx,y+dy)],tint(colors[c],.42),.55)
                ball(x+dx,y+dy,.055,colors[c])
            ax.add_patch(Circle((x,y),.51,fill=False,edgecolor=tint(review,.45),lw=.6,linestyle=(0,(3,3))))
            ax.add_patch(Circle((x,y),.085,facecolor='white',edgecolor=review,lw=1.1,zorder=6))
        def ranking(x,y):
            for i,w in enumerate([.70,.59,.47,.38,.28]):
                ax.add_patch(Rectangle((x,y-i*.18),w,.105,facecolor=chromatin if i==0 else tint(review,.30+.11*i),edgecolor='none'))
        def person(x,y):
            ax.add_patch(Circle((x,y+.18),.10,facecolor=tint(review,.25),edgecolor=review,lw=.75))
            ax.add_patch(Polygon([(x-.22,y-.22),(x-.16,y-.01),(x,y+.04),(x+.16,y-.01),(x+.22,y-.22)],
                                closed=True,facecolor=tint(review,.1),edgecolor=review,lw=.75))
            line([(x-.05,y+.01),(x,y-.1),(x+.05,y+.01)],review,.6)

        # A. Source-informed inference; arrows denote the computations in the study.
        text(.12,10.78,'A',14,weight='bold',ha='left')
        text(.62,10.78,'Prediction and expert-review workflow',13,weight='bold',ha='left')
        for x,label in [(1.28,'Inputs'),(4.20,'Representation'),(8.0,'Prediction and uncertainty'),(15.42,'Interpretation outputs')]:
            text(x,10.23,label,11.3,weight='bold')

        text(1.27,9.65,'Jurkat MPRA',10.7,source)
        cell(.94,9.12,source)
        line([(1.40,9.12),(2.02,9.12)],source,.7)
        for xx in np.linspace(1.4,2.02,4):line([(xx,9.08),(xx,9.16)],source,.5)
        ball(1.73,9.24,.06,source);line([(1.73,9.12),(1.73,9.24)],source,.7)
        text(1.31,8.58,r'$y_J,\;s_J$',10,source)
        arrow([(2.35,9.12),(3.08,9.12)],source)
        vector(4.10,9.12,3,source,h=.42)
        text(4.20,8.58,'Measured + shrunk effect',10.1,source)
        arrow([(5.35,9.12),(6.58,9.12)],source)

        text(1.27,8.02,'Regulatory variant',10.7,molecular)
        dna(1.28,7.51)
        text(1.28,7.10,'REF / ALT',9.3,molecular)
        arrow([(2.35,7.51),(3.08,7.51)],molecular)
        vector(4.10,7.51,4,molecular)
        text(4.20,6.92,'Alleles and motif effects',10.1,molecular)
        line([(5.35,7.51),(6.05,7.51)],molecular)

        text(1.27,6.42,'Immune-cell chromatin',10.3,chromatin)
        track(1.28,5.96,source);track(1.28,5.52,chromatin)
        arrow([(2.35,5.96),(3.08,5.96)],chromatin)
        vector(4.10,5.96,3,chromatin,h=.48)
        text(4.20,5.40,'Jurkat / CD4 DNase',10.1,chromatin)
        line([(5.35,5.96),(6.05,5.96),(6.05,8.14)],chromatin)
        arrow([(6.05,8.14),(6.58,8.14)],molecular)

        cuboid(6.62,7.46,2.80,2.16,molecular,.14,.035)
        text(8.02,9.32,'Effect predictors',11.7,molecular,'bold')
        text(8.02,8.86,'Source-only references',10)
        line([(6.85,8.57),(9.18,8.57)],tint(molecular,.28),.5)
        text(8.02,8.20,'Molecular augmentation',10)
        text(8.02,7.76,'Ridge / boosting',9.4,muted)

        cuboid(6.62,5.04,2.80,1.63,review,.14,.035)
        text(8.02,6.37,'Difficulty scores',11.7,review,'bold')
        text(8.02,5.96,'Source SE / learned error',9.7)
        text(8.02,5.56,'Rough memberships',9.9)
        text(8.02,4.66,'Neighborhood detail in B',9.1,review)
        arrow([(8.02,7.42),(8.02,6.84)],review)
        arrow([(6.05,5.96),(6.58,5.96)],chromatin)
        line([(5.62,9.12),(5.62,6.90)],source,.65)
        arrow([(5.62,6.90),(6.32,6.90),(6.32,6.39),(6.58,6.39)],source)

        # Point prediction and biological target.
        text(15.42,9.66,'Primary CD4 effect',11.7,response,'bold')
        cell(14.35,9.02,review)
        line([(15.05,8.92),(16.70,8.92)],response,.8)
        for xx in np.linspace(15.05,16.70,6):line([(xx,8.85),(xx,8.99)],response,.55)
        line([(15.99,8.92),(15.99,9.23)],response,.8);ball(15.99,9.23,.065,response)
        text(15.90,8.50,r'$\hat y_T$',12,response)
        arrow([(9.60,8.98),(13.54,8.98)],response)

        # Selected score determines a finite review queue, not a corrected effect.
        cuboid(10.68,5.32,1.78,1.32,review,.09,.035)
        ranking(11.17,6.20)
        text(11.58,4.94,'Budgeted ranking',10.3,review,'bold')
        arrow([(9.60,5.92),(10.60,5.92)],review)
        arrow([(12.59,6.14),(13.03,6.14),(13.03,7.10),(13.63,7.10)],response)
        arrow([(12.59,5.62),(13.03,5.62),(13.03,5.15),(13.63,5.15)],review)
        text(15.42,7.63,'Retained estimates',11.3,response,'bold')
        for i,h in enumerate([.31,.45,.24,.40]):
            xx=14.77+i*.30
            line([(xx,6.92),(xx,6.92+h)],response,.8);ball(xx,6.92+h,.048,response)
        text(15.42,6.47,'Effect + uncertainty evidence',9.8,response)
        text(15.42,5.80,'Expert-review candidates',11.1,review,'bold')
        person(14.78,5.15);worksheet(15.80,5.13,review)
        text(15.42,4.43,'Variant evidence and review reasons',9.8,review)

        # B. Meaningful algorithmic expansion, in the same reading direction.
        line([(.12,3.96),(17.84,3.96)],tint(ink,.23),.55)
        text(.12,3.61,'B',14,weight='bold',ha='left')
        text(.62,3.61,'Neighborhood uncertainty and review ranking',12.7,weight='bold',ha='left')
        for yy,txt,col in [(2.72,'Source effect',source),(2.22,'Molecular evidence',molecular),(1.72,'Chromatin signal',chromatin)]:
            ball(.45,yy,.075,col);text(.68,yy,txt,10.1,ha='left')
        arrow([(3.03,2.22),(3.61,2.22)],review)
        neighborhood(4.28,2.22)
        text(4.28,1.20,'Training neighborhoods',9.8,review)
        text(4.28,.75,f'$k={neighbors}$',10,muted)
        arrow([(4.98,2.22),(5.73,2.22)],review)
        for i,(h,col) in enumerate(zip([.60,.32,.42,.22],[molecular,chromatin,source,review])):
            xx=6.00+i*.24
            cuboid(xx,1.96,.16,h,col,.045,.35)
        text(6.44,3.02,'Crisp / soft membership',9.8,review)
        text(6.44,1.20,'Class support',10,review)
        text(6.44,.75,r'$p_i(c)=\sum_j w_{ij}\,\mu_j(c)$',10)
        arrow([(7.16,2.22),(7.70,2.22)],review)
        ax.add_patch(Polygon([(7.83,2.22),(8.74,2.83),(9.65,2.22),(8.74,1.61)],
                             facecolor=tint(review,.035),edgecolor=review,lw=.7))
        text(8.74,2.22,r'$1-\max_c p_i(c)$',10.1,review)
        text(8.74,1.20,'Rough difficulty',10,review)
        text(8.74,.75,'Compared with SE / learned error',8.8,muted)
        arrow([(9.73,2.22),(10.44,2.22)],review)
        ranking(10.60,2.58)
        text(10.96,1.20,'Rank variants',10,review)
        arrow([(11.44,2.22),(12.12,2.22)],review)
        ax.add_patch(Polygon([(12.22,2.22),(12.92,2.72),(13.62,2.22),(12.92,1.72)],
                             facecolor=tint(chromatin,.04),edgecolor=chromatin,lw=.7))
        text(12.92,2.22,'Top budget',9.6,chromatin)
        text(12.92,1.20,'Fixed review capacity',9.8,chromatin)
        arrow([(13.69,2.22),(14.16,2.22),(14.16,2.79),(14.65,2.79)],review)
        arrow([(14.16,2.22),(14.16,1.63),(14.65,1.63)],response)
        worksheet(15.20,2.79,review)
        text(15.88,2.79,'Review queue',10.3,review,ha='left')
        vector(15.12,1.63,3,response,w=.14,h=.34)
        text(15.88,1.63,'Retain estimate',10.3,response,ha='left')

        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for label in labels:
            box=label.get_window_extent(renderer)
            if not fig.bbox.contains(box.x0,box.y0) or not fig.bbox.contains(box.x1,box.y1):
                raise ValueError(f'Architecture label outside canvas: {label.get_text()}')
        for extension in ('svg','pdf','png'):
            fig.savefig(Path(output)/f'architecture.{extension}',dpi=600,facecolor='white')
        plt.close(fig)


def regenerate_report(output):
    """Rebuild every table/figure from saved numerical results; no model refitting."""
    import json
    output=Path(output)
    manifest=json.loads((output/'artifact_manifest.json').read_text())
    for name in REPORT_INPUTS:
        if manifest.get(name) != sha256(output/name):
            raise ValueError(f'Report input hash mismatch: {name}; saved numerical inputs must remain unchanged.')
    protocol=json.loads((output/'protocol.json').read_text())
    results=json.loads((output/'results.json').read_text())
    diagnostics=json.loads((output/'replicate_diagnostics.json').read_text())
    write_study_summary(output,protocol,results,diagnostics)
    write_json(output/'report_provenance.json',{'reporting_code_sha256':sha256(Path(__file__)),
        'input_sha256':{name:sha256(output/name) for name in REPORT_INPUTS}})
    write_json(output/'artifact_manifest.json',{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'})
