"""Report-only replicate sensitivity; no target counts enter a predictor or selector."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_counts(path):
    counts=pd.read_csv(Path(path),sep='\t',index_col=0)
    dna=[c for c in counts if c.startswith('OLJR.C_Plasmid_')]
    rna=[c for c in counts if c.startswith('OLJR.C_Tcell_')]
    if len(dna)<4 or len(rna)<4:
        raise ValueError('Expected at least four DNA and four T-cell RNA replicates.')
    if not np.isfinite(counts[dna+rna].to_numpy(float)).all() or (counts[dna+rna]<0).any().any():
        raise ValueError('Replicate counts must be finite and nonnegative.')
    return counts,dna,rna


def paired_counts(counts, columns):
    identifiers=pd.Series(counts.index.astype(str)).str.extract(r'^(?P<variant_id>[^:]+:[^:]+:[ACGT]+:[ACGT]+):(?P<tag>[AR]):wC$')
    identifiers['row']=np.arange(len(counts))
    identifiers=identifiers.dropna()
    if identifiers.duplicated(['variant_id','tag']).any():
        raise ValueError('Ambiguous duplicate allele constructs; resolve explicitly before analysis.')
    a=identifiers[identifiers.tag.eq('A')].set_index('variant_id')
    r=identifiers[identifiers.tag.eq('R')].set_index('variant_id')
    ids=a.index.intersection(r.index)
    values=counts[columns].to_numpy(float)
    return ids, values[a.loc[ids,'row']],values[r.loc[ids,'row']]


def replicate_diagnostics(path, draws=50, seed=20261122):
    if draws < 1: raise ValueError('At least one replicate partition required.')
    counts,dna,rna=load_counts(path)
    ids,a,r=paired_counts(counts,dna+rna)
    d=len(dna); n=len(rna)
    rng=np.random.default_rng(seed)
    results={k:[] for k in ('shared_dna','disjoint_dna')}
    # Identical estimator in both arms. DNA sharing is not the only change:
    # partitioning also reduces the number of DNA controls per half.
    rna_skew=np.log2(a[:,d:]+1)-np.log2(r[:,d:]+1)
    def dna_skew(cols):
        return np.log2(a[:,cols].mean(axis=1)+1)-np.log2(r[:,cols].mean(axis=1)+1)
    shared=dna_skew(np.arange(d))
    for _ in range(draws):
        rp=rng.permutation(n); dp=rng.permutation(d)
        left,right=rna_skew[:,rp[:n//2]].mean(1),rna_skew[:,rp[n//2:]].mean(1)
        for label,dl,dr in [('shared_dna',shared,shared),
                            ('disjoint_dna',dna_skew(dp[:d//2]),dna_skew(dp[d//2:]))]:
            results[label].append(float(spearmanr(left-dl,right-dr).statistic))
    return {'paired_variants':len(ids),'rna_replicates':n,'dna_replicates':d,'partitions':draws,
        'rna_half_sizes':[n//2,n-n//2],'dna_half_sizes':[d//2,d-d//2],
        'correlations':{k:{'median_spearman':float(np.median(v)),
                          'partition_percentiles_2_5_97_5':np.quantile(v,[.025,.975]).tolist()} for k,v in results.items()},
        'estimator':'Mean per-RNA log2((ALT+1)/(REF+1)) minus log2((mean DNA ALT+1)/(mean DNA REF+1)).',
        'scope':'All unambiguous single-variant allele pairs in normalized counts, before endpoint-cohort QC.',
        'limitations':['Partition percentiles are not confidence intervals.',
                      'Disjoint DNA halves use fewer controls; differences cannot isolate shared-normalization bias.',
                      'Approximate count estimator does not reproduce MPRAmodel/DESeq2 published effects.',
                      'Correlations do not establish a prediction ceiling or latent biological truth.'],
        'usage':'Report-only; target replicates are forbidden in prediction and routing.'}
