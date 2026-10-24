import pickle, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
R = pickle.load(open('results_main.pkl','rb'))
probs = [("10c1","10-bar, case 1"),("10c2","10-bar, case 2"),("25","25-bar"),("72","72-bar")]
algs = ["Jaya","FA","Hybrid","Hybrid (unscaled γ)"]
out = {}
for pk, pname in probs:
    fstar = min(min(r['best'] for r in R[(pk,a)]) for a in algs)
    out[pk] = {'fstar': fstar, 'rows': {}}
    for a in algs:
        b = np.array([r['best'] for r in R[(pk,a)]])
        # NFE to reach within 0.5% of fstar
        hits = []
        for r in R[(pk,a)]:
            tr = r['trace']; ok = np.where(tr[:,1] <= fstar*1.005)[0]
            hits.append(tr[ok[0],0] if len(ok) else np.nan)
        hits = np.array(hits)
        out[pk]['rows'][a] = dict(best=b.min(), mean=b.mean(), worst=b.max(), sd=b.std(ddof=1),
                                  succ=int(np.sum(b <= fstar*1.001)), reach=int(np.sum(~np.isnan(hits))),
                                  nfe_mean=float(np.nanmean(hits)) if np.any(~np.isnan(hits)) else np.nan)
    hb = [r['best'] for r in R[(pk,'Hybrid')]]
    for a in ["Jaya","FA","Hybrid (unscaled γ)"]:
        ob = [r['best'] for r in R[(pk,a)]]
        p = mannwhitneyu(hb, ob, alternative='two-sided').pvalue
        out[pk]['p_vs_'+a] = p
for pk,_ in probs:
    print(pk, 'f*=%.2f'%out[pk]['fstar'])
    for a in algs:
        r=out[pk]['rows'][a]; print('  %-30s best %.2f mean %.2f worst %.2f sd %.2f succ %d reach %d nfe %.0f'%(a,r['best'],r['mean'],r['worst'],r['sd'],r['succ'],r['reach'],r['nfe_mean']))
    print('  p:', {k:round(v,4) for k,v in out[pk].items() if k.startswith('p_')})
json.dump(out, open('summary.json','w'), indent=1, default=float)

# Figures: convergence (median best feasible vs NFE) and diversity
cols = {"Jaya":"#1B3A5C","FA":"#C0392B","Hybrid":"#27AE60","Hybrid (unscaled γ)":"#8E7CC3"}
ls = {"Jaya":"-","FA":"--","Hybrid":"-","Hybrid (unscaled γ)":":"}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8})
fig, axs = plt.subplots(2,4, figsize=(7.2,4.2), dpi=300)
for j,(pk,pname) in enumerate(probs):
    fstar = out[pk]['fstar']
    for a in algs:
        trs = R[(pk,a)]; n = min(len(r['trace']) for r in trs)
        nfe = trs[0]['trace'][:n,0]
        W = np.array([r['trace'][:n,1] for r in trs]); D = np.array([r['trace'][:n,2] for r in trs])
        gap = (np.median(W,axis=0) - fstar)/fstar*100
        gap = np.where(np.isfinite(gap), gap, np.nan)
        axs[0,j].semilogy(nfe, np.maximum(gap,1e-4), ls[a], color=cols[a], lw=1, label=a)
        axs[1,j].semilogy(nfe, np.median(D,axis=0), ls[a], color=cols[a], lw=1)
    axs[0,j].set_title(pname, fontsize=8)
    axs[0,j].set_ylim(1e-3, 1e2)
    axs[1,j].set_xlabel('Structural analyses')
    for i in range(2): axs[i,j].tick_params(labelsize=6.5); axs[i,j].xaxis.get_offset_text().set_fontsize(6)
axs[0,0].set_ylabel('Median gap to best (%)'); axs[1,0].set_ylabel('Population diversity')
h,l = axs[0,0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=4,fontsize=7,frameon=False,bbox_to_anchor=(0.5,-0.02))
plt.tight_layout(rect=(0,0.05,1,1))
plt.savefig('fig2_convergence.png',dpi=300,bbox_inches='tight',facecolor='white')
print('figs ok')
