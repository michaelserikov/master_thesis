import numpy as np, matplotlib.pyplot as plt

base=-1.384
lp=np.array([1,3,6,8,11,13,16,18,21,22])
goal=np.array([-1.026,-1.143,-1.246,-1.301,-1.335,-1.355,-1.371,-1.376,-1.382,-1.384])
evpi=goal-base
year=2025+(lp-1)*5

fig,ax=plt.subplots(figsize=(9,5))
ax.plot(year,evpi,'-o',color='#22456c',lw=0.8,ms=5,label='EVPI($\\alpha$) simulated decline')
ax.axhline(0,color='k',lw=.5)
ax.set_xticks(year)
ax.tick_params(axis="x",labelrotation=45)
ax.set_yticks(np.arange(0,0.4,0.05))
ax.set_xlabel('Learning year'); ax.set_ylabel('EVPI($\\alpha$)')
ax.set_title('Value of information decay with delayed learning')
ax.grid(alpha=.3); ax.minorticks_on(); ax.legend()
fig.tight_layout(); fig.savefig("old_evpi_decay_curve.png",dpi=130)
