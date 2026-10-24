import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trusses import *
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9})
fig = plt.figure(figsize=(7.2, 3.0), dpi=300)
# (a) 10-bar
ax = fig.add_subplot(1, 3, 1)
t = ten_bar(1)
for m,(a,b) in enumerate(t.members):
    xa,ya = t.nodes[a]; xb,yb = t.nodes[b]
    ax.plot([xa,xb],[ya,yb],'k-',lw=1.2)
    f = 0.5 if (ya==yb or xa==xb) else 0.27
    mx,my = xa+f*(xb-xa), ya+f*(yb-ya)
    ax.text(mx, my, f'{m+1}', fontsize=7, color='#1B3A5C', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.15', fc='white', ec='#1B3A5C', lw=0.5))
for n,(x,y) in enumerate(t.nodes):
    ax.plot(x,y,'o',ms=4,mfc='white',mec='k')
    ax.text(x+(18 if x>0 else -18), y+(18 if y>0 else -26), str(n+1), fontsize=8, ha='center')
for n in (4,5):
    x,y=t.nodes[n]; ax.plot([x-10,x-10],[y-30,y+30],'k-',lw=2)
for n in (1,3):
    x,y=t.nodes[n]; ax.annotate('',xy=(x,y-5),xytext=(x,y-80),arrowprops=dict(arrowstyle='<-',lw=1,color='#b22222'))
    ax.text(x+8,y-75,'P',fontsize=8,color='#b22222')
[ax.annotate('',xy=(x0,440),xytext=(x0+360,440),arrowprops=dict(arrowstyle='<->',lw=0.6)) for x0 in (0,360)]
ax.text(180,455,'360 in',ha='center',fontsize=7); ax.text(540,455,'360 in',ha='center',fontsize=7)
ax.annotate('',xy=(760,0),xytext=(760,360),arrowprops=dict(arrowstyle='<->',lw=0.6)); ax.text(775,180,'360 in',rotation=90,va='center',fontsize=7)
ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-60,800); ax.set_ylim(-110,490)
ax.set_title('(a) 10-bar planar truss',fontsize=9)
# (b) 25-bar, (c) 72-bar 3D
for k,(tr,title,el,az) in enumerate([(twentyfive_bar(),'(b) 25-bar space truss',18,-60),(seventytwo_bar(),'(c) 72-bar space truss',15,-55)]):
    ax = fig.add_subplot(1,3,k+2,projection='3d')
    for a,b in tr.members:
        p,q=tr.nodes[a],tr.nodes[b]
        ax.plot(*zip(p,q),'k-',lw=0.8)
    ax.scatter(*tr.nodes.T,s=5,c='white',edgecolors='k',depthshade=False)
    ax.view_init(el,az); ax.set_axis_off(); ax.set_title(title,fontsize=9)
    ax.set_box_aspect(np.ptp(tr.nodes,axis=0)+1e-9)
plt.tight_layout()
plt.savefig('fig1_geometry.png',dpi=300,bbox_inches='tight',facecolor='white')
