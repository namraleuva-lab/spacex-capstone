import pandas as pd, numpy as np, sqlite3, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sns.set_theme(style="whitegrid")
d=pd.read_csv('data/dataset_part_2.csv'); X=pd.read_csv('data/dataset_part_3.csv')
dash=pd.read_csv('data/spacex_launch_dash.csv')
d['Year']=pd.to_datetime(d.Date).dt.year
R={}
def save(name): plt.tight_layout(); plt.savefig(f'charts/{name}.png',dpi=130); plt.close()
pal={0:'#d62728',1:'#2ca02c'}
def scat(x,y,name,xl,yl,title):
    plt.figure(figsize=(9,4.2)); sns.scatterplot(data=d,x=x,y=y,hue='Class',palette=pal,s=70,alpha=.85)
    plt.xlabel(xl);plt.ylabel(yl);plt.title(title);plt.legend(title='Landing (1=success)'); save(name)
scat('FlightNumber','LaunchSite','e1','Flight Number','Launch Site','Flight Number vs Launch Site')
scat('PayloadMass','LaunchSite','e2','Payload Mass (kg)','Launch Site','Payload Mass vs Launch Site')
scat('FlightNumber','Orbit','e4','Flight Number','Orbit','Flight Number vs Orbit')
scat('PayloadMass','Orbit','e5','Payload Mass (kg)','Orbit','Payload Mass vs Orbit')
orb=d.groupby('Orbit').Class.mean().sort_values(ascending=False); R['orbit']=orb.round(3).to_dict()
plt.figure(figsize=(9,4.2)); sns.barplot(x=orb.index,y=orb.values,color='#1F6FEB'); plt.ylabel('Success rate');plt.xlabel('Orbit');plt.title('Success Rate by Orbit Type');plt.ylim(0,1.05); save('e3')
yr=d.groupby('Year').Class.mean(); R['year']=yr.round(3).to_dict()
plt.figure(figsize=(9,4.2)); plt.plot(yr.index,yr.values,marker='o',color='#1F6FEB'); plt.xlabel('Year');plt.ylabel('Success rate');plt.title('Yearly Landing Success Rate');plt.ylim(0,1.05); save('e6')
site=d.groupby('LaunchSite').Class.agg(['mean','count','sum']); R['site']=site.round(3).to_dict('index'); R['overall']=round(d.Class.mean(),4); R['n']=len(d)
R['first_succ_year']=int(d[d.Class==1].Year.min()); R['y2020']=R['year'].get(max(R['year']))
# SQL
con=sqlite3.connect(':memory:'); d.drop(columns=['Year']).to_sql('SPACEX',con,index=False)
dash.rename(columns=lambda c:c.replace(' ','_').replace('(','').replace(')','')).to_sql('DASH',con,index=False)
Q={
'q1':("SELECT DISTINCT LaunchSite FROM SPACEX",),
'q2':("SELECT FlightNumber, Date, LaunchSite, Orbit, Outcome FROM SPACEX WHERE LaunchSite LIKE 'CCA%' LIMIT 5",),
'q3':("SELECT SUM(PayloadMass) AS total_payload_ISS_kg FROM SPACEX WHERE Orbit='ISS'",),
'q4':("SELECT AVG(Payload_Mass_kg) AS avg_payload_F9_v1_1 FROM DASH WHERE Booster_Version LIKE 'F9 v1.1%'",),
'q5':("SELECT MIN(Date) AS first_ground_pad_success FROM SPACEX WHERE Outcome='True RTLS'",),
'q6':("SELECT Serial, PayloadMass FROM SPACEX WHERE Outcome='True ASDS' AND PayloadMass BETWEEN 4000 AND 6000",),
'q7':("SELECT Outcome, COUNT(*) AS n FROM SPACEX GROUP BY Outcome ORDER BY n DESC",),
'q8':("SELECT DISTINCT Serial, PayloadMass FROM SPACEX WHERE PayloadMass=(SELECT MAX(PayloadMass) FROM SPACEX)",),
'q9':("SELECT substr(Date,6,2) AS month, Serial, LaunchSite FROM SPACEX WHERE Outcome LIKE 'False ASDS' AND substr(Date,1,4)='2015'",),
'q10':("SELECT Outcome, COUNT(*) AS n FROM SPACEX WHERE Date BETWEEN '2010-06-04' AND '2017-03-20' GROUP BY Outcome ORDER BY n DESC",),
}
R['sql']={}
for k,(q,) in Q.items():
    df=pd.read_sql(q,con); R['sql'][k]={'q':q,'rows':df.round(2).values.tolist(),'cols':df.columns.tolist()}
# ML
Y=d.Class.to_numpy(); Xs=StandardScaler().fit_transform(X)
Xtr,Xte,ytr,yte=train_test_split(Xs,Y,test_size=0.2,random_state=2); R['split']=[len(Xtr),len(Xte)]
models={'Logistic Regression':(LogisticRegression(),{'C':[0.01,0.1,1],'penalty':['l2'],'solver':['lbfgs']}),
'SVM':(SVC(),{'kernel':['linear','rbf','poly','sigmoid'],'C':np.logspace(-3,3,5),'gamma':np.logspace(-3,3,5)}),
'Decision Tree':(DecisionTreeClassifier(random_state=12345),{'criterion':['gini','entropy'],'splitter':['best','random'],'max_depth':[2*n for n in range(1,10)],'max_features':['sqrt'],'min_samples_leaf':[1,2,4],'min_samples_split':[2,5,10]}),
'KNN':(KNeighborsClassifier(),{'n_neighbors':list(range(1,11)),'algorithm':['auto','ball_tree','kd_tree','brute'],'p':[1,2]})}
R['ml']={}; best=None
for n,(m,g) in models.items():
    gs=GridSearchCV(m,g,cv=10).fit(Xtr,ytr); acc=gs.score(Xte,yte)
    R['ml'][n]={'cv':round(gs.best_score_,4),'test':round(acc,4),'params':{k:(float(v) if isinstance(v,(np.floating,)) else v) for k,v in gs.best_params_.items()}}
    if best is None or acc>best[1] or (acc==best[1] and gs.best_score_>R['ml'][best[0]]['cv']): best=(n,acc,gs)
R['best']=best[0]
plt.figure(figsize=(8,4.2)); a=[R['ml'][n]['test'] for n in models]; ax=sns.barplot(x=list(models),y=a,color='#1F6FEB')
for i,v in enumerate(a): ax.text(i,v+.01,f'{v:.1%}',ha='center')
plt.ylim(0,1.05);plt.ylabel('Test accuracy');plt.title('Classification Accuracy by Model (test set, n=%d)'%len(yte)); save('m1')
cm=confusion_matrix(yte,best[2].predict(Xte)); R['cm']=cm.tolist()
fig,ax=plt.subplots(figsize=(5,4.2)); ConfusionMatrixDisplay(cm,display_labels=['did not land','landed']).plot(ax=ax,cmap='Blues',colorbar=False); ax.grid(False); plt.title('Confusion Matrix: '+best[0]); save('m2')
# Dash charts
dash.columns=['i','FlightNumber','Site','Outcome','Class','Payload','Booster','Cat']; dash['Site']=dash['Site'].replace({'CCAFS LC-40':'CCAFS SLC-40'})
s=dash.groupby('Site').Class.agg(['sum','count']); s['rate']=s['sum']/s['count']; R['dash_site']=s.round(3).to_dict('index')
plt.figure(figsize=(6,4.5)); plt.pie(s['sum'],labels=s.index,autopct='%1.0f%%',colors=sns.color_palette('Set2')); plt.title('Total Successful Launches by Site'); save('d1')
top=s.rate.idxmax(); R['dash_top']=top; c=dash[dash.Site==top].Class.value_counts()
plt.figure(figsize=(6,4.5)); plt.pie([c.get(1,0),c.get(0,0)],labels=['Success','Failure'],autopct='%1.1f%%',colors=['#2ca02c','#d62728']); plt.title(f'Success vs Failure: {top}'); save('d2')
plt.figure(figsize=(9,4.2)); sns.scatterplot(data=dash,x='Payload',y='Class',hue='Cat',s=70,alpha=.8); plt.xlabel('Payload Mass (kg)');plt.ylabel('Class (1=success)');plt.title('Payload vs Launch Outcome by Booster Version');save('d3')
lo=dash[dash.Payload<=4000].Class.mean(); hi=dash[dash.Payload>4000].Class.mean(); R['pay_lo_hi']=[round(lo,3),round(hi,3)]
R['dash_cat']=dash.groupby('Cat').Class.agg(['mean','count']).round(3).to_dict('index')
R['dash_payload_range']={'0-2000':round(dash[dash.Payload<=2000].Class.mean(),3),'2000-4000':round(dash[(dash.Payload>2000)&(dash.Payload<=4000)].Class.mean(),3),'4000-6000':round(dash[(dash.Payload>4000)&(dash.Payload<=6000)].Class.mean(),3),'6000+':round(dash[dash.Payload>6000].Class.mean(),3)}
# Map + proximity
def hav(a,b,c,e):
    p=np.pi/180; x=np.sin((c-a)*p/2)**2+np.cos(a*p)*np.cos(c*p)*np.sin((e-b)*p/2)**2; return 12742*np.arcsin(np.sqrt(x))
sites=d.groupby('LaunchSite')[['Latitude','Longitude']].first(); sites['rate']=site['mean']
cities={'CCAFS SLC 40':('Cocoa Beach',28.32,-80.6076),'KSC LC 39A':('Titusville',28.6122,-80.8076),'VAFB SLC 4E':('Lompoc',34.6392,-120.4579)}
R['sites']={}
for s_,r in sites.iterrows():
    cn,cl,co=cities[s_]; R['sites'][s_]={'lat':round(r.Latitude,4),'lon':round(r.Longitude,4),'city':cn,'city_km':round(hav(r.Latitude,r.Longitude,cl,co),1),'rate':round(r.rate,3)}
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for a_,(xl,xr,yl,yr_) in zip(ax,[(-125,-70,24,42),(-82,-79.5,28,29.2)]): pass
ax[0].scatter(sites.Longitude,sites.Latitude,s=150,c='#1F6FEB'); ax[0].set_xlim(-125,-75); ax[0].set_ylim(25,40)
for s_,r in sites.iterrows(): ax[0].annotate(s_,(r.Longitude,r.Latitude),xytext=(6,6),textcoords='offset points',fontsize=8)
ax[0].set_title('Launch sites (US coasts)');ax[0].set_xlabel('Longitude');ax[0].set_ylabel('Latitude')
for o,g in d.groupby('LaunchSite'): ax[1].scatter(g.Longitude,g.Latitude,s=0)
fl=d[d.LaunchSite!='VAFB SLC 4E'].groupby('LaunchSite'); 
for (o,g),mk in zip(fl,['o','s']): ax[1].scatter(g.Longitude+np.random.RandomState(1).uniform(-.002,.002,len(g)),g.Latitude,c=g.Class.map({0:'#d62728',1:'#2ca02c'}),marker=mk,s=40,label=o)
ax[1].set_title('Florida sites: launch outcomes (green=success, red=failure)'); ax[1].set_xlabel('Longitude');ax[1].legend(fontsize=8); save('f1')
fig,ax=plt.subplots(figsize=(8,4.2)); ls=d.groupby('LaunchSite').Class.agg(['sum',lambda s:(1-s).sum()]); ls.columns=['Success','Failure']; ls.plot(kind='bar',stacked=True,color=['#2ca02c','#d62728'],ax=ax); ax.set_ylabel('Launches'); ax.set_title('Success and failure counts per launch site'); plt.xticks(rotation=0); save('f2')
json.dump(R,open('results.json','w'),indent=1,default=str); print(json.dumps(R,indent=1,default=str)[:6000])
