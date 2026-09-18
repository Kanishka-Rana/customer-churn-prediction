import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier

st.set_page_config(page_title='Customer Churn Intelligence', page_icon='📊', layout='wide')
ROOT=Path(__file__).resolve().parent.parent
RAW=ROOT/'Data'/'Raw'; PROCESSED=ROOT/'Data'/'Processed'

st.markdown('''<style>
.stApp{background:#f6f7fb}[data-testid="stSidebar"]{background:#111827}[data-testid="stSidebar"] *{color:#f9fafb!important}
[data-testid="stFileUploader"] section{background:#1f2937!important;border:1px solid #4b5563!important;border-radius:12px}
[data-testid="stFileUploader"] section *{color:#f9fafb!important}[data-testid="stFileUploader"] button{background:#fff!important;color:#111827!important;border:1px solid #d1d5db!important}
[data-testid="stFileUploader"] button *{color:#111827!important}.block-container{padding-top:1.3rem}.hero{background:linear-gradient(135deg,#111827,#374151);color:#fff;padding:30px;border-radius:20px;margin-bottom:22px}.hero h1{margin:0 0 5px;font-size:2.2rem}.hero p{margin:0;opacity:.88}
[data-testid="stMetric"]{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:12px}
</style>''',unsafe_allow_html=True)

NUM=['Age','Tenure_Months','Monthly_Charges','Login_Count_30D','Login_Count_7D','Usage_Minutes_30D','Usage_Minutes_7D','Avg_Session_Minutes_30D','Avg_Session_Minutes_7D','Support_Tickets','Open_Tickets','Avg_Resolution_Hours','Payment_Delays','Discount_Percent','NPS_Score','Days_Since_Last_Login']
CAT=['Subscription_Plan','Billing_Frequency']

@st.cache_data
def read_csv(path):
    df=pd.read_csv(path); df.columns=df.columns.astype(str).str.strip().str.replace(' ','_',regex=False); return df

def find_data():
    p=PROCESSED/'customer_churn_features.csv'
    if p.exists(): return read_csv(p),p.name
    for p in [RAW/'customer_churn_dataset-1.csv',RAW/'customer_churn_dataset.csv',ROOT/'customer_churn_dataset.csv']:
        if p.exists(): return engineer(read_csv(p)),p.name
    return None,None

def engineer(df):
    x=df.copy()
    for c in NUM:
        if c in x: x[c]=pd.to_numeric(x[c],errors='coerce')
    if {'Login_Count_30D','Login_Count_7D'}<=set(x.columns):
        e=x.Login_Count_30D/4.285714285714286
        x['Expected_Login_7D']=e
        x['Login_Drop_Percent']=((e-x.Login_Count_7D)/e.replace(0,np.nan)).clip(lower=0)
    if {'Usage_Minutes_30D','Usage_Minutes_7D'}<=set(x.columns):
        e=x.Usage_Minutes_30D/4.285714285714286
        x['Expected_Usage_7D']=e
        x['Usage_Drop_Percent']=((e-x.Usage_Minutes_7D)/e.replace(0,np.nan)).clip(lower=0)
    if {'Support_Tickets','Avg_Resolution_Hours'}<=set(x.columns): x['Support_Friction_Index']=x.Support_Tickets*x.Avg_Resolution_Hours
    if {'Login_Count_7D','Usage_Minutes_7D'}<=set(x.columns): x['Engagement_Score']=x.Login_Count_7D+x.Usage_Minutes_7D/10
    if 'Days_Since_Last_Login' in x: x['High_Inactivity']=(x.Days_Since_Last_Login>=14).astype(int)
    if 'Payment_Delays' in x: x['Payment_Risk']=(x.Payment_Delays>0).astype(int)
    if 'Login_Count_7D' in x: x['Low_Engagement']=(x.Login_Count_7D<5).astype(int)
    return x

def split_xy(df):
    x=df.drop(columns=['Customer_ID','Churn','index'],errors='ignore').copy()
    y=pd.to_numeric(df['Churn'],errors='coerce').astype(int) if 'Churn' in df else None
    cats=[c for c in CAT if c in x]; nums=[c for c in x.columns if c not in cats]
    for c in nums: x[c]=pd.to_numeric(x[c],errors='coerce')
    x[nums]=x[nums].replace([np.inf,-np.inf],np.nan).fillna(0)
    for c in cats: x[c]=x[c].fillna('Unknown').astype(str)
    return x,y

def preprocessor(x):
    cats=[c for c in CAT if c in x]; nums=[c for c in x.columns if c not in cats]
    return ColumnTransformer([('num',StandardScaler(),nums),('cat',OneHotEncoder(handle_unknown='ignore'),cats)])

@st.cache_resource
def train_models(df):
    x,y=split_xy(df)
    if y is None: raise ValueError('Churn column is required.')
    Xtr,Xte,ytr,yte=train_test_split(x,y,test_size=.20,random_state=42,stratify=y)
    def pipe(model): return Pipeline([('preprocessor',preprocessor(x)),('model',model)])
    models={
      'Logistic Regression':pipe(LogisticRegression(max_iter=1000)),
      'Random Forest':pipe(RandomForestClassifier(n_estimators=200,random_state=42,class_weight='balanced')),
      'XGBoost':pipe(XGBClassifier(n_estimators=200,max_depth=5,learning_rate=.05,random_state=42,eval_metric='logloss'))}
    for m in models.values(): m.fit(Xtr,ytr)
    return models,(Xtr,Xte,ytr,yte)

def score(model,x): return np.asarray(model.predict_proba(x))[:,1]
def tier(p): return 'High Risk' if p>=.70 else ('Medium Risk' if p>=.40 else 'Low Risk')

def evaluation(models,split):
    _,xt,_,yt=split; rows=[]; preds={}
    for name,m in models.items():
        p=score(m,xt); pred=(p>=.5).astype(int); preds[name]=pred
        rows.append({'Model':name,'Accuracy':accuracy_score(yt,pred),'Precision':precision_score(yt,pred,zero_division=0),'Recall':recall_score(yt,pred,zero_division=0),'F1 Score':f1_score(yt,pred,zero_division=0),'ROC-AUC':roc_auc_score(yt,p),'PR-AUC':average_precision_score(yt,p)})
    return pd.DataFrame(rows),preds

data,source=find_data()
if data is None: st.error('customer_churn_features.csv not found in Data/Processed.'); st.stop()

st.sidebar.markdown('## 📊 Churn Intelligence')
st.sidebar.caption('Subscription Customer Retention')
page=st.sidebar.radio('MENU',['🏠 Overview','📁 Data Explorer','🤖 Prediction','📈 Analytics','🧠 Explainability','📊 Model Evaluation'])
st.sidebar.divider(); st.sidebar.markdown('### Data Source')
upload=st.sidebar.file_uploader('Upload customer CSV',type=['csv'])
if upload:
    active=engineer(pd.read_csv(upload)); source=upload.name; st.sidebar.success('CSV uploaded')
else:
    active=data.copy(); st.sidebar.info('Using demo dataset')
st.sidebar.caption(f'Source: {source}'); st.sidebar.caption(f'Rows: {len(active):,}'); st.sidebar.caption(f'Columns: {len(active.columns):,}')

# Models are retrained from the same notebook configuration instead of loading old pickle files.
# This removes scikit-learn pickle-version errors while preserving the original training setup.
if 'Churn' in data:
    models,split=train_models(data)
else:
    models,split={},None

if page=='🏠 Overview':
    st.markdown('<div class="hero"><h1>Customer Churn Intelligence</h1><p>Predict churn risk, understand customer behaviour and turn insights into retention actions.</p></div>',unsafe_allow_html=True)
    churn=pd.to_numeric(active.get('Churn',pd.Series(dtype=float)),errors='coerce').fillna(0)
    total=len(active); n=int(churn.sum()); rate=n/total*100 if total else 0
    charges=pd.to_numeric(active.get('Monthly_Charges',pd.Series(dtype=float)),errors='coerce').sum()
    a,b,c,d=st.columns(4); a.metric('Total Customers',f'{total:,}'); b.metric('Churned Customers',f'{n:,}'); c.metric('Churn Rate',f'{rate:.1f}%'); d.metric('Monthly Charges',f'₹{charges:,.0f}')
    st.markdown('### Business Snapshot'); c1,c2=st.columns(2)
    if 'Churn' in active:
        z=active.Churn.map({0:'Retained',1:'Churned'}).value_counts().reset_index(); z.columns=['Status','Customers']; c1.plotly_chart(px.pie(z,names='Status',values='Customers',hole=.58,title='Customer Churn Distribution'),use_container_width=True)
    if 'Subscription_Plan' in active:
        z=active.Subscription_Plan.astype(str).value_counts().reset_index(); z.columns=['Plan','Customers']; c2.plotly_chart(px.bar(z,x='Plan',y='Customers',title='Customers by Subscription Plan'),use_container_width=True)
    c1,c2=st.columns(2)
    if {'Billing_Frequency','Churn'}<=set(active.columns):
        z=active.groupby('Billing_Frequency').Churn.mean().mul(100).reset_index(); z.columns=['Billing Frequency','Churn Rate']; c1.plotly_chart(px.bar(z,x='Billing Frequency',y='Churn Rate',title='Churn Rate by Billing Frequency'),use_container_width=True)
    if {'Days_Since_Last_Login','Churn'}<=set(active.columns): c2.plotly_chart(px.box(active,x='Churn',y='Days_Since_Last_Login',title='Inactivity vs Churn'),use_container_width=True)
    st.info('The dashboard covers behavioural feature engineering, supervised classification, model evaluation, SHAP explainability and actionable retention strategy.')

elif page=='📁 Data Explorer':
    st.title('Data Explorer'); a,b,c=st.columns(3); a.metric('Rows',f'{len(active):,}'); b.metric('Columns',f'{len(active.columns):,}'); c.metric('Missing Values',f'{int(active.isna().sum().sum()):,}')
    q=st.text_input('Search Customer ID'); view=active.copy()
    if q and 'Customer_ID' in view: view=view[view.Customer_ID.astype(str).str.contains(q,case=False,na=False)]
    st.dataframe(view,use_container_width=True,height=500)
    st.download_button('⬇ Download Dataset',active.to_csv(index=False).encode(),'customer_churn_dataset.csv','text/csv',use_container_width=True)

elif page=='🤖 Prediction':
    st.title('Churn Prediction'); st.caption('The models are trained automatically with the exact configuration from the model-training notebook, avoiding saved-pickle version conflicts.')
    if not models: st.error('Models could not be prepared.'); st.stop()
    selected=st.selectbox('Select Model',list(models)); threshold=st.slider('Churn probability threshold',.10,.90,.50,.05)
    st.dataframe(active.head(10),use_container_width=True)
    if st.button('🚀 Run Churn Prediction',type='primary',use_container_width=True):
        try:
            x,_=split_xy(active); p=score(models[selected],x); pred=(p>=threshold).astype(int); result=active.copy(); result['Churn Probability (%)']=np.round(p*100,2); result['Prediction']=np.where(pred,'Yes','No'); result['Risk Tier']=[tier(v) for v in p]
            a,b,c,d=st.columns(4); a.metric('Predicted Churn',f'{pred.sum():,}'); b.metric('Predicted Retained',f'{len(pred)-pred.sum():,}'); c.metric('Average Risk',f'{p.mean()*100:.1f}%'); d.metric('High Risk',f'{(p>=.70).sum():,}')
            st.success('Prediction completed successfully.'); st.dataframe(result.sort_values('Churn Probability (%)',ascending=False),use_container_width=True,height=500)
            st.download_button('⬇ Download Prediction Results',result.to_csv(index=False).encode(),'churn_prediction_results.csv','text/csv',use_container_width=True)
            z=result["Risk Tier"].value_counts().reindex(['High Risk','Medium Risk','Low Risk'],fill_value=0).reset_index(); z.columns=['Risk Tier','Customers']; st.plotly_chart(px.bar(z,x='Risk Tier',y='Customers',title='Customer Risk Distribution'),use_container_width=True)
            st.markdown('### Retention Strategy'); a,b,c=st.columns(3); a.markdown('**🔴 High Risk**'); a.write('Prioritize Customer Success outreach and tailor the intervention using churn reasons.'); b.markdown('**🟠 Medium Risk**'); b.write('Use targeted educational content, webinars or a suitable re-engagement offer.'); c.markdown('**🟢 Low Risk**'); c.write('Continue normal engagement and monitor behaviour for changes.')
        except Exception as e: st.error('Prediction could not be completed.'); st.exception(e)

elif page=='📈 Analytics':
    st.title('Customer Analytics'); nums=[c for c in active.select_dtypes(include=np.number).columns if c not in ['index','Churn']]
    if nums:
        f=st.selectbox('Select Feature',nums); st.plotly_chart(px.histogram(active,x=f,color='Churn' if 'Churn' in active else None,nbins=30,title=f'{f} Distribution'),use_container_width=True)
    c1,c2=st.columns(2)
    if {'Tenure_Months','Monthly_Charges'}<=set(active.columns): c1.plotly_chart(px.scatter(active,x='Tenure_Months',y='Monthly_Charges',color='Churn' if 'Churn' in active else None,title='Tenure vs Monthly Charges'),use_container_width=True)
    if {'NPS_Score','Churn'}<=set(active.columns): c2.plotly_chart(px.box(active,x='Churn',y='NPS_Score',title='NPS Score vs Churn'),use_container_width=True)
    c1,c2=st.columns(2)
    if {'Support_Tickets','Churn'}<=set(active.columns): c1.plotly_chart(px.box(active,x='Churn',y='Support_Tickets',title='Support Tickets vs Churn'),use_container_width=True)
    if {'Payment_Delays','Churn'}<=set(active.columns): c2.plotly_chart(px.box(active,x='Churn',y='Payment_Delays',title='Payment Delays vs Churn'),use_container_width=True)

elif page=='🧠 Explainability':
    st.title('SHAP Explainability'); st.caption('Global and per-customer explanations using the project XGBoost model.')
    try:
        import shap, matplotlib.pyplot as plt
        m=models['XGBoost']; x,_=split_xy(active); prep=m.named_steps['preprocessor']; est=m.named_steps['model']; tx=prep.transform(x); tx=tx.toarray() if hasattr(tx,'toarray') else tx; names=list(prep.get_feature_names_out()); explainer=shap.TreeExplainer(est); vals=np.asarray(explainer.shap_values(tx)); vals=vals[1] if vals.ndim==3 else vals
        imp=pd.DataFrame({'Feature':names,'Mean Absolute SHAP':np.abs(vals).mean(axis=0)}).sort_values('Mean Absolute SHAP',ascending=False).head(15)
        st.plotly_chart(px.bar(imp.sort_values('Mean Absolute SHAP'),x='Mean Absolute SHAP',y='Feature',orientation='h',title='Top Global Churn Drivers'),use_container_width=True); st.dataframe(imp,use_container_width=True)
        st.subheader('Individual Customer Explanation')
        ids=active.Customer_ID.astype(str).tolist() if 'Customer_ID' in active else [str(i) for i in range(len(active))]; chosen=st.selectbox('Select Customer',ids); pos=ids.index(chosen); p=float(m.predict_proba(x.iloc[[pos]])[:,1][0]); a,b,c=st.columns(3); a.metric('Customer',chosen); b.metric('Churn Probability',f'{p*100:.1f}%'); c.metric('Risk Tier',tier(p))
        st.markdown('### Top 3 Churn Reasons')
        for i in np.argsort(np.abs(vals[pos]))[::-1][:3]: st.write(f"• **{names[i].replace('num__','').replace('cat__','')}** {'increases' if vals[pos,i]>0 else 'reduces'} churn risk")
        expected=explainer.expected_value; expected=float(np.asarray(expected).reshape(-1)[-1]) if isinstance(expected,(list,np.ndarray)) else float(expected)
        exp=shap.Explanation(values=vals[pos],base_values=expected,data=tx[pos],feature_names=names); fig=plt.figure(figsize=(10,7)); shap.plots.waterfall(exp,max_display=12,show=False); st.pyplot(fig,clear_figure=True); plt.close(fig)
    except Exception as e: st.error('SHAP explanation could not be generated.'); st.exception(e)

elif page=='📊 Model Evaluation':
    st.title('Model Evaluation'); st.caption('Same 80/20 stratified split and random_state=42 as the training notebook.')
    if 'Churn' not in active: st.warning('Evaluation requires the Churn column.')
    elif st.button('▶ Run Model Evaluation',type='primary',use_container_width=True):
        try:
            x,y=split_xy(active); Xtr,Xte,ytr,yte=train_test_split(x,y,test_size=.20,random_state=42,stratify=y); rows=[]; preds={}
            for name,m in models.items():
                p=score(m,Xte); pr=(p>=.5).astype(int); preds[name]=pr; rows.append({'Model':name,'Accuracy':accuracy_score(yte,pr),'Precision':precision_score(yte,pr,zero_division=0),'Recall':recall_score(yte,pr,zero_division=0),'F1 Score':f1_score(yte,pr,zero_division=0),'ROC-AUC':roc_auc_score(yte,p),'PR-AUC':average_precision_score(yte,p)})
            res=pd.DataFrame(rows).sort_values('ROC-AUC',ascending=False); st.dataframe(res.style.format({c:'{:.3f}' for c in res.columns[1:]}),use_container_width=True); best=res.iloc[0].Model; st.success(f'Best model by ROC-AUC: {best}')
            long=res.melt(id_vars='Model',var_name='Metric',value_name='Score'); st.plotly_chart(px.bar(long,x='Model',y='Score',color='Metric',barmode='group',title='Model Performance Comparison'),use_container_width=True)
            cm=confusion_matrix(yte,preds[best]); st.subheader(f'Confusion Matrix — {best}'); st.dataframe(pd.DataFrame(cm,index=['Actual No Churn','Actual Churn'],columns=['Predicted No Churn','Predicted Churn']),use_container_width=True); st.download_button('⬇ Download Model Comparison',res.to_csv(index=False).encode(),'model_comparison.csv','text/csv',use_container_width=True)
        except Exception as e: st.error('Evaluation could not be completed.'); st.exception(e)

st.sidebar.divider(); st.sidebar.success('Models: trained from project notebook'); st.sidebar.caption('Machine Learning • Analytics • SHAP • Retention')
