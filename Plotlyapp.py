import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
df = pd.read_csv('updatedata.csv')
st.title('THE CENSUS :blue[ANALYSIS] :sunglasses:')

df['Sex Ratio'] = round((df['Male']/df['Female'])*100)
df['Literacy Rate'] = round((df['Literate']/df['Population'])*100)

state_list = list(df['State name'].unique())
state_list.insert(0,'OVER ALL INDIA')

st.sidebar.title('India Census 2011 Analysis')
option = st.sidebar.selectbox('Select A State',state_list,index=None)
st.sidebar.write('Your Selected :',option)


primary = st.sidebar.selectbox('Select The First Parameter',sorted(df.columns[5:]),index=None)
st.sidebar.write('First Parameter',primary)
secondary = st.sidebar.selectbox('Select The Second Parameter',sorted(df.columns[5:]),index=None)
st.sidebar.write('Second Parameter',secondary)

plot_graph = st.sidebar.button('PLOT GRAPH')
analysis = st.sidebar.button('ANALYSIS')



def popplot(list):
    df2 = df.groupby('State name')[list].sum()
    fig = px.scatter_3d(df2,x='Population',y='Sex Ratio',z='Literacy Rate',color=df2.index,size_max=40,title='pop vs sex ratio vs litracy rate')
    return st.plotly_chart(fig,use_container_width=True)


def popdf(list):
    df1 = df.groupby('State name')[list].sum().sort_values(by='Population',ascending=False).head(5)
    return st.dataframe(df1)

def smat(list):
    df2 = df.groupby('State name')[list].sum()
    fig = px.scatter_matrix(df2,dimensions=list,color=df2.index)
    return st.plotly_chart(fig,use_container_width=True)

def psdf(list):
    df1 = df.groupby('State name')[list].sum().sort_values(by=list,ascending=[False,False]).head(5)
    return st.dataframe(df1)
def spsdf():
    return st.dataframe(c_df) 
def ggiven(list):
    fig = px.scatter_matrix(c_df,dimensions=list,color='District',title='First parameter and second parameter graph')
    return st.plotly_chart(fig,use_container_width=True)

def lgraph(list,lf):
    fig = px.line(x=lf[list[0]],y=lf[list[1]])
    return st.plotly_chart(fig,use_container_width=True)

def hg(list,lf):
    fig = px.scatter(lf,x=lf[list[0]],y=lf[list[1]],size='Population',color='District',size_max=30,hover_name='District',title='Graph Respect to Population')
    return st.plotly_chart(fig,use_container_width=True)

def pieg(values0,label):
    fig = px.pie(c_df,values = values0,names = label,hover_name = 'District',hover_data=primary,title='FIRST AND SECOND BASIS OF FIRST')
    return st.plotly_chart(fig,use_container_width=True)
def pieg1(values1,label):
    fig = px.pie(c_df,values = values1,names = label,hover_name = 'District',hover_data=secondary,title='FIRST AND SECOND BASIS OF SECOND')
    return st.plotly_chart(fig,use_container_width=True)

if plot_graph:
    if (option == 'OVER ALL INDIA'):
        fig = px.scatter_mapbox(df,lat='Latitude',lon='Longitude',size=primary,color=secondary,color_continuous_scale='viridis',
                             zoom=3,size_max=20,mapbox_style='carto-positron',hover_name='District',
                             width=1000,height=600)
        st.plotly_chart(fig,use_container_width=True)
 
    else :
        state_df = df[df['State name']==option]
        st.header(option,divider=True)
        fig = px.scatter_mapbox(state_df,lat='Latitude',lon='Longitude',size=primary,color=secondary,color_continuous_scale='viridis',
                            zoom=5,size_max=20,mapbox_style='carto-positron',hover_name='District',
                            width=1000,height=600)
        st.plotly_chart(fig,use_container_width=True)

if analysis:
    if(option == 'OVER ALL INDIA'):
        st.header('OVER ALL ANALYSIS',divider=True)
        st.subheader('Five :blue[MAXIMUM]: Population State')
        l = ['Population','Literate','Sex Ratio','Literacy Rate']
        obj1=popdf(l)
        k=[primary,secondary]
        m = ['Population','Sex Ratio','Literacy Rate']
        obj2= popplot(m)
        obj3=smat(k)
        obj4 = psdf(k)
        
    else:
        s_name = option
        s_df = df[df['State name'] == s_name]
        s_df = s_df.iloc[:,1:]
        s_df.index = np.arange(1, len(s_df) + 1)
        c_df = s_df.copy()
        c_df.drop(columns = ['State name','Latitude','Longitude'],inplace = True)
        values0 = c_df[primary]
        values1 = c_df[secondary]
        label = c_df['District']
            
        st.header(option,divider=True)
        k=[primary,secondary]
        ob2=spsdf()
        ob1=ggiven(k)
        ob_4=hg(k,c_df)
        ob_5 = pieg(values0,label)
        ob_6=pieg1(values1,label)
        

