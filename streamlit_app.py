import streamlit as st
import pandas as pd
import plotly.express as px
import sklearn.ensemble import RandomForestClassfier


st.title('🎈 Я что то делаю')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

with st.expander('Data'):
  st.write('X')   
  X_raw = df.drop('species', axis=1)
  st.dataframe(X_raw)
  
  st.write('y')
  y_raw = df.species
  st.dataframe(y_raw)


with st.sidebar:
  st.header("Input same:")
  island = st.selectbox('Island', ('Torgerson', 'Dream', 'Biscose'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 44.5)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.3)
  flipper_length_mm = st.slider('Flipper length (mm)', 32.1, 59.6, 44.5)
  body_mass_g = st.slider('Body Mass (g)', 32.1, 59.6, 44.5)
  gender = st.selectbox('Gender', ('female', 'male'))
st.subheader('Data Visualization')
fig = px.scatter(
    df,
    x='bill_length_mm',
    y='bill_depth_mm',
    color='island',
    title='Bill Length vs. Bill Depth by Island'
)
st.plotly_chart(fig)

fig2 = px.histogram(
    df, 
    x='body_mass_g', 
    nbins=30, 
    title='Distribution of Body Mass'
)
st.plotly_chart(fig2)

data = {
    'island': island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex': gender
}

input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Inp Feat'):
  st.write('**Input pengunis**')
  st.dataframe(input_df)
  st.write('com')
  st.dataframe(input_penguins)
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
input_row = df_penguins[:1]

target_mapper= {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}

def tar_enc(val):
  return target_mapper[val] 
y = y_raw.apply(tar_enc)

with st.expander('Data prerp'):
  st.write('**Encded y x (Input pengunis)**')
  st.dataframe(input_row)
  st.write('**Encded y**')
  st.dataframe(y)  


base_rf = RandomForestClassfier(random_state=42)
base_rf.fit(X, y)
prediction = base_rf.predict(input_row)
predict_proba = base_rf.predicr_prba(input_row)
df_pred_proba = pd.DataFrame(predict_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])
st.subheader('Predicted Species')
st.dataframe(
    df_prediction_proba,
    column_config={
        'Adelie': st.column_config.ProgressColumn(
            'Adelie',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
        'Chinstrap': st.column_config.ProgressColumn(
            'Chinstrap',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
        'Gentoo': st.column_config.ProgressColumn(
            'Gentoo',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
    },
    hide_index=True
)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f"Predicted species: **{penguins_species[prediction][0]}**")














