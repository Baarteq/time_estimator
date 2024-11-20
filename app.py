import streamlit as st
import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI
from langfuse import Langfuse
import base64
from langfuse.decorators import observe
from langfuse.openai import OpenAI as LangfuseOpenAI
from pycaret.regression import load_model, predict_model
import datetime
import json
import os
import numpy as np


# Inicjacja Langfuse
langfuse = Langfuse()
langfuse.auth_check()

models = ['gpt-4o-mini']

env = dotenv_values(".env")

# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env['OPENAI_API_KEY']
    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])
openai_client = get_openai_client()

llm_client = LangfuseOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

dataset_name = "time_estimator"
langfuse.create_dataset(name=dataset_name)

@observe
def get_data_from_text_observed(user_input, model="gpt-4o-mini"):
    prompt = """
    Jesteś pomocnikiem w analizowaniu informacji przesłanych przez użytkownika.
    Zostaną Ci dostarczone treści wiadomości, które zawierają następujące dane:
    
    <imię> - imię użytkownika.
    <Rocznik> - Rok w którym urodził się użytkownik.
    <Płeć> - Płeć użytkownika- kobieta lub mężczyzna
    <Czas na 5km> - Czas w jakim użytkownik przebiega 5km
    <Czas na 15km> - Czas w jakim użytkownik przebiega 15km

    Zwróć wartość jako słownik następujących kluczy:
    - imię - jako ciąg znaków lub null, jeśli nie podano
    - Rocznik - jako liczba całkowita lub null, jeśli nie podano
    - Płeć - jako ciąg znaków lub null, jeśli nie podano
    - 5 km Czas - jako czas w sekundach lub null, jeśli nie podano
    - 15 km Czas - jako czas w sekundach lub null, jeśli nie podano

    Użytkownik poda dane w języku polskim, ale powinieneś odpowiedzieć za pomocą json, jak zdefiniowano powyżej
    i powinieneś użyć dokładnie tych kluczy. Zwróć prawidłowy słownik, nic więcej.
    Przeanalizuję wynik za pomocą funkcji json.loads() w pythonie, więc upewnij się, że wynik jest prawidłowy.

    Oto przykładowa treść wiadomości:
    ```
    Mam na imię Bartek, urodziłem się w 1987 roku, jestem mężczyzną, 5km przebiegam w 20 minut, a 15km w 45 minut.
    ```

    W tym przypadku słownik powinien wyglądać następująco:
    {
    "imię": "Bartek",
    "Rocznik": "1987",
    "Płeć": "mężczyzna",
    „5 km Czas”: 1200,
    „15 km Czas”: 2700
    }
    """
    messages=[
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"```{user_input}```",
        },
    ]

    chat_completion = llm_client.chat.completions.create(
        response_format={"type": "json_object"},
        messages=messages, 
        model=model,
        # dodatkowe
        name="get_data_from_text_observed",
    )
    resp = chat_completion.choices[0].message.content
    try:
        output = json.loads(resp)
    except:
        output = {"error": resp}
    return output

MODEL_NAME = 'regression_pipeline'

@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)

#MAIN

st.set_page_config(page_title="Estymator czasu na półmaratonie", layout="wide")
st.title("Estymator czasu na półmaratonie")
st.markdown("Witaj w aplikacji, w której możesz estymować swój czas w półmaratonie na podstawie kilku informacji")
st.markdown(" podając swoje imię, podaj Twój rok urodzenia, płeć, Twój średni czas na 5km i 15km")
if 'user_input' not in st.session_state:
    st.session_state['user_input']=''
st.session_state['user_input']= st.text_area('Tutaj wpisz swoje dane', value=st.session_state['user_input'])

if 'dobre_dane' not in st.session_state:
    st.session_state['dobre_dane']=''

if st.button("Weryfikuj dane"):
    with st.spinner("Analizuję dane. Czekaj...."):
        data= get_data_from_text_observed(st.session_state['user_input'])
        index_values = ['0']
        if "df" not in st.session_state:
            st.session_state['df']=''
        st.session_state['df']=pd.DataFrame(data, index_values)
        if st.session_state['df']['imię'].isnull().any():
            st.error('Nie podałeś swojego imienia')
        if st.session_state['df']['Rocznik'].isnull().any():
            st.error('Nie podałeś swojego roku urodzenia')
        if st.session_state['df']['Płeć'].isnull().any():
            st.error('Nie podałeś swojej płci')
        if st.session_state['df']['5 km Czas'].isnull().any():
            st.error('Nie podałeś swojego czasu na 5 km')
        if st.session_state['df']['15 km Czas'].isnull().any():
            st.error('Nie podałeś swojego czasu na 15 km')
        else: 
            st.session_state['dobre_dane']=st.session_state['df'].isnull().sum().sum() == 0
            if (st.session_state['dobre_dane']):
                st.success("Dziękuję za podanie niezbędnych danych")
if st.session_state['dobre_dane']:
    if st.button("Estymuj czas na maratonie"):
        model = get_model()
        prediction = predict_model(model, st.session_state["df"])
        prediction_seconds = round(prediction["prediction_label"][0], 2)
        prediction_time = str(datetime.timedelta(seconds=int(prediction_seconds)))
        estimate_result = st.success(f'Estymowany czas ukończenia półmaratonu to: {prediction_time}')
if st.button('Wyczyść dane'):
    st.session_state['user_input']= ""
    st.session_state['user_data'] = ""
    st.session_state['dobre_dane']= ""
    st.session_state['df'] = ""
    st.rerun()