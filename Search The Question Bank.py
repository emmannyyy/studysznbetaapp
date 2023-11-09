import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
from tqdm import tqdm
import re
from IPython.display import display, Markdown
from PIL import Image
import os
import numpy as np

#---------Templates for CSS ---------

html_with_css = """
<style>
    body {
        font-family: Arial, sans-serif;
    }
    p {
        line-height: 1.5;
    }
</style>
"""             

#----------Settings-----------
page_title = "H2 Physics Question Bank"
page_icon = ":books:"
layout = "centered"
#----------------------------

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

#---- Drop Down Values for Selecting the period -----

@st.cache_data
def load_data(url): 
    df = pd.read_csv(url)
    return df

url = "https://firebasestorage.googleapis.com/v0/b/studyszn.appspot.com/o/H2_physics_vectorised.csv?alt=media"

df = load_data(url)

@st.cache_data
def apply_transformation(data):
    data['Embeddings'] = data['Embeddings'].apply(lambda x: np.array(eval(x)), 0) 
    return data

df = apply_transformation(df)

def display_open_ended_question(content, base_url="https://firebasestorage.googleapis.com/v0/b/studyszn.appspot.com/o/"):
    
    # Use a regular expression to find all placeholders like [image1], [image2], etc.
    placeholders = re.findall(r'\[image\d+\]', content)
    
    # Dynamically replace placeholders with image tags
    for placeholder in placeholders:
        # Extract the image number from the placeholder
        image_number = int(re.search(r'\d+', placeholder).group())
        # Create the image tag with the specified URL
        image_tag = f'![image]({base_url}image{image_number}.png?alt=media)'
        # Replace the placeholder with the image tag
        content = content.replace(placeholder, image_tag)
    # Create an HTML object and display it
    return content


def isOpenEndedQn(df, questionNo):
    return (pd.isnull(df.iloc[questionNo]['Option A Image']) and pd.isnull(df.iloc[questionNo]['Option A']))

def isImageMCQ(df, questionNo):
    return pd.notnull(df.iloc[questionNo]['Option A Image']) 

def isMCQWithoutOptions(df,questionNo):
    search_text = "<blockquote>\n<p>&nbsp;</p>\n</blockquote>\n"
    option_a_text = str(df.iloc[questionNo]['Option A'])
    return search_text in option_a_text

def Image_MCQ_replace_image_tags(row):
    text = row['Question']
    
    for Letter in ["A","B","C","D"]:
        text = text + "\n" + "[" + row[f'Option {Letter} Image'] + "] "
        
    for i in range(1, 7):
        try:
            if(pd.isnull(row[f'Qimage {i}'])):
                continue
            tag = f'[image{i}]'
            replacement = "[" + row[f'Qimage {i}'] + "] " #need a space right after!
            text = text.replace(tag, replacement)
    
        except KeyError:
            pass  # Handle the case where the Qimage column doesn't exist
    return text

def MCQ_text_Parser(row,text):
    pattern = r'<p>(.*?)<\/p>'
    
    for Letter in ["A","B","C","D"]:
        
        option = row[f'Option {Letter}']
        matches = re.findall(pattern, option, re.DOTALL)
        extracted_content = [match.strip() for match in matches]
        
        text = text + "\n" + "\n" + Letter + "\n" + ":     " + extracted_content[0] + "\n"
    return text


def replace_image_tags(row):
    text = row['Question']
    for i in range(1, 7):
        try:
            if(pd.isnull(row[f'Qimage {i}'])):
                continue
            tag = f'[image{i}]'
            replacement = "[" + row[f'Qimage {i}'] + "] " #need a space right after!
            text = text.replace(tag, replacement)
        except KeyError:
            pass  # Handle the case where the Qimage column doesn't exist
    return text

def parseQuestion(df, questionNum):
    if (isOpenEndedQn(df,questionNum)):
        content = replace_image_tags(df.iloc[questionNum])
        return (display_open_ended_question(content))
    else:
        row = df.iloc[questionNum]
        #is an mcq question
            
            
        if (isImageMCQ(df,questionNum)):
            return (display_open_ended_question(Image_MCQ_replace_image_tags(row)))

        elif (isMCQWithoutOptions(df,questionNum)):
            return display_open_ended_question(replace_image_tags(df.iloc[questionNum]))

        else:
            content = replace_image_tags(df.iloc[questionNum])
            return (display_open_ended_question(MCQ_text_Parser(df.iloc[questionNum],content)))        


def replace_image_tags_answers(row):
    text = row['Answer Open']
    #text = text.replace("<p>", "").replace("</p>", "")
    for i in range(1, 7):
        try:
            if(pd.isnull(row[f'Answer Image{i}'])):
                continue
            tag = f'[image{i}]'
            replacement = "[" + row[f'Answer Image{i}'] + "] " #need a space right after!
            text = text.replace(tag, replacement)
        except KeyError:
            pass  # Handle the case where the Qimage column doesn't exist
    return text

def display_open_ended_answers(content, base_url="https://firebasestorage.googleapis.com/v0/b/studyszn.appspot.com/o/"):
    # Use a regular expression to find all placeholders like [image1], [image2], etc.
    placeholders = re.findall(r'\[image\d+\]', content)

    # Dynamically replace placeholders with HTML <img> tags
    for placeholder in placeholders:
        # Extract the image number from the placeholder
        image_number = int(re.search(r'\d+', placeholder).group())
        # Create the HTML <img> tag with the specified URL
        img_tag = f'<img src="{base_url}image{image_number}.png?alt=media" alt="image">'
        # Replace the placeholder with the HTML <img> tag
        content = content.replace(placeholder, img_tag)
    # Create an HTML object and display it
    return content

def parseAnswer(df, questionNum):
    if (isOpenEndedQn(df,questionNum)):
        content = replace_image_tags_answers(df.iloc[questionNum])
        return (display_open_ended_answers(content))
    elif not (isOpenEndedQn(df,questionNum)):
            row = df.iloc[questionNum]
            ans = "The answer to this MCQ question is: " + str(df.iloc[questionNum]['Answer Option'])
            return (ans)
    else:
            raise Exception("Error!")

def find_most_similar_question(questionnum, dataframe):
    tempdf = df.drop(df.iloc[questionnum].name)
    dot_products = np.dot(np.stack(tempdf['Embeddings']), dataframe.iloc[questionnum]['Embeddings'])
    idx = np.argmax(dot_products) +1
    return idx 


def find_most_similar_questions(questionnum, dataframe, top_n=5):
    # Create a DataFrame without the target question
    tempdf = dataframe.drop(dataframe.index[questionnum])

    # Calculate dot products with the target question's embedding
    dot_products = np.dot(np.stack(tempdf['Embeddings']), dataframe.iloc[questionnum]['Embeddings'])

    # Get the indices of the top N most similar questions
    top_indices = np.argpartition(dot_products, -top_n)[-top_n:]

    # Sort the top indices by dot product values in descending order
    top_indices = top_indices[np.argsort(-dot_products[top_indices])]

    # Return the top N most similar question indices (add 1 to match question numbering)
    most_similar_indices = top_indices + 1

    return list(most_similar_indices)


def get_topic(questionnum, dataframe=df):
    return dataframe.iloc[questionnum]['topic']

def get_topics(questionnum, dataframe=df):
    return dataframe.iloc[questionnum]['related_topics'].strip('[]').replace("'", "")

def find_indices_containing_string(search_string, dataframe=df):
    # Use the str.contains() method to check for the presence of the search_string in the 'Question' column
    mask = dataframe['Question'].str.contains(search_string, case=False)
    # Get the indices where the mask is True
    indices = mask[mask].index.tolist()
    return indices

"---"
st.header("Search for a question that contains certain keywords")


if "similarQs" not in st.session_state:
    st.session_state.similarQs =0

if "current_question" not in st.session_state:
    st.session_state.current_question = -1

if "similar_question_indices" not in st.session_state:
    st.session_state.similar_question_indices =[]

if "search_list" not in st.session_state:
    st.session_state.search_list =[]

if "search_list_mcq" not in st.session_state:
    st.session_state.search_list_mcq =[]

if "search_list_oeq" not in st.session_state:
    st.session_state.search_list_oeq =[]

st.markdown(
    """
    <style>
    .stButton button {
        width: 100% !important; /* Make the button take up the entire column width */
        background-color: green; /* Green background */
        color: white; /* White text color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.form("Type keywords that you wish to search for:", clear_on_submit=False):
        # Add an input field for the user to enter a question
        question = st.text_input("Searching for question that contains keywords...")

        # Add a submit button
        if st.form_submit_button("Submit"):
            try: 
                questionstring = str(question)
                #update state
                st.session_state.similarQs = 1
                st.session_state.search_list = find_indices_containing_string(questionstring)
            except ValueError:
                st.error("Invalid input, please key in a valid string")

key = 0

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
def callback():
    #button was clicked!
    st.session_state.button_clicked = True

if len(st.session_state.search_list) >0:
    for i in st.session_state.search_list:
        if not (isOpenEndedQn(df,i)):
            st.session_state.search_list_mcq.append(i)
        else:
            st.session_state.search_list_oeq.append(i)
    st.header("MCQ Questions")
    for i in st.session_state.search_list_mcq:
        val = int(key)
        key +=10
        boxexpander = st.expander(f"Question{i+1}")
        with boxexpander:
            question_output = html_with_css + parseQuestion(df, i) 
            st.markdown(question_output,unsafe_allow_html=True)
            if st.checkbox(f"Show Answer for Question {i+1}", key=val+1):
                answer_output = html_with_css + parseAnswer(df, i)
                st.markdown(answer_output, unsafe_allow_html=True)
            viewsimilar = st.button("View Similar Questions", key = val+2, on_click=callback)
            if viewsimilar or st.session_state.button_clicked:
                similar_question_indices = find_most_similar_questions(i, df, top_n=5)
                tab_labels = [f"Question {i+1}" for i in similar_question_indices]

                for j,tab in enumerate(st.tabs(tab_labels)):
                    with tab:
                        question_output = html_with_css + parseQuestion(df,similar_question_indices[j])
                        st.markdown(question_output,unsafe_allow_html=True)

                        if st.checkbox(f"Show answer for Question {similar_question_indices[j]+1}", key=val+3+j):
                            answer_output = html_with_css + parseAnswer(df, similar_question_indices[j])
                            st.markdown(answer_output, unsafe_allow_html=True)

                
    st.divider()
    st.header("Open Ended Questions")
    for i in st.session_state.search_list_oeq:
        val = int(key)
        key +=10
        boxexpander = st.expander(f"Question{i+1}")
        with boxexpander:
            question_output = html_with_css + parseQuestion(df, i) 
            st.markdown(question_output,unsafe_allow_html=True)
            if st.checkbox(f"Show Answer for Question {i+1}", key =key+1):
                key+=1
                answer_output = html_with_css + parseAnswer(df, i)
                st.markdown(answer_output, unsafe_allow_html=True)

            viewsimilar = st.button("View Similar Questions", key = val+2, on_click=callback)
            if viewsimilar or st.session_state.button_clicked:
                similar_question_indices = find_most_similar_questions(i, df, top_n=5)
                tab_labels = [f"Question {i+1}" for i in similar_question_indices]

                for j,tab in enumerate(st.tabs(tab_labels)):
                    with tab:
                        question_output = html_with_css + parseQuestion(df,similar_question_indices[j])
                        st.markdown(question_output,unsafe_allow_html=True)

                        if st.checkbox(f"Show answer for Question {similar_question_indices[j]+1}", key=val+3+j):
                            answer_output = html_with_css + parseAnswer(df, similar_question_indices[j])
                            st.markdown(answer_output, unsafe_allow_html=True)

    #rest all indices for next search
    st.session_state.search_list_mcq =[]
    st.session_state.search_list_oeq =[]
