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
url = "https://firebasestorage.googleapis.com/v0/b/studyszn.appspot.com/o/H2_physics_vectorised.csv?alt=media"
df = pd.read_csv(url)
df['Embeddings'] = df['Embeddings'].apply(lambda x: np.array(eval(x)), 0)


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


"---"
st.header("Select a question")


if "similarQs" not in st.session_state:
    st.session_state.similarQs =0

if "current_question" not in st.session_state:
    st.session_state.current_question = -1

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


with st.form("question_choice", clear_on_submit=False):
    left_column, right_column = st.columns(2)


    with left_column:   
        chosen_question = st.number_input(label="Choose a question from 1 to 681",min_value=1,max_value=681) -1
    if st.form_submit_button("View Question"):
        try:
            chosen_question = int(chosen_question)
            st.session_state.current_question = chosen_question
            st.session_state.similarQs = 1
        except ValueError:
            st.error("Invalid input, please key a valid integer")

    

    with right_column:
        if st.session_state.current_question < 0:
            st.info("Current question: Not Selected Yet")
        
        else:
            st.info(f"Current Selected Question: {chosen_question}  \nTopic: {get_topic(chosen_question)}", icon="📔")

    

    if st.session_state.similarQs > 0:
        question_expander = st.expander("Click to view question")
        with question_expander:
            question_output = html_with_css + parseQuestion(df,chosen_question)
            st.markdown(question_output, unsafe_allow_html=True)
            st.session_state.similarQs +=1

        submitAnswer = st.expander("Click to view Answers")
        with submitAnswer:
             output = html_with_css + parseAnswer(df, st.session_state.current_question)
             st.markdown(output, unsafe_allow_html=True)

if st.session_state.similarQs>0:
    if st.button("View Similar Questions"):
        st.header("Similar Questions")
        similar_question_indices = find_most_similar_questions(st.session_state.current_question, df, top_n=5)
        tab_labels = [f"Question {i+1}" for i in range(len(similar_question_indices))]
        tab1,tab2,tab3,tab4,tab5 = st.tabs(tab_labels)
        tabs = [tab1, tab2, tab3, tab4, tab5]

        for i,tab in enumerate(tabs):
            with tab:
                question_expander = st.expander("Click to view question")
                with question_expander:
                    st.markdown(parseQuestion(df, similar_question_indices[i]))

                submitAnswer = st.expander("Click to view Answers")
                with submitAnswer:
                    html_with_css = parseAnswer(df, similar_question_indices[i])
                    st.markdown(html_with_css, unsafe_allow_html=True)

