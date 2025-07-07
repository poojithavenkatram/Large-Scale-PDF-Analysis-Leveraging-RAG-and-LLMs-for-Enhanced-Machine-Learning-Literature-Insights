import os
import streamlit as st
import cv2
import tempfile
import base64
import uuid
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain.schema.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import Any

# Set the Groq API key
os.environ["GROQ_API_KEY"] = "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"
class Element(BaseModel):
    type: str
    text: Any

class ImageAnalyzer:
    def __init__(self, image_path) -> None:
        self.image_path = image_path

    def analyze_image(self):
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text_data = pytesseract.image_to_string(gray)
        edges = cv2.Canny(gray, 100, 200)
        summary = f"Extracted Text: {text_data}\nDetected Edges: {np.count_nonzero(edges)} edges identified."
        return summary

class ImageSummarizer:
    def __init__(self, image_path) -> None:
        self.image_path = image_path
        self.chat = ChatGroq(model="Llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

    def summarize(self, image_analysis_summary):
        prompt = f"""
        You are provided with details extracted from an image or a graph.

        Details: {image_analysis_summary}

        Your task is to thoroughly analyze and describe the image or graph. Specifically:
        1. Identify and describe **key visual elements** in the image or graph, such as shapes, patterns, colors, labels, and any symbols.
        2. For graphs, capture:
          - Axis labels, scales, units, and numerical markers.
          - Trends, peaks, troughs, and outliers in the data.
          - Legends, annotations, or labels that provide context.
          - Any symmetry, correlations, or relationships between variables.
        3. For images (photos, charts, or other visuals), describe:
          - Key objects, their positions, and relationships.
          - Text content, if present, and any embedded annotations.
          - Distinct patterns, shapes, or visual themes.
        4. Summarize the **purpose** or **function** of the image or graph, if apparent.
        5. Extract **structured information** where applicable, ensuring all observations are precise, clear, and easy to interpret.

        Provide your response in a structured format suitable for downstream scientific, business, or general-purpose analysis. Aim for maximum accuracy and comprehensiveness.
      """
        response = self.chat.invoke([HumanMessage(content=prompt)])
        return response.content

# Text and table summarization setup
prompt_text = """
  You are responsible for capturing all values, trends, and data points from the table or text chunk:

  {element}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)
summarize_chain = (
    {"element": lambda x: x}
    | prompt
    | ChatGroq(model="Llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
    | StrOutputParser()
)

# MultiVectorRetriever setup for storing all summaries
id_key = "doc_id"
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=HuggingFaceEmbeddings(),
    persist_directory="chromadb_data"
)
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=InMemoryStore(), id_key=id_key)
def set_background_color():
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: black;
            color: white;
        }
        input, textarea {
            background-color: #333;
            color: white;
        }
        button {
            color: white !important;
            background-color: orange !important;
            border: none;
            border-radius: 5px;
        }
        button:hover {
            background-color: darkorange !important;
        }
        .stSidebar {
            background-color: #222;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def extract_pdf_content(pdf_file_path, output_dir):
    return partition_pdf(
        filename=pdf_file_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=output_dir
    )

def img_prompt_func(data_dict):
    messages = []
    for image in data_dict["context"]["images"]:
        image_message = f"Image Summary: {image}"
        messages.append(HumanMessage(content=image_message))

    formatted_texts = "\n".join(data_dict["context"]["texts"])
    text_message = f"Context: {formatted_texts}\nQuestion: {data_dict['question']}"
    messages.append(HumanMessage(content=text_message))
    return messages

def main():
   
    st.set_page_config(
        page_title="📄 Multimodal Rag",
        page_icon="📊",
        layout="wide",
    )

    st.title("📄 Multimodal RAG")
    st.subheader(
        ""
    )
    st.markdown(
        """
        
        """
    )
    st.markdown("---")

    st.sidebar.title("")
    st.sidebar.info(
        """
        
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Developed with ❤️ by Team-8")
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []  # Initialize conversation history
    set_background_color()
    # Display conversation history
    ### Added Section
    st.markdown("## 💬")
    if st.session_state["conversation"]:
        for entry in st.session_state["conversation"]:
            role, message = entry
            if role == "User":
                st.markdown(f"**You:** {message}")
            elif role == "Bot":
                st.markdown(f"**Bot:** {message}")
    if st.button("Clear Chat"):
        st.session_state["conversation"] = []  # Reset the conversation
        st.success("Chat history cleared!")
    pdf_file = st.file_uploader("Drag and drop or browse to upload a PDF file:", type="pdf")

    if pdf_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name

            image_output_dir = "/content/figures"
            os.makedirs(image_output_dir, exist_ok=True)

            st.info("🔄 Extracting content from the PDF...")
            elements = extract_pdf_content(tmp_file_path, image_output_dir)
            st.success("✅ Content extraction completed! Summaries are displayed below.")

            # Summarize tables and texts
            st.markdown(" ")
            table_elements, text_elements = [], []
            for element in elements:
                if "unstructured.documents.elements.Table" in str(type(element)):
                    table_elements.append(Element(type="table", text=str(element)))
                elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                    text_elements.append(Element(type="text", text=str(element)))

            tables = [i.text for i in table_elements]
            texts = [i.text for i in text_elements]
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

            #for original, summary in zip(tables, table_summaries):
                #with st.expander("Original Table Content"):
                 #   try:
                 #       table_df = pd.read_json(original)
                 #       st.table(table_df)
                 #   except ValueError:
                 #       st.write(original)
                #st.markdown("**Summary:**")
                #st.write(summary)

            #st.markdown("### 📝 Text Summaries")
            #for original, summary in zip(texts, text_summaries):
                #with st.expander("Original Text Content"):
                #    st.write(original)
               # st.markdown("**Summary:**")
               # st.write(summary)

            # Process images
            #st.markdown("### 🖼️ Image Summaries")
            cols = st.columns(2)
            image_summary_list = []
            for idx, figure_file in enumerate(sorted(os.listdir(image_output_dir))):
                if figure_file.endswith((".png", ".svg", ".jpg")):
                    image_path = os.path.join(image_output_dir, figure_file)
                    analyzer = ImageAnalyzer(image_path)
                    analysis_summary = analyzer.analyze_image()

                    summarizer = ImageSummarizer(image_path)
                    image_summary = summarizer.summarize(analysis_summary)
                    image_summary_list.append(image_summary)
            

                    #with cols[idx % 2]:
                        #img = Image.open(image_path)
                       # st.image(img, caption=f"Extracted Image: {figure_file}", use_container_width=True)
                        #st.markdown(f"**Summary:** {image_summary}")
            # Add image summaries to the RAG system
            image_ids = [str(uuid.uuid4()) for _ in image_summary_list]
            image_documents = [
                Document(page_content=summary, metadata={id_key: image_ids[i]})
                for i, summary in enumerate(image_summary_list)
            ]
            retriever.vectorstore.add_documents(image_documents)
            retriever.docstore.mset(list(zip(image_ids, image_summary_list)))

            text_ids = [str(uuid.uuid4()) for _ in texts]
            summary_texts = [Document(page_content=s, metadata={id_key: text_ids[i]}) for i, s in enumerate(text_summaries)]
            retriever.vectorstore.add_documents(summary_texts)
            retriever.docstore.mset(list(zip(text_ids, texts)))

            st.markdown("## 🤔 Ask a Question About the PDF")
            question = st.text_input("Type your question here:")
            if question:
                st.session_state["conversation"].append(("User", question)) 
                model = ChatGroq(model="Llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
                context = img_prompt_func({
                    "context": {
                        "images": image_summary_list,
                        "texts": text_summaries
                    },
                    "question": question
                })
                with st.spinner("Generating answer..."):
                    try:
                        answer = model.invoke(context)
                        clean_answer = answer.content.strip()
                        st.markdown("**Answer:**")
                        # Displaying only the content of the response
                        st.write(answer.content.strip())
                        st.session_state["conversation"].append(("Bot", clean_answer))
                    except Exception as e:
                        error_message = f"Error in question-answering chain: {e}"
                        st.error(error_message)
                        st.session_state["conversation"].append(("Bot", error_message))


        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
