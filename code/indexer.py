import os
from langchain_openai import AzureOpenAIEmbeddings
import warnings
from dotenv import load_dotenv
import glob

load_dotenv()

warnings.filterwarnings("ignore")

CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")

# If getting upgrade warning, run the following from command line:
# chroma utils vacuum --path chroma_db
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    print(f"Created {len(chunks)} chunks")
    return chunks 


def create_embeddings_chroma(chunks, persist_directory=CHROMA_PATH):
    from langchain_community.vectorstores import Chroma
 
    # Instantiate an embedding model from Azure OpenAI
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    )

    # Create a Chroma vector store using the provided text chunks and embedding model, 
    # configuring it to save data to the specified directory 
    Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory) 


#### MAIN PROCESSING STARTS HERE

data_dir = os.path.join(os.getcwd(), 'data')
pdf_files = glob.glob(os.path.join(data_dir, '*.pdf'))

# Load all PDF documents
documents = []
for pdf_file in pdf_files:
    try:
        doc = load_document(pdf_file)
        if doc:  # Check if document was loaded successfully
            documents.extend(doc)  # Use extend() instead of append()
            print(f"Loaded: {os.path.basename(pdf_file)} with {len(doc)} pages")
    except Exception as e:
        print(f"Error loading {pdf_file}: {e}")

print(f'Loaded all resumes... Total documents: {len(documents)}')

# Split the document into chunks
print('Splitting documents into chunks')
chunks = chunk_data(documents, chunk_size=256)

# Create embeddings and store in Chroma
print('Creating a Chroma vector store using the provided text chunks and embedding model')
create_embeddings_chroma(chunks)

print('Chroma vector db created')