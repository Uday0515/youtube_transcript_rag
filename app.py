"""
YouTube Transcript RAG

Streamlit app that lets a user paste a YouTube URL, transcribes the audio
locally with Whisper, embeds the transcript into ChromaDB, and answers
questions about the video using Mistral with clickable timestamp citations.
"""

import os
import re
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
WHISPER_MODEL_SIZE = "base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MISTRAL_MODEL = "mistral-small-latest"

YOUTUBE_URL_PATTERNS = [
    r"[?&]v=([a-zA-Z0-9_-]{11})",
    r"youtu\.be/([a-zA-Z0-9_-]{11})",
    r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    r"youtube\.com/v/([a-zA-Z0-9_-]{11})",
    r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
]


def extract_video_id(url: str) -> str | None:
    """Pull the 11-character YouTube video ID out of any supported URL shape."""
    for pattern in YOUTUBE_URL_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def collection_name_for(video_id: str) -> str:
    """
    Turn a video ID into a name ChromaDB will accept.

    ChromaDB collection names must be 3-63 characters, use only
    [a-zA-Z0-9_-], and cannot start or end with '-' or '_'. Wrapping the
    ID with letters on both sides guarantees those boundary rules are met
    no matter what the raw video ID looks like.
    """
    return f"yt_{video_id}_yt"[:63]


def format_timestamp(seconds: float) -> str:
    """Render a second offset as M:SS, or H:MM:SS once past the hour mark."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def youtube_timestamp_link(video_id: str, seconds: float) -> str:
    """Build a YouTube link that seeks straight to the given second offset."""
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load the local Whisper model used for transcription."""
    import whisper
    return whisper.load_model(WHISPER_MODEL_SIZE)


@st.cache_resource(show_spinner=False)
def load_embedding_function():
    """Load the sentence-transformers model ChromaDB will use for embeddings."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """Connect to the on-disk, persistent ChromaDB store."""
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


@st.cache_resource(show_spinner=False)
def get_mistral_client():
    """Build the Mistral chat client, or None if no API key is configured."""
    from langchain_mistralai import ChatMistralAI

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return None
    return ChatMistralAI(model=MISTRAL_MODEL, api_key=api_key)


def collection_already_exists(client, name: str) -> bool:
    """Check whether a video has already been processed and stored before."""
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def download_audio(url: str, destination_dir: str) -> str:
    """Download the best available audio track for a video as an MP3."""
    import yt_dlp

    output_template = os.path.join(destination_dir, "audio")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([url])
    return f"{output_template}.mp3"


def transcribe_audio(audio_path: str) -> list[dict]:
    """Run Whisper on a local audio file and return its timed segments."""
    model = load_whisper_model()
    result = model.transcribe(audio_path, verbose=False)
    return result["segments"]


def chunk_transcript(segments: list[dict]) -> list[dict]:
    """
    Turn Whisper's segments into overlapping, fixed-size word chunks.

    Whisper segments are sentence-ish and irregular in length, which makes
    poor, inconsistent retrieval units. Instead we flatten every segment
    down to individual words (each tagged with its segment's start time),
    then slide a fixed-size window over that word stream. The overlap
    between windows means an idea sitting right on a chunk boundary still
    appears in full inside at least one chunk, instead of being split and
    losing context on both sides.
    """
    words_with_timestamps = [
        {"word": word, "start": segment["start"]}
        for segment in segments
        for word in segment["text"].strip().split()
    ]

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start_idx in range(0, len(words_with_timestamps), step):
        window = words_with_timestamps[start_idx : start_idx + CHUNK_SIZE]
        if not window:
            break
        chunks.append({
            "text": " ".join(entry["word"] for entry in window),
            "start": window[0]["start"],
            "index": len(chunks),
        })

    return chunks


def ingest_video(url: str, video_id: str, status) -> object:
    """Run the full pipeline for a new video: download, transcribe, chunk, embed, store."""
    chroma_client = get_chroma_client()
    embedding_fn = load_embedding_function()
    collection_name = collection_name_for(video_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        status.write("Downloading audio...")
        audio_path = download_audio(url, tmp_dir)

        status.write("Transcribing with Whisper (longer videos take a few minutes)...")
        segments = transcribe_audio(audio_path)

    status.write("Splitting transcript into chunks...")
    chunks = chunk_transcript(segments)

    status.write("Generating embeddings and storing in ChromaDB...")
    collection = chroma_client.get_or_create_collection(collection_name, embedding_function=embedding_fn)
    collection.add(
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"start": chunk["start"], "video_id": video_id} for chunk in chunks],
        ids=[f"{collection_name}_{chunk['index']}" for chunk in chunks],
    )
    return collection


def load_existing_collection(video_id: str) -> object:
    """Reconnect to a video's previously-stored ChromaDB collection."""
    chroma_client = get_chroma_client()
    embedding_fn = load_embedding_function()
    return chroma_client.get_collection(collection_name_for(video_id), embedding_function=embedding_fn)


def retrieve_relevant_chunks(collection, question: str) -> list[dict]:
    """Embed the question and fetch the TOP_K most similar transcript chunks."""
    results = collection.query(query_texts=[question], n_results=TOP_K)
    return [
        {"text": doc, "start": meta["start"], "video_id": meta["video_id"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a markdown block with clickable timestamps."""
    sections = []
    for chunk in chunks:
        timestamp = format_timestamp(chunk["start"])
        link = youtube_timestamp_link(chunk["video_id"], chunk["start"])
        sections.append(f"[Timestamp {timestamp}]({link})\n{chunk['text']}")
    return "\n\n---\n\n".join(sections)


def generate_answer(question: str, chunks: list[dict], history: list[dict]) -> str:
    """Ask Mistral to answer the question, grounded strictly in the retrieved chunks."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    llm = get_mistral_client()
    if llm is None:
        return "MISTRAL_API_KEY is not set. Add it to your .env file and restart the app."

    system_prompt = (
        "You are a knowledgeable assistant that answers questions strictly based on "
        "the provided YouTube transcript excerpts.\n"
        "Each excerpt below is preceded by a clickable timestamp. When your answer "
        "references something from an excerpt, cite it as a markdown link in the same "
        "format, e.g. [1:23](https://youtu.be/VIDEO_ID?t=83).\n"
        "If the excerpts don't contain enough information to answer, say so honestly "
        "rather than guessing.\n\n"
        f"Relevant transcript excerpts:\n\n{build_context_block(chunks)}"
    )

    messages = [SystemMessage(content=system_prompt)]
    for turn in history[-10:]:
        message_cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(message_cls(content=turn["content"]))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content


st.set_page_config(
    page_title="YouTube Transcript RAG",
    page_icon=":movie_camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_SESSION_STATE = {
    "messages": [],
    "history": [],
    "video_id": None,
    "collection": None,
}
for key, default_value in DEFAULT_SESSION_STATE.items():
    st.session_state.setdefault(key, default_value)


def render_sidebar() -> tuple[str, bool]:
    """Render the sidebar and return the entered URL and whether 'Process' was clicked."""
    with st.sidebar:
        st.title("YouTube RAG")
        st.caption("Ask questions about any YouTube video.")
        st.divider()

        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
        process_clicked = st.button(
            "Process Video",
            use_container_width=True,
            type="primary",
            disabled=not url.strip(),
        )

        if st.session_state.video_id:
            _render_active_video_controls()

        st.divider()
        _render_info_panel()

        if not os.getenv("MISTRAL_API_KEY"):
            st.warning("MISTRAL_API_KEY not found in .env")

    return url, process_clicked


def _render_active_video_controls() -> None:
    video_id = st.session_state.video_id
    st.success(f"Loaded: {video_id}")
    st.markdown(f"[Open on YouTube](https://youtu.be/{video_id})")

    if st.session_state.messages and st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    if st.button("Reset", use_container_width=True):
        for key, default_value in DEFAULT_SESSION_STATE.items():
            st.session_state[key] = default_value
        st.rerun()


def _render_info_panel() -> None:
    with st.expander("How it works"):
        st.markdown(
            "1. Paste a YouTube URL and click Process Video\n"
            "2. Audio is downloaded with yt-dlp\n"
            "3. Transcribed locally with OpenAI Whisper\n"
            "4. Chunks are stored as embeddings in ChromaDB\n"
            "5. Your question retrieves the top 5 relevant chunks\n"
            "6. Mistral Small answers with timestamp references\n\n"
            "Re-pasting the same URL skips transcription - it loads from cache instantly."
        )


def handle_video_processing(url: str) -> None:
    """Resolve a submitted URL into either a cached or freshly-ingested collection."""
    video_id = extract_video_id(url.strip())
    if not video_id:
        st.sidebar.error("Could not find a video ID in that URL.")
        return

    if video_id == st.session_state.video_id:
        st.sidebar.info("This video is already loaded and ready.")
        return

    chroma_client = get_chroma_client()
    collection_name = collection_name_for(video_id)

    if collection_already_exists(chroma_client, collection_name):
        with st.spinner("Loading existing transcript from ChromaDB..."):
            collection = load_existing_collection(video_id)
        _activate_video(video_id, collection)
        st.toast("Loaded from cache")
        st.rerun()
        return

    with st.status("Processing video...", expanded=True) as status:
        try:
            collection = ingest_video(url.strip(), video_id, status)
            _activate_video(video_id, collection)
            status.update(label="Video ready. Start asking questions.", state="complete")
        except Exception as exc:
            status.update(label=f"Failed: {exc}", state="error")
            st.error(
                f"Processing error: {exc}\n\n"
                "Make sure ffmpeg is installed and the video is publicly accessible."
            )


def _activate_video(video_id: str, collection: object) -> None:
    st.session_state.video_id = video_id
    st.session_state.collection = collection
    st.session_state.messages = []
    st.session_state.history = []


def render_chat() -> None:
    """Render the main chat area: history, input box, retrieval, and generation."""
    st.title("YouTube Transcript RAG")
    st.markdown(
        "Semantic search over video transcripts, powered by Whisper, "
        "ChromaDB and Mistral Small."
    )

    if st.session_state.collection is None:
        st.info(
            "Paste a YouTube URL in the sidebar and click Process Video to get started.\n\n"
            "Transcription happens locally - no OpenAI API key needed."
        )
        return

    video_id = st.session_state.video_id
    st.markdown(f"Chatting about: [youtube.com/watch?v={video_id}](https://www.youtube.com/watch?v={video_id})")
    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the video...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching transcript and generating answer..."):
            chunks = retrieve_relevant_chunks(st.session_state.collection, question)
            answer = generate_answer(question, chunks, st.session_state.history)
        st.markdown(answer)
        _render_source_chunks(chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})


def _render_source_chunks(chunks: list[dict]) -> None:
    with st.expander("Source transcript chunks", expanded=False):
        for i, chunk in enumerate(chunks, start=1):
            timestamp = format_timestamp(chunk["start"])
            link = youtube_timestamp_link(chunk["video_id"], chunk["start"])
            st.markdown(f"Chunk {i} - [{timestamp}]({link})")
            st.caption(chunk["text"])
            if i < len(chunks):
                st.divider()


def main() -> None:
    url, process_clicked = render_sidebar()
    if process_clicked and url.strip():
        handle_video_processing(url)
    render_chat()


if __name__ == "__main__":
    main()