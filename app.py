import os
import streamlit as st
import google.generativeai as genai
from st_audiorec import st_audiorec
from datetime import datetime
import tempfile

# Configure Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def generate_summary(audio_file_path, user_instruction):
    print(f"Generating summary:")
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-002",
        generation_config=generation_config,
        system_instruction=f"You are a world-class summarizer. Create a summary based on the content below. Output in nicely markdown syntax with bullet points if appropriated.  {user_instruction}"
    )

    file = upload_to_gemini(audio_file_path, mime_type="audio/wav")
    
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [file],
            },
        ]
    )
    response = chat_session.send_message("Please provide a summary of the audio content.")
    return response.text

def main():
    st.set_page_config(page_title="Medical Transcriber", layout="wide")
    st.title("Medical Transcriber")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Audio Recording")
        st.info("Please wait for the success message after hitting stop, so the tool can process the audio.")
        wav_audio_data = st_audiorec()
        
        if wav_audio_data is not None:
            # st.info("Processing the audio file...")
            # Save the recorded audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(wav_audio_data)
                st.session_state.audio_file_path = temp_audio_file.name
                st.success("Audio file has been processed and saved successfully.")

        st.subheader("Or Select Saved Audio")
        uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(uploaded_file.read())
                st.session_state.audio_file_path = temp_audio_file.name

    with col2:
        st.subheader("Custom Instruction for Summary")
        user_instruction = st.text_area("Edit your custom instruction for summary:", height=200, value="""
        Create a concise summary that captures the main points and important details from the audio content. Follow these guidelines:

        1. Identify the main topic or theme.
        2. List key points, arguments, or findings.
        3. Note any important data, statistics, or quotes.
        4. Present the ideas in a logical order.
        5. Use clear, simple language.
        6. Ensure the summary is concise yet comprehensive, with a maximum of 200 words.
        7. Format the summary using paragraphs, bullet points, or numbered lists as appropriate.
        8. Review for clarity, coherence, and accuracy.

        Present your final summary within <summary> tags.
                """)

    if st.button("Summarize Audio"):
        if hasattr(st.session_state, 'audio_file_path'):
            with st.spinner("Generating summary..."):
                summary = generate_summary(st.session_state.audio_file_path, user_instruction)

            st.subheader("Summary")
            st.write(summary)

            # Clean up the temporary file
            os.unlink(st.session_state.audio_file_path)
            del st.session_state.audio_file_path
        else:
            st.error("Please record or upload an audio file first.")

if __name__ == "__main__":
    main()