import streamlit as st
import os
import tempfile
import numpy as np
import time
import json
from pathlib import Path
from faster_whisper import WhisperModel
import openai
from datetime import datetime
from audio_recorder_streamlit import audio_recorder
from PIL import Image
import base64
from io import BytesIO
from google.oauth2 import service_account
from google.cloud import texttospeech
import google.auth
from urllib.parse import quote
# Import the new answer evaluation module
from answer_evaluation import get_answer_evaluation, save_evaluation_data, calculate_aggregate_scores, aggregate_skill_assessment, generate_career_insights


# Set path to Google Cloud credentials file
try:
    service_account_info = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
except Exception as e:
    st.error(f"Error initializing Google Cloud TTS client: {e}")
    tts_client = None

# Set OpenAI API key
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize session state variables
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_question_idx" not in st.session_state:
    st.session_state.current_question_idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "use_gpu" not in st.session_state:
    st.session_state.use_gpu = False
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False
if "selected_job_field" not in st.session_state:
    st.session_state.selected_job_field = None
if "setup_stage" not in st.session_state:
    st.session_state.setup_stage = "welcome_page"  # Changed to welcome_page
if "question_spoken" not in st.session_state:
    st.session_state.question_spoken = False
if "use_voice" not in st.session_state:
    st.session_state.use_voice = True
if "interviewer_name" not in st.session_state:
    st.session_state.interviewer_name = ""
if "voice_type" not in st.session_state:
    st.session_state.voice_type = "en-US-Neural2-D"
# New variable to track interview stage
if "interview_stage" not in st.session_state:
    st.session_state.interview_stage = "introduction"
# Add evaluations list to store structured evaluation data
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

@st.cache_resource
def load_whisper_model():
    model_size = "small" if st.session_state.get("faster_transcription", True) else "medium"
    device = "cuda" if st.session_state.get("use_gpu", False) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)

# Initialize Google Cloud Text-to-Speech client
@st.cache_resource
def get_tts_client():
    try:
        # Load credentials from Streamlit secrets
        service_account_info = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        client = texttospeech.TextToSpeechClient(credentials=credentials)
        return client
    except Exception as e:
        st.error(f"Error initializing Google Cloud TTS client: {str(e)}")
        return None


# Function to generate speech from text using Google Cloud TTS
def text_to_speech(text):
    client = get_tts_client()
    if not client:
        raise Exception("Failed to initialize Google Cloud TTS client")
    
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=st.session_state.voice_type,
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    
    # Select the type of audio file
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.95,  # Slightly slower for interview questions
        pitch=0.0,  # Natural pitch
        volume_gain_db=1.0  # Slightly louder
    )
    
    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    # Return the audio content as a BytesIO object
    fp = BytesIO(response.audio_content)
    fp.seek(0)
    return fp

# Function to create an HTML audio player with autoplay for TTS
def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# Helper function to get the base URL
def get_base_url():
    """Get the base URL for the current deployment"""
    # Update to use the Vercel-hosted dashboard URL
    return "https://intervuai-dashboard.vercel.app"  # Production dashboard URL

st.set_page_config(
    page_title="Interview Agent",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
JOB_FIELDS = {
    "Software Engineering": {
        "Technical": [
            "Explain the difference between arrays and linked lists.",
            "What's your approach to debugging a complex issue?",
            "Describe a challenging technical problem you solved recently.",
            "Explain the concept of time and space complexity.",
            "What design patterns have you used in your projects?",
            "How do you ensure your code is maintainable and scalable?",
            "Explain how you would implement error handling in a distributed system."
        ],
        "Behavioral": [
            "Tell me about a time you had to work under pressure to meet a deadline.",
            "Describe a situation where you disagreed with a team member on a technical approach.",
            "How do you handle feedback on your code during code reviews?",
            "Tell me about a time you identified and fixed a bug that others couldn't solve.",
            "How do you keep up with the latest technologies and programming languages?"
        ],
        "Role-specific": [
            "How do you approach testing your code?",
            "Describe your experience with CI/CD pipelines.",
            "How do you balance technical debt with delivering features?",
            "Explain your approach to optimizing application performance.",
            "How would you explain a complex technical concept to a non-technical stakeholder?"
        ]
    },
    "Data Science/Analysis": {
        "Technical": [
            "Explain the difference between supervised and unsupervised learning.",
            "How do you handle missing data in a dataset?",
            "Describe a data cleaning process you've implemented.",
            "What statistical methods do you use to validate your findings?",
            "Explain the concept of overfitting and how to avoid it.",
            "How would you approach feature selection for a machine learning model?",
            "Explain the difference between correlation and causation with an example."
        ],
        "Behavioral": [
            "Tell me about a time when your data analysis led to a significant business decision.",
            "How do you communicate complex data insights to non-technical stakeholders?",
            "Describe a situation where you had to defend your analytical approach.",
            "Tell me about a project where you had to work with messy or incomplete data.",
            "How do you ensure your analysis is accurate and reliable?"
        ],
        "Role-specific": [
            "What visualization tools do you prefer and why?",
            "How do you determine which statistical test to use for a given problem?",
            "Describe your approach to A/B testing.",
            "How do you translate business questions into data queries?",
            "What metrics would you track to measure the success of a product feature?"
        ]
    },
    "Project Management": {
        "Technical": [
            "What project management methodologies are you familiar with?",
            "How do you create and maintain a project schedule?",
            "Describe your approach to risk management.",
            "How do you track and report project progress?",
            "What tools do you use for project planning and why?",
            "How do you handle resource allocation in a project?",
            "Explain how you would manage scope creep."
        ],
        "Behavioral": [
            "Tell me about a time when a project was falling behind schedule.",
            "Describe how you've managed stakeholder expectations.",
            "How do you motivate team members during challenging phases of a project?",
            "Tell me about a project that failed and what you learned from it.",
            "How do you handle conflicts between team members or departments?"
        ],
        "Role-specific": [
            "How do you prioritize competing deadlines across multiple projects?",
            "Describe how you communicate project status to different audiences.",
            "How do you ensure quality deliverables while maintaining timelines?",
            "What's your approach to gathering requirements from stakeholders?",
            "How do you manage project budgets and resources?"
        ]
    },
    "UX/UI Design": {
        "Technical": [
            "Walk me through your design process.",
            "How do you approach user research?",
            "Describe how you create and use personas.",
            "What tools do you use for wireframing and prototyping?",
            "How do you incorporate accessibility into your designs?",
            "Explain the importance of design systems.",
            "How do you use data to inform design decisions?"
        ],
        "Behavioral": [
            "Tell me about a time when you received difficult feedback on your design.",
            "Describe a situation where you had to compromise on a design decision.",
            "How do you advocate for the user when there are business constraints?",
            "Tell me about a design challenge you faced and how you overcame it.",
            "How do you collaborate with developers to implement your designs?"
        ],
        "Role-specific": [
            "How do you measure the success of a design?",
            "Describe how you stay current with design trends and best practices.",
            "How do you balance aesthetics with usability?",
            "Explain your approach to responsive design.",
            "How would you improve the user experience of our product?"
        ]
    },
    "IT Support": {
        "Technical": [
            "Explain the difference between hardware and software troubleshooting.",
            "How would you approach a user who can't connect to the internet?",
            "Describe your experience with ticketing systems.",
            "What steps would you take to secure a workstation?",
            "How do you prioritize multiple support requests?",
            "Explain how you would troubleshoot a slow computer."
        ],
        "Behavioral": [
            "Tell me about a time when you had to explain a technical issue to a non-technical user.",
            "Describe a situation where you went above and beyond for a user.",
            "How do you handle frustrated or angry users?",
            "Tell me about a time when you couldn't solve a technical problem immediately.",
            "How do you stay patient when dealing with repetitive support issues?"
        ],
        "Role-specific": [
            "What remote support tools are you familiar with?",
            "How do you document your troubleshooting steps?",
            "Describe your approach to user training and education.",
            "How do you keep up with new technologies and support techniques?",
            "What's your experience with supporting remote workers?"
        ]
    },
    "Cybersecurity": {
        "Technical": [
            "Explain the concept of defense in depth.",
            "What's the difference between authentication and authorization?",
            "How would you respond to a potential data breach?",
            "Describe common network vulnerabilities and how to mitigate them.",
            "What's your approach to vulnerability assessment?",
            "Explain the importance of patch management."
        ],
        "Behavioral": [
            "Tell me about a time when you identified a security risk before it became an issue.",
            "How do you balance security needs with user convenience?",
            "Describe a situation where you had to convince management to invest in security measures.",
            "How do you stay current with evolving security threats?",
            "Tell me about a time when you had to respond to a security incident."
        ],
        "Role-specific": [
            "What security tools and technologies are you experienced with?",
            "How would you implement a security awareness program?",
            "Describe your experience with compliance requirements (GDPR, HIPAA, etc.)",
            "What's your approach to security logging and monitoring?",
            "How would you conduct a security audit?"
        ]
    }
}

COMMON_QUESTIONS = {
    "Background": [
        "Tell me more about yourself and why you're interested in this field."
    ]
}

def generate_questions(job_field, num_questions):
    questions = []
    
    # Define all categories we want to include
    all_categories = ["Background", "Technical", "Behavioral", "Role-specific"]
    
    # Define the question order sequence
    category_order = []
    
    # Add all categories that exist in the job field
    for category in all_categories:
        if category in COMMON_QUESTIONS:
            category_order.append(category)
        elif category in JOB_FIELDS[job_field]:
            category_order.append(category)
    
    # Start with at least one question from each category in the specified order
    for category in category_order:
        if category in JOB_FIELDS[job_field]:
            questions.append({
                "category": category,
                "question": np.random.choice(JOB_FIELDS[job_field][category])
            })
        elif category in COMMON_QUESTIONS:
            questions.append({
                "category": category,
                "question": np.random.choice(COMMON_QUESTIONS[category])
            })
    
    # Fill remaining slots (if needed)
    remaining_slots = num_questions - len(questions)
    if remaining_slots > 0:
        question_pool = []
        for category in category_order:
            if category in JOB_FIELDS[job_field]:
                category_questions = JOB_FIELDS[job_field][category]
            elif category in COMMON_QUESTIONS:
                category_questions = COMMON_QUESTIONS[category]
            else:
                continue
            
            # Add more questions from the same categories, maintaining the order
            for q in category_questions:
                if {"category": category, "question": q} not in questions:
                    question_pool.append({"category": category, "question": q})
        
        if question_pool:
            # Group by category to maintain order while selecting additional questions
            grouped_pool = {}
            for q in question_pool:
                if q["category"] not in grouped_pool:
                    grouped_pool[q["category"]] = []
                grouped_pool[q["category"]].append(q)
            
            additional_questions = []
            remaining = remaining_slots
            
            # Take additional questions from each category in order until we've filled the slots
            while remaining > 0 and any(len(grouped_pool.get(cat, [])) > 0 for cat in category_order):
                for cat in category_order:
                    if cat in grouped_pool and grouped_pool[cat]:
                        # Randomly select one question from this category
                        idx = np.random.randint(0, len(grouped_pool[cat]))
                        additional_questions.append(grouped_pool[cat].pop(idx))
                        remaining -= 1
                        if remaining == 0:
                            break
            
            questions.extend(additional_questions)
    
    # No need to shuffle since we want to maintain the category order
    return questions

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file)
        temp_audio_path = temp_audio.name
    
    model = load_whisper_model()
    
    if st.session_state.get("faster_transcription", True):
        segments, info = model.transcribe(
            temp_audio_path, 
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            language="en"
        )
    else:
        segments, info = model.transcribe(
            temp_audio_path, 
            beam_size=5,
            language="en"
        )
    
    transcript = ""
    for segment in segments:
        transcript += segment.text + " "
    
    os.unlink(temp_audio_path)
    
    return transcript.strip()

# Enhanced answer feedback function that uses structured evaluation
def get_answer_feedback(question, answer):
    """
    Get detailed feedback for an interview answer
    
    Parameters:
    - question: The interview question
    - answer: The candidate's answer
    
    Returns:
    - String containing formatted feedback
    """
    # Get structured evaluation data
    job_field = st.session_state.selected_job_field or "General"
    eval_data = get_answer_evaluation(question, answer, job_field)
    
    # Store the evaluation data for later use with the dashboard
    if 'evaluations' not in st.session_state:
        st.session_state.evaluations = []
    
    # Only add to evaluations if not already there (to avoid duplicates)
    question_answers = [(e["question"], e["answer"]) for e in st.session_state.evaluations]
    if (question, answer) not in question_answers:
        st.session_state.evaluations.append(eval_data)
    
    # Format the evaluation data as a string for display
    feedback = f"""
    ## Feedback on Your Answer
    
    ### Strengths:
    {chr(10).join(['- ' + s for s in eval_data['feedback']['strengths']])}
    
    ### Areas for Improvement:
    {chr(10).join(['- ' + s for s in eval_data['feedback']['areas_for_improvement']])}
    
    ### Missing Elements:
    {chr(10).join(['- ' + s for s in eval_data['feedback']['missing_elements']])}
    
    ### Performance Scores:
    - Content: {eval_data['scores']['content']}/10
    - Clarity: {eval_data['scores']['clarity']}/10
    - Technical Accuracy: {eval_data['scores']['technical_accuracy']}/10
    - Confidence: {eval_data['scores']['confidence']}/10
    - Overall: {eval_data['scores']['overall']}/10
    
    ### Example Improved Response:
    {eval_data['improved_answer']}
    """
    
    return feedback

# Function to convert image to base64 for embedding in HTML
def get_image_base64(image_path):
    """Convert an image to base64 encoding"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Interview progress sidebar - show only when in interview mode
if st.session_state.questions and st.session_state.setup_stage == "interview":
    with st.sidebar:
        st.title("Interview Progress")
        progress = (st.session_state.current_question_idx) / len(st.session_state.questions)
        st.progress(progress)
        st.write(f"Question {st.session_state.current_question_idx + 1} of {len(st.session_state.questions)}")
        
        if st.button("End Interview & See Results"):
            st.session_state.interview_complete = True
            st.rerun()
        
        if st.button("Restart Interview"):
            for key in ['questions', 'current_question_idx', 'answers', 'feedbacks', 
                       'recording', 'audio_data', 'transcription', 'interview_complete', 
                       'show_feedback', 'question_spoken', 'evaluations']:
                if key in st.session_state:
                    if isinstance(st.session_state[key], list):
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = False
            st.session_state.current_question_idx = 0
            st.session_state.questions = []
            st.session_state.interview_stage = "introduction"
            st.session_state.setup_stage = "job_selection"
            st.rerun()

# MAIN APPLICATION FLOW - now with proper if/elif structure 
# Welcome page/landing screen
if st.session_state.setup_stage == "welcome_page" and not st.session_state.questions:
    # Use the absolute path for the logo
    logo_path = Path(__file__).parent / "image.png"
    logo_image = Image.open(logo_path)
    
    
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: white;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    div[data-testid="stDecoration"] {
        display: none;
    }
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    #MainMenu {
        display: none;
    }
    footer {
        display: none;
    }
    header {
        display: none;
    }
    .main-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .main.css-k1vhr4.egzxvld5, .main.css-k1vhr4.egzxvld4, .main.css-k1vhr4.egzxvld3, .main.css-k1vhr4.egzxvld2, .main.css-k1vhr4.egzxvld1, .main.css-k1vhr4.egzxvld0 {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    section[data-testid="stSidebar"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    div[data-testid="stVerticalBlock"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
        max-width: none !important;
    }
    div.block-container.css-91z34k.egzxvld4, div.block-container.css-91z34k.egzxvld3, div.block-container.css-91z34k.egzxvld2, div.block-container.css-91z34k.egzxvld1, div.block-container.css-91z34k.egzxvld0 {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .element-container, .stMarkdown {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .stButton > button {
        background-color: #3498db;  /* Blue to match logo */
        color: white;
        border: none;
        border-radius: 4px;
        padding: 15px 25px;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        white-space: nowrap !important;     /* Prevent line break */
        overflow: hidden !important;        /* Hide overflow */
        text-overflow: ellipsis !important; /* Add "..." if it does overflow */
        word-break: normal !important;      /* Prevent breaking within words */
        display: inline-flex !important;    /* Aligns content horizontally */
        justify-content: center !important;
        align-items: center !important;
    }
    .stButton > button:hover {
        background-color: #2980b9;  /* Darker blue on hover */
        transform: scale(1.02);
    }
    .feature-card {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 130px;  /* Fixed height for cards */
        overflow-y: auto;  /* Allow scrolling if content exceeds height */
    }
    .feature-icon {
        font-size: 24px;
        margin-bottom: 10px;
        color: #3498db;  /* Blue to match logo */
    }
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto;  /* Centered with no bottom margin */
        padding: 0;
    }
    .logo-image {
        width: 220px;  /* Slightly smaller */
        height: auto;
    }
    .section-container {
        max-height: 600px;  /* Limit the overall height */
        overflow-y: auto;  /* Enable scrolling if needed */
    }
    /* Force fullscreen mode */
    [data-testid="stHeader"] {
        display: none !important;
    }
    .appview-container .main .block-container {
        padding-top: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Custom styling for Streamlit button columns */
    div.row-widget.stButton {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    # Find this CSS section in your welcome_page code and replace it with the following:

/* Sign In button */
div[data-testid="column"]:nth-of-type(2) .stButton > button {
    background-color: transparent !important;
    color: white !important;
    border: 1px solid #3498db !important;
    border-radius: 4px !important;
    padding: 8px 15px !important;
    height: 40px !important;
    font-size: 14px !important;
    font-weight: bold !important;
    margin: 0 !important;
    text-align: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    width: 1500px !important;  /* ← increased width */
    min-width: 110px !important;
    max-width: 110px !important;
}

/* Sign Up button */
div[data-testid="column"]:nth-of-type(3) .stButton > button {
    background-color: #3498db !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 8px 15px !important;
    height: 40px !important;
    font-size: 14px !important;
    font-weight: bold !important;
    margin: 0 !important;
    text-align: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    width: 110px !important;  /* ← increased width */
    min-width: 110px !important;
    max-width: 110px !important;
}


/* Fix button container widths to match buttons */
div[data-testid="column"]:nth-of-type(2) .stButton,
div[data-testid="column"]:nth-of-type(3) .stButton {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100px !important;  /* Match the button width */
    min-width: 100px !important;
    max-width: 100px !important;
    margin: 0 auto !important;
}
# FIXED BUTTON STYLES - END
    /* FIXED BUTTON STYLES - END */
    
    /* Fix column spacing */
    div[data-testid="column"] {
        padding: 0 5px !important;
    }
    
    /* Align top row */
    div.css-ocqkz7.e1tzin5v4:first-of-type {
        margin-top: 10px !important;
        padding-top: 0 !important;
    }
    
    /* Fix for ensuring buttons are at the same height */
    div[data-testid="column"]:nth-of-type(2), div[data-testid="column"]:nth-of-type(3) {
        display: flex !important;
        align-items: center !important;
    }
    
    /* Fix button container alignment */
    div.row-widget.stHorizontal {
        display: flex !important;
        justify-content: flex-end !important;
        padding-right: 10px !important;
        
    }
</style>
    """, unsafe_allow_html=True)
    
    # Add super aggressive negative margin to move everything up
    st.markdown('<div style="margin-top: -150px;"></div>', unsafe_allow_html=True)
    
    # Direct approach by inserting a full-page container with no margins
    st.markdown("""
    <div style="position: absolute; top: 0; left: 0; width: 100%; margin: 0; padding: 0;">
    """, unsafe_allow_html=True)
    
    # Use columns to create a row for auth buttons at the top
    # Changed from [8, 1, 1, 1] to [8, 1, 1] to fix alignment
    auth_col_spacer, col_signin, col_signup = st.columns([8, 1, 1])
    
    with col_signin:
        if st.button("Sign In", key="signin_button", use_container_width=True):
            st.session_state.setup_stage = "sign_in"
            st.rerun()
    
    with col_signup:
        if st.button("Sign Up", key="signup_button", use_container_width=True):
            st.session_state.setup_stage = "sign_up"
            st.rerun()
    
    # Logo and main title - moved further up with less margin
    try:
        logo_base64 = get_image_base64(logo_path)
        st.markdown("""
        <div class="logo-container" style="margin-top: 10px;">
            <img src="data:image/png;base64,{}" class="logo-image" alt="IntervuAI Logo">
        </div>
        <div style="text-align: center; padding: 0; margin: 0;">
            <h1 style="font-size: 22px; font-weight: bold; margin: 0 0 5px 0;padding-left: 20px;">
                Your AI-Powered Interview Preparation Assistant
            </h1>
        </div>
        """.format(logo_base64), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading logo: {str(e)}")
        st.markdown("""
        <div style="text-align: center; padding: 0;">
            <h1 style="font-size: 48px; font-weight: bold; margin: 0; color: #3498db;">
                IntervuAI
            </h1>
            <h2 style="font-size: 22px; font-weight: normal; margin: 5px 0 10px 0; color: #e0e0e0;">
                Your AI-Powered Interview Preparation Assistant
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Added a container div with max-height to limit the overall section height
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    
    # Features and How it Works sections - both with matching card style
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("<h3>Key Features</h3>", unsafe_allow_html=True)
        
        # Feature cards with fixed height - UPDATED LEFT COLUMN with matching styles
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 24px; margin-right: 10px; color: #3498db;">🎯</div>
                <h4 style="margin: 0; padding: 0; display: inline; color: #3498db;">Industry-Specific Questions</h4>
            </div>
            <p>Tailored questions for software engineering, data science, project management, UX/UI design, and more.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 24px; margin-right: 10px; color: #3498db;">🎤</div>
                <h4 style="margin: 0; padding: 0; display: inline; color: #3498db;">Voice-Enabled Interviews</h4>
            </div>
            <p>Experience realistic interviews with voice questions and verbal answers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 24px; margin-right: 10px; color: #3498db;">📊</div>
                <h4 style="margin: 0; padding: 0; display: inline; color: #3498db;">Performance Analysis</h4>
            </div>
            <p>Get scores for content, clarity, technical accuracy, and confidence.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 24px; margin-right: 10px; color: #3498db;">💡</div>
                <h4 style="margin: 0; padding: 0; display: inline; color: #3498db;">AI-Powered Improvement Tips</h4>
            </div>
            <p>Receive suggestions to improve your answers with example responses.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("<h3>How It Works</h3>", unsafe_allow_html=True)
        
        # Using a flexbox approach for the right column
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="background-color: #3498db; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">1</div>
                <h4 style="margin: 0; padding: 0; display: inline; color: #3498db;">Select Your Industry</h4>
            </div>
            <p>Choose your preferred job field for targeted practice.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="background-color: #3498db; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">2</div>
                <h4 style="margin: 0; padding: 0; display: inline; color: #3498db;">Customize Your Interview</h4>
            </div>
            <p>Set the number of questions and voice preferences.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="background-color: #3498db; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">3</div>
                <h4 style="margin: 0; padding: 0; display: inline; color: #3498db;">Practice with Realistic Questions</h4>
            </div>
            <p>Answer technical, behavioral, and role-specific questions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="background-color: #3498db; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">4</div>
                <h4 style="margin: 0; padding: 0; display: inline; color: #3498db;">Review Your Performance</h4>
            </div>
            <p>Access detailed feedback and performance insights.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Close the section container and the full-page container
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Start button centered at the bottom
    _, center_btn_col, _ = st.columns([1, 2, 1])
    
    with center_btn_col:
        st.markdown("<div style='text-align: center; padding: 15px 0;'>", unsafe_allow_html=True)
        if st.button("Start Your Interview Practice", key="start_welcome_button", use_container_width=True):
            st.session_state.setup_stage = "job_selection"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Below is where you would add the actual authentication functionality once the buttons work
    # For example, you could set up a simple modal system or expand this to use a database
elif st.session_state.setup_stage == "sign_in":
    st.title("🔐 Sign In")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # In real implementation: check from database
        if email == "admin@example.com" and password == "admin123":
            st.success("Logged in successfully!")
            st.session_state.setup_stage = "job_selection"
            st.rerun()
        else:
            st.error("Invalid credentials")

    if st.button("Back to Welcome"):
        st.session_state.setup_stage = "welcome_page"
        st.rerun()

elif st.session_state.setup_stage == "sign_up":
    st.title("📝 Sign Up")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match")
        else:
            st.success("Account created! You can now log in.")
            st.session_state.setup_stage = "sign_in"
            st.rerun()

    if st.button("Back to Welcome"):
        st.session_state.setup_stage = "welcome_page"
        st.rerun()
# Job selection screen
elif st.session_state.setup_stage == "job_selection" and not st.session_state.questions:
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    div[data-testid="stDecoration"] {
        display: none;
    }
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    #MainMenu {
        display: none;
    }
    footer {
        display: none;
    }
    header {
        display: none;
    }
    .stButton > button {
        background-color: #1E1E1E;  /* Darker background like feature cards */
        color: white;  /* White text */
        border: 1px solid #3498db;  /* Blue border matching welcome page */
        border-radius: 4px;
        padding: 15px;
        font-size: 16px;
        font-weight: normal;
        text-align: left;
        margin: 10px 0;
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;  /* Smooth transition for hover effect */
    }
    .stButton > button:hover {
        background-color: #3498db;  /* Same blue as welcome page on hover */
        transform: scale(1.02);  /* Slight scale effect on hover like welcome page */
    }
    .stButton > button::after {
        content: "›";
        font-size: 24px;
    }
    h1, h2 {
        color: white !important;
    }
    h2 span {
        color: #3498db !important;  /* Blue accent color for part of the heading */
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 5px;"></div>', unsafe_allow_html=True)
    
    _, center_title_col, _ = st.columns([1, 3, 1])
    
    with center_title_col:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="font-size: 48px; font-weight: normal; margin: 0; padding: 0; color: white;">
                Interview Agent
            </h1>
            <h2 style="font-size: 28px; font-weight: normal; margin: 0; padding: 0;">
                What <span style="color: #3498db;">field</span> do you want to practice for?
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        for job_field in JOB_FIELDS.keys():
            if st.button(f"{job_field}", key=f"{job_field}Button", use_container_width=True):
                st.session_state.selected_job_field = job_field
                st.session_state.setup_stage = "interview_settings"  # Changed to interview_settings
                st.rerun()
# Interview settings screen
# Question phase - Empty Version
elif st.session_state.setup_stage == "interview_settings" and not st.session_state.questions:
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    div[data-testid="stDecoration"] {
        display: none;
    }
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    #MainMenu {
        display: none;
    }
    footer {
        display: none;
    }
    header {
        display: none;
    }

    /* Text and heading colors */
    h1, h2, h3, h4 {
        color: white !important;
    }
    .blue-accent {
        color: #3498db !important;  /* Blue accent color */
    }

    /* Remove slider styling to use default */
    /* Slider styling removed as requested */

    /* Button styling to match job selection screen */
    .stButton > button {
        background-color: #1E1E1E !important;  /* Dark button */
        color: white !important;
        border: 1px solid #3498db !important;  /* Blue border */
        border-radius: 4px;
        padding: 10px 15px;
        font-size: 16px;
        font-weight: normal;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #3498db !important;  /* Blue on hover */
        transform: scale(1.02);
    }

    /* Primary button (Start Practice) */
    .stButton > [data-testid="baseButton-primary"] {
        background-color: #3498db !important;  /* Blue button */
        color: white !important;
        border: none !important;
        font-weight: bold;
    }

    /* Checkbox styling */
    [data-testid="stCheckbox"] > div > div {
        background-color: #3498db !important;  /* Blue checkbox */
    }

    /* Input fields - dark with blue accents */
    [data-testid="stTextInput"] > div > div > input {
        background-color: #1E1E1E !important;  /* Dark input fields */
        color: white !important;
        border-color: #3498db !important;  /* Blue border */
    }

    /* Selectbox - dark with blue accents */
    [data-testid="stSelectbox"] > div[data-baseweb="select"] > div {
        background-color: #1E1E1E !important;  /* Dark dropdown */
        border-color: #3498db !important;  /* Blue border */
    }
    [data-testid="stSelectbox"] div[role="listbox"] {
        background-color: #1E1E1E !important;  /* Dark dropdown items */
    }
    [data-testid="stSelectbox"] div[role="option"] {
        background-color: #1E1E1E !important;  /* Dark dropdown items */
        color: white !important;
    }
    [data-testid="stSelectbox"] div[role="option"]:hover {
        background-color: #3498db !important;  /* Blue hover */
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 48px; font-weight: normal; margin: 0; padding: 0; color: white;">
            Interview Settings
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display information about the selected job field
        st.markdown(f'<h2 style="font-size: 28px; font-weight: normal; margin-bottom: 20px;">Preparing interview for: <span class="blue-accent">{st.session_state.selected_job_field}</span></h2>', unsafe_allow_html=True)
        
        st.write("Your interview will include questions from multiple categories:")
        st.write("• Technical questions about your skills and knowledge")
        st.write("• Behavioral questions about your work experiences")
        st.write("• Role-specific questions relevant to the position")
        st.write("• Background questions to get to know you better")
    
    with col2:
        num_questions = st.slider("Number of questions:", 5, 15, 7)
        
        # Change label to clarify this is for the interviewee's name
        interviewee_name = st.text_input("Your name (interviewee):", value=st.session_state.interviewer_name)
        st.session_state.interviewer_name = interviewee_name
        
        st.session_state.use_voice = st.checkbox("Enable voice for questions", value=True)
        
        if st.session_state.use_voice:
            voice_options = {
                "Male (Default)": "en-US-Neural2-D", 
                
            }
            selected_voice = st.selectbox(
                "Select interviewer voice:",
                options=list(voice_options.keys()),
                index=0
            )
            st.session_state.voice_type = voice_options[selected_voice]
    
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", key="back_button"):
            st.session_state.setup_stage = "job_selection"
            st.session_state.selected_job_field = None
            st.rerun()
    
    with col3:
        if st.button("Start Practice →", type="primary", key="start_practice"):
            # Generate questions without category selection
            st.session_state.questions = generate_questions(
                st.session_state.selected_job_field,
                num_questions
            )
            st.session_state.current_question_idx = 0
            st.session_state.answers = [""] * len(st.session_state.questions)
            st.session_state.feedbacks = [""] * len(st.session_state.questions)
            st.session_state.evaluations = []  # Reset evaluations
            st.session_state.interview_complete = False
            st.session_state.show_feedback = False
            st.session_state.question_spoken = False
            st.session_state.interview_stage = "introduction"
            st.session_state.setup_stage = "interview"
            st.rerun()

# Interview results screen
# Interview results screen
elif st.session_state.interview_complete:
    st.title("Interview Practice Results")
    
    if st.session_state.answers and not all(answer == "" for answer in st.session_state.answers):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        session_data = {
            "timestamp": timestamp,
            "questions": st.session_state.questions,
            "answers": st.session_state.answers,
            "feedbacks": st.session_state.feedbacks
        }
        st.session_state.session_history.append(session_data)
    
    # Create a file for the dashboard if we have evaluations
    dashboard_url = None
    if st.session_state.evaluations and len(st.session_state.evaluations) > 0:
        interviewee_name = st.session_state.interviewer_name or "candidate"
        try:
            output_path = save_evaluation_data(st.session_state.evaluations, interviewee_name)
            
            # Check if we got a URL or local path
            if output_path and isinstance(output_path, str) and output_path.startswith("http"):
                # Firebase upload succeeded
                dashboard_url = f"https://intervuai-dashboard.vercel.app/?data={output_path}"
                
                st.success("Your interview analysis is ready to view!")
                st.markdown(f"""
                ## Career Dashboard
                
                View a detailed analysis of your interview performance, including:
                - Career aptitude assessment
                - Skills analysis
                - Career path recommendations
                - Development opportunities
                
                [Open Career Dashboard]({dashboard_url})
                """)
            else:
                # Firebase upload failed, show a message and continue with fallback
                st.warning("Career dashboard will display limited data due to cloud storage limitations.")
                if output_path:
                    st.info(f"Your interview data has been saved locally at: {output_path}")
                
                # Create a simplified dashboard experience directly in the app
                st.subheader("Performance Summary")
                
                # Calculate and display aggregate scores
                agg_scores = calculate_aggregate_scores(st.session_state.evaluations)
                cols = st.columns(len(agg_scores))
                for i, (category, score) in enumerate(agg_scores.items()):
                    with cols[i]:
                        st.metric(category.title(), f"{score}/10")
                
                # Show top skills as a bar chart
                skill_data = aggregate_skill_assessment(st.session_state.evaluations)
                if skill_data["demonstrated_skills"]:
                    st.subheader("Top Skills Demonstrated")
                    # Convert to a format for Streamlit charting
                    skill_names = [item["name"] for item in skill_data["demonstrated_skills"][:5]]
                    skill_counts = [item["count"] for item in skill_data["demonstrated_skills"][:5]]
                    
                    # Create a dataframe for the chart
                    import pandas as pd
                    chart_data = pd.DataFrame({
                        "Skill": skill_names,
                        "Frequency": skill_counts
                    })
                    st.bar_chart(chart_data, x="Skill", y="Frequency")
                
                # Show career insights
                career_insights = generate_career_insights(st.session_state.evaluations)
                if career_insights and career_insights.get("careerPaths"):
                    st.subheader("Potential Career Paths")
                    for path in career_insights["careerPaths"][:2]:
                        st.markdown(f"""
                        **{path['name']}** - {path['compatibility']}% match
                        
                        {path['description']}
                        
                        **Key Skills:** {', '.join(path['keySkills'])}
                        """)
        except Exception as e:
            st.error(f"Error saving evaluation data: {str(e)}")
            st.info("Continuing with local results display.")
    
    st.markdown("---")
    
    # Display individual question feedback
    for i, (question_data, answer, feedback) in enumerate(zip(
            st.session_state.questions, 
            st.session_state.answers, 
            st.session_state.feedbacks)):
        
        with st.expander(f"Question {i+1}: {question_data['question']} ({question_data['category']})", expanded=i==0):
            st.write("**Your Answer:**")
            if answer:
                st.write(answer)
            else:
                st.write("*No answer provided*")
            
            st.write("**Feedback:**")
            if feedback:
                st.write(feedback)
            else:
                if answer:
                    with st.spinner("Generating feedback..."):
                        feedback = get_answer_feedback(question_data['question'], answer)
                        st.session_state.feedbacks[i] = feedback
                        st.write(feedback)
                else:
                    st.write("*No feedback available (no answer provided)*")
    
    if st.button("Practice Again", type="primary"):
        for key in ['questions', 'current_question_idx', 'answers', 'feedbacks', 
                   'recording', 'audio_data', 'transcription', 'interview_complete',
                   'question_spoken', 'evaluations']:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                else:
                    st.session_state[key] = False
        st.session_state.current_question_idx = 0
        st.session_state.questions = []
        st.session_state.interview_stage = "introduction"
        st.session_state.setup_stage = "job_selection"
        st.rerun()
    
    if st.session_state.session_history and st.button("View Practice History"):
        st.subheader("Your Practice History")
        for i, session in enumerate(reversed(st.session_state.session_history)):
            with st.expander(f"Session {len(st.session_state.session_history) - i}: {session['timestamp']}"):
                for j, (q_data, ans, feed) in enumerate(zip(session['questions'], session['answers'], session['feedbacks'])):
                    st.write(f"**Q{j+1}: {q_data['question']}** ({q_data['category']})")
                    st.write("*Your answer:*")
                    st.write(ans if ans else "*No answer recorded*")
                    if feed:
                        st.write("*Feedback:*")
                        st.write(feed)
                    st.divider()


# Interview Screen (introduction and questions)
elif st.session_state.setup_stage == "interview":
    # Validation to ensure job field is selected before interview
    if not st.session_state.selected_job_field:
        st.warning("No job field is selected. Please select a job field first.")
        st.session_state.setup_stage = "job_selection"
        st.rerun()
    
    # Introduction phase
    elif st.session_state.interview_stage == "introduction":
        try:
            # Create personalized introduction
            interviewee_name = st.session_state.interviewer_name or "candidate"
            job_role = st.session_state.selected_job_field
            
            intro_text = f"Hi {interviewee_name}, welcome to your interview practice for a {job_role} role. I'll be asking you a series of questions."
            
            # Generate audio for the introduction
            if st.session_state.use_voice:
                audio_fp = text_to_speech(intro_text)
                autoplay_audio(audio_fp)
            
            # Display the introduction text with minimal styling
            st.info(intro_text)
            
            # Show a "Continue" button to proceed to the first question
            if st.button("Continue to First Question", type="primary"):
                st.session_state.current_question_idx = 0
                st.session_state.interview_stage = "question"
                st.session_state.question_spoken = False
                st.rerun()
                
        except Exception as e:
            st.error(f"Error in introduction: {str(e)}")
            if st.button("Continue to Questions"):
                st.session_state.interview_stage = "question"
                st.rerun()
    
    # Question phase
    else:
        current_q_data = st.session_state.questions[st.session_state.current_question_idx]
        current_category = current_q_data["category"]
        current_question = current_q_data["question"]

        # Add a check to ensure we're in question mode
        if st.session_state.interview_stage == "question":
            # Add styling for the question phase (keeping dark theme with blue accents)
            st.markdown("""
            <style>
            /* Dark theme with blue accents */
            .stApp {
                background-color: #121212;
                color: white;
            }
            
            /* Progress bar styling */
            .stProgress > div > div {
                background-color: #3498db !important;
            }
            
            /* Button styling */
            .stButton > button {
                border-color: #3498db !important;
                color: white !important;
            }
            
            /* Primary button */
            .stButton > [data-testid="baseButton-primary"] {
                background-color: #3498db !important;
                color: white !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # Display question with minimalistic styling but dark theme
            # Font size increased from 20px to 24px
            st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <p style="color: #3498db; font-size: 14px; margin-bottom: 5px;">Question: {st.session_state.current_question_idx + 1} of {len(st.session_state.questions)}</p>
                <p style="font-size: 24px; margin-top: 0;">{current_question}</p>
            </div>
            """, unsafe_allow_html=True)

            # Display a simple progress bar
            st.progress((st.session_state.current_question_idx) / len(st.session_state.questions))

            # Ensure the question gets spoken once
            if not st.session_state.question_spoken and st.session_state.use_voice:
                try:
                    # Simple spoken question
                    spoken_question = current_question

                    # Generate TTS audio
                    audio_fp = text_to_speech(spoken_question)

                    if audio_fp:
                        autoplay_audio(audio_fp)
                        time.sleep(1)
                        st.session_state.question_spoken = True
                    else:
                        st.error("TTS audio was not generated correctly.")

                except Exception as e:
                    st.error(f"Error playing question audio: {str(e)}")
                    st.session_state.question_spoken = True

            # Response input area with minimal styling
            if st.session_state.transcription:
                # Display transcribed answer with dark theme styling
                st.markdown(f"""
                <div style="margin-bottom: 20px;">
                    <p style="color: #3498db; font-size: 14px; margin-bottom: 5px;">Your answer:</p>
                    <p style="background-color: #1E1E1E; padding: 15px; border-radius: 4px;">{st.session_state.transcription}</p>
                </div>
                """, unsafe_allow_html=True)

                edited_answer = st.text_area(
                    "Edit your answer if needed:",
                    value=st.session_state.transcription,
                    height=150
                )

                if st.button("Save Answer & Continue", type="primary"):
                    st.session_state.answers[st.session_state.current_question_idx] = edited_answer
                    with st.spinner("Generating feedback..."):
                        feedback = get_answer_feedback(current_question, edited_answer)
                        st.session_state.feedbacks[st.session_state.current_question_idx] = feedback

                    st.session_state.current_question_idx += 1
                    st.session_state.transcription = ""
                    st.session_state.audio_data = None
                    st.session_state.question_spoken = False

                    if st.session_state.current_question_idx >= len(st.session_state.questions):
                        st.session_state.interview_complete = True

                    st.rerun()

            else:
                # WhatsApp-style input with microphone button
                st.markdown("""
                <style>
                /* Styling for the input container */
                .whatsapp-input {
                    display: flex;
                    align-items: center;
                    background-color: #1E2130;
                    border-radius: 8px;
                    padding: 5px;
                    margin-top: 20px;
                    border: 1px solid #333333;
                }
                
                /* Styling for text area inside the container */
                .whatsapp-input textarea {
                    flex-grow: 1;
                    background-color: transparent !important;
                    border: none !important;
                    color: white !important;
                    resize: none;
                    padding: 10px;
                    min-height: 50px;
                }
                
                /* Styling for the mic button */
                .mic-button {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 5px;
                    cursor: pointer;
                }
                
                /* Mic icon */
                .mic-icon {
                    font-size: 20px;
                }
                
                /* Submit button styling */
                .stButton > button {
                    background-color: #3498db !important;
                    color: white !important;
                    border: none !important;
                }
                
                /* Remove focus outline */
                textarea:focus {
                    outline: none !important;
                    box-shadow: none !important;
                }
                </style>
                
                <p style="color: #3498db; font-size: 14px; margin-bottom: 5px;">Your answer:</p>
                """, unsafe_allow_html=True)
                
                # Creating columns for the input and recording status
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    text_answer = st.text_area("", 
                                              height=100, 
                                              key="text_answer_field", 
                                              placeholder="Type your answer or click the mic to record...")
                
                with col2:
                    # Using a streamlit button styled to look like a mic button
                    st.markdown("""
                    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                        <div id="mic-button-container"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    is_recording = st.button("🎤", key="record_button")
                
                # Handle recording state
                if is_recording:
                    st.session_state.is_recording = True
                    st.markdown("""
                    <p style="color: #3498db; text-align: center; margin-top: 5px; font-size: 12px;">
                        Recording... (speak now)
                    </p>
                    """, unsafe_allow_html=True)
                    
                    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=16000)
                    
                    if audio_bytes:
                        st.session_state.audio_data = audio_bytes
                        st.markdown("""
                        <p style="margin-top: 8px; font-size: 13px; color: #cccccc; text-align: center;">
                            Processing audio...
                        </p>
                        """, unsafe_allow_html=True)

                        with st.spinner("Transcribing..."):
                            transcript = transcribe_audio(audio_bytes)
                            st.session_state.transcription = transcript
                            st.rerun()
                
                # Submit button
                if st.button("Submit Answer", type="primary"):
                    if text_answer.strip():
                        st.session_state.transcription = text_answer
                        st.rerun()
                    else:
                        st.error("Please provide an answer before submitting.")

            # Feedback section with minimal styling
            if st.session_state.show_feedback:
                st.markdown("""
                <p style="color: #3498db; font-size: 16px; margin-top: 25px; margin-bottom: 10px;">Feedback on Your Answer</p>
                """, unsafe_allow_html=True)
                
                with st.spinner("Generating feedback..."):
                    if not st.session_state.feedbacks[st.session_state.current_question_idx]:
                        feedback = get_answer_feedback(current_question, st.session_state.transcription)
                        st.session_state.feedbacks[st.session_state.current_question_idx] = feedback
                    else:
                        feedback = st.session_state.feedbacks[st.session_state.current_question_idx]
                    
                    st.markdown(f"""
                    <div style="background-color: #1E1E1E; padding: 15px; border-radius: 4px; border: 1px solid #3498db;">
                        {feedback}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("Continue to Next Question", type="primary"):
                        st.session_state.answers[st.session_state.current_question_idx] = st.session_state.transcription
                        
                        st.session_state.current_question_idx += 1
                        st.session_state.transcription = ""
                        st.session_state.audio_data = None
                        st.session_state.show_feedback = False
                        st.session_state.question_spoken = False
                        
                        if st.session_state.current_question_idx >= len(st.session_state.questions):
                            st.session_state.interview_complete = True
                        
                        st.rerun()

# If we got to this point without displaying a page, something went wrong
else:
    st.error("An error occurred in the application flow. Please restart the application.")
    if st.button("Reset Application"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state.setup_stage = "welcome_page"
        st.rerun()