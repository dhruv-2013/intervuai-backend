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
import re
import PyPDF2
import docx
from io import BytesIO
# Import the answer evaluation module
from answer_evaluation import get_answer_evaluation, save_evaluation_data, calculate_aggregate_scores, aggregate_skill_assessment, generate_career_insights

# The page config MUST be the first Streamlit command used in your app
st.set_page_config(
    page_title="Interview Agent",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize TTS and Firebase credentials with safer error handling
tts_client = None
firebase_credentials = None

# Safely load TTS credentials
try:
    # Parse the JSON credentials properly
    tts_credentials_info = json.loads(st.secrets["GOOGLE_TTS_CREDENTIALS_JSON"])
    tts_credentials = service_account.Credentials.from_service_account_info(tts_credentials_info)
    tts_client = texttospeech.TextToSpeechClient(credentials=tts_credentials)
except Exception as e:
    st.error(f"Error initializing Google TTS client: {e}")

# Safely load Firebase credentials
try:
    # Parse the JSON credentials properly
    firebase_credentials_info = json.loads(st.secrets["FIREBASE_CREDENTIALS_JSON"])
    firebase_credentials = service_account.Credentials.from_service_account_info(firebase_credentials_info)
except Exception as e:
    st.error(f"Error initializing Firebase credentials: {e}")
    
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
    st.session_state.setup_stage = "welcome_page"
if "question_spoken" not in st.session_state:
    st.session_state.question_spoken = False
if "use_voice" not in st.session_state:
    st.session_state.use_voice = True
if "interviewer_name" not in st.session_state:
    st.session_state.interviewer_name = ""
if "voice_type" not in st.session_state:
    st.session_state.voice_type = "en-US-Neural2-D"
if "interview_stage" not in st.session_state:
    st.session_state.interview_stage = "introduction"
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []
if "chatbot_mode" not in st.session_state:
    st.session_state.chatbot_mode = False
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "personalized_questions" not in st.session_state:
    st.session_state.personalized_questions = []
if "career_recommendations" not in st.session_state:
    st.session_state.career_recommendations = []
if "manual_input" not in st.session_state:
    st.session_state.manual_input = False

@st.cache_resource
def load_whisper_model():
    model_size = "small" if st.session_state.get("faster_transcription", True) else "medium"
    device = "cuda" if st.session_state.get("use_gpu", False) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)

# Get the TTS client with proper caching
@st.cache_resource
def get_tts_client():
    global tts_client
    if tts_client:
        return tts_client
    
    try:
        # If not initialized yet, try to load from secrets
        tts_creds_raw = st.secrets["GOOGLE_TTS_CREDENTIALS_JSON"]
        tts_credentials_info = json.loads(tts_creds_raw)
        tts_credentials = service_account.Credentials.from_service_account_info(tts_credentials_info)
        tts_client = texttospeech.TextToSpeechClient(credentials=tts_credentials)
        return tts_client
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

# Get Firebase credentials for save_evaluation_data function
def get_firebase_credentials():
    global firebase_credentials
    if firebase_credentials:
        return firebase_credentials
    
    try:
        # If not loaded yet, try to load from secrets
        firebase_creds_raw = st.secrets["FIREBASE_CREDENTIALS_JSON"]
        firebase_credentials_info = json.loads(firebase_creds_raw)
        firebase_credentials = service_account.Credentials.from_service_account_info(firebase_credentials_info)
        return firebase_credentials
    except Exception as e:
        st.error(f"Error loading Firebase credentials: {str(e)}")
        return None

# The rest of your code follows...
 
# Enhanced Question Bank for Interview Agent
# Replace the existing JOB_FIELDS dictionary in your code with this expanded version

JOB_FIELDS = {
    "Software Engineering": {
        "Technical": [
            "Explain the difference between arrays and linked lists.",
            "What's your approach to debugging a complex issue?",
            "Describe a challenging technical problem you solved recently.",
            "Explain the concept of time and space complexity.",
            "What design patterns have you used in your projects?",
            "How do you ensure your code is maintainable and scalable?",
            "Explain how you would implement error handling in a distributed system.",
            "What's the difference between SQL and NoSQL databases?",
            "How would you optimize a slow database query?",
            "Explain the concept of RESTful APIs and their principles.",
            "What's the difference between synchronous and asynchronous programming?",
            "How do you handle memory management in your preferred language?",
            "Explain the concept of dependency injection.",
            "What's the difference between unit testing and integration testing?",
            "How would you implement caching in a web application?",
            "Explain the CAP theorem and its implications.",
            "What's the difference between horizontal and vertical scaling?",
            "How do you ensure thread safety in concurrent programming?",
            "Explain the concept of microservices architecture.",
            "What's your approach to code versioning and branching strategies?",
            "How would you design a URL shortener like bit.ly?",
            "Explain the difference between authentication and authorization.",
            "What's the purpose of containerization and how have you used it?",
            "How do you handle API rate limiting?",
            "Explain the concept of load balancing.",
            "What's your experience with message queues?",
            "How would you implement real-time features in a web application?",
            "Explain the concept of database normalization.",
            "What's the difference between stateful and stateless applications?",
            "How do you approach performance profiling and optimization?"
        ],
        "Behavioral": [
            "Tell me about a time you had to work under pressure to meet a deadline.",
            "Describe a situation where you disagreed with a team member on a technical approach.",
            "How do you handle feedback on your code during code reviews?",
            "Tell me about a time you identified and fixed a bug that others couldn't solve.",
            "How do you keep up with the latest technologies and programming languages?",
            "Describe a time when you had to learn a new technology quickly.",
            "Tell me about a project that didn't go as planned and how you handled it.",
            "How do you prioritize tasks when working on multiple projects?",
            "Describe a time when you had to explain a complex technical concept to a non-technical person.",
            "Tell me about a time you received constructive criticism on your work.",
            "How do you handle working with legacy code?",
            "Describe a situation where you had to make a difficult technical decision.",
            "Tell me about a time you mentored a junior developer.",
            "How do you approach working in a team with different skill levels?",
            "Describe a time when you had to refactor a large codebase.",
            "Tell me about a mistake you made in your code and how you learned from it.",
            "How do you handle tight deadlines without compromising code quality?",
            "Describe a time when you had to work with a difficult stakeholder.",
            "Tell me about a time you improved team productivity or processes.",
            "How do you stay motivated during long, challenging projects?"
        ],
        "Role-specific": [
            "How do you approach testing your code?",
            "Describe your experience with CI/CD pipelines.",
            "How do you balance technical debt with delivering features?",
            "Explain your approach to optimizing application performance.",
            "How would you explain a complex technical concept to a non-technical stakeholder?",
            "What's your experience with agile development methodologies?",
            "How do you ensure security best practices in your code?",
            "Describe your approach to documentation and knowledge sharing.",
            "How do you handle production incidents and post-mortems?",
            "What's your experience with cloud platforms and services?",
            "How do you approach capacity planning for applications?",
            "Describe your experience with monitoring and observability tools.",
            "How do you handle database migrations and schema changes?",
            "What's your approach to API design and versioning?",
            "How do you ensure accessibility in web applications?",
            "Describe your experience with mobile development considerations.",
            "How do you approach internationalization and localization?",
            "What's your experience with DevOps practices?",
            "How do you handle cross-browser compatibility issues?",
            "Describe your approach to technical leadership and decision-making."
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
            "Explain the difference between correlation and causation with an example.",
            "What's the difference between precision and recall?",
            "How do you handle imbalanced datasets?",
            "Explain the bias-variance tradeoff.",
            "What's your approach to cross-validation?",
            "How do you evaluate the performance of a regression model?",
            "Explain the concept of regularization in machine learning.",
            "What's the difference between bagging and boosting?",
            "How do you handle categorical variables in machine learning?",
            "Explain the concept of dimensionality reduction.",
            "What's your experience with time series analysis?",
            "How do you approach outlier detection and treatment?",
            "Explain the concept of ensemble methods.",
            "What's the difference between parametric and non-parametric models?",
            "How do you handle data leakage in machine learning?",
            "Explain the concept of feature engineering.",
            "What's your approach to model selection and hyperparameter tuning?",
            "How do you handle multi-collinearity in regression models?",
            "Explain the concept of clustering and its applications.",
            "What's your experience with deep learning frameworks?",
            "How do you approach natural language processing tasks?",
            "Explain the concept of recommendation systems.",
            "What's your experience with big data technologies?",
            "How do you approach experiment design and A/B testing?"
        ],
        "Behavioral": [
            "Tell me about a time when your data analysis led to a significant business decision.",
            "How do you communicate complex data insights to non-technical stakeholders?",
            "Describe a situation where you had to defend your analytical approach.",
            "Tell me about a project where you had to work with messy or incomplete data.",
            "How do you ensure your analysis is accurate and reliable?",
            "Describe a time when your initial hypothesis was proven wrong by the data.",
            "Tell me about a challenging data problem you solved creatively.",
            "How do you handle conflicting requirements from different stakeholders?",
            "Describe a time when you had to work under tight deadlines on a data project.",
            "Tell me about a time you had to learn a new analytical tool or technique quickly.",
            "How do you approach working with domain experts who aren't data-savvy?",
            "Describe a situation where you found an unexpected pattern in data.",
            "Tell me about a time you had to present negative or disappointing results.",
            "How do you handle situations where data quality is poor?",
            "Describe a time when you had to balance speed vs. accuracy in analysis.",
            "Tell me about a project where you had to collaborate with multiple teams.",
            "How do you stay current with new developments in data science?",
            "Describe a time when you had to question the data collection process.",
            "Tell me about a mistake you made in analysis and how you corrected it.",
            "How do you approach ethical considerations in data science?"
        ],
        "Role-specific": [
            "What visualization tools do you prefer and why?",
            "How do you determine which statistical test to use for a given problem?",
            "Describe your approach to A/B testing.",
            "How do you translate business questions into data queries?",
            "What metrics would you track to measure the success of a product feature?",
            "How do you approach data governance and privacy considerations?",
            "Describe your experience with cloud-based analytics platforms.",
            "How do you handle version control for data science projects?",
            "What's your approach to model deployment and monitoring?",
            "How do you ensure reproducibility in your analysis?",
            "Describe your experience with real-time data processing.",
            "How do you approach feature stores and ML operations?",
            "What's your experience with automated machine learning tools?",
            "How do you handle data pipeline failures and monitoring?",
            "Describe your approach to data storytelling and presentation.",
            "How do you work with data engineers and other technical teams?",
            "What's your experience with customer segmentation and targeting?",
            "How do you approach predictive modeling for business outcomes?",
            "Describe your experience with dashboard design and KPI tracking.",
            "How do you validate and test machine learning models in production?"
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
            "Explain how you would manage scope creep.",
            "What's your experience with agile project management?",
            "How do you approach project budgeting and cost control?",
            "Describe your experience with waterfall methodology.",
            "How do you handle dependencies between different project tasks?",
            "What's your approach to quality assurance in projects?",
            "How do you manage project documentation and knowledge transfer?",
            "Describe your experience with project portfolio management.",
            "How do you approach change management in projects?",
            "What's your experience with remote project team management?",
            "How do you handle project communication and reporting?",
            "Describe your approach to vendor and contractor management.",
            "How do you ensure project deliverables meet requirements?",
            "What's your experience with project management software and tools?",
            "How do you approach project closure and lessons learned?",
            "Describe your experience with cross-functional project teams.",
            "How do you handle project governance and compliance requirements?",
            "What's your approach to project estimation and planning?",
            "How do you manage project integration and coordination?"
        ],
        "Behavioral": [
            "Tell me about a time when a project was falling behind schedule.",
            "Describe how you've managed stakeholder expectations.",
            "How do you motivate team members during challenging phases of a project?",
            "Tell me about a project that failed and what you learned from it.",
            "How do you handle conflicts between team members or departments?",
            "Describe a time when you had to make a difficult decision under pressure.",
            "Tell me about a situation where project requirements changed significantly.",
            "How do you handle team members who are not meeting expectations?",
            "Describe a time when you had to manage a project with limited resources.",
            "Tell me about a time you had to communicate bad news to stakeholders.",
            "How do you handle working with difficult or unresponsive team members?",
            "Describe a situation where you had to negotiate with vendors or contractors.",
            "Tell me about a time you had to manage multiple competing priorities.",
            "How do you handle situations where stakeholders have conflicting requirements?",
            "Describe a time when you successfully turned around a failing project.",
            "Tell me about a situation where you had to work with a tight deadline.",
            "How do you approach building relationships with new team members?",
            "Describe a time when you had to present project results to senior management.",
            "Tell me about a situation where you had to adapt your management style.",
            "How do you handle stress and maintain team morale during difficult projects?"
        ],
        "Role-specific": [
            "How do you prioritize competing deadlines across multiple projects?",
            "Describe how you communicate project status to different audiences.",
            "How do you ensure quality deliverables while maintaining timelines?",
            "What's your approach to gathering requirements from stakeholders?",
            "How do you manage project budgets and resources?",
            "Describe your experience with project risk assessment and mitigation.",
            "How do you handle project team development and training?",
            "What's your approach to project metrics and KPI tracking?",
            "How do you manage project scope and prevent scope creep?",
            "Describe your experience with client or customer-facing projects.",
            "How do you approach project retrospectives and continuous improvement?",
            "What's your experience with regulatory or compliance-driven projects?",
            "How do you handle project escalation and issue resolution?",
            "Describe your approach to project resource planning and allocation.",
            "How do you manage project timelines when working with external dependencies?",
            "What's your experience with digital transformation or technology projects?",
            "How do you approach stakeholder analysis and engagement planning?",
            "Describe your experience with project procurement and contract management.",
            "How do you ensure effective knowledge transfer at project completion?",
            "What's your approach to managing project risks and assumptions?"
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
            "How do you use data to inform design decisions?",
            "What's your experience with usability testing?",
            "How do you approach information architecture?",
            "Describe your experience with interaction design.",
            "How do you ensure consistency across different platforms?",
            "What's your approach to mobile-first design?",
            "How do you handle design for different screen sizes and devices?",
            "Describe your experience with design thinking methodology.",
            "How do you approach color theory and typography in your designs?",
            "What's your experience with motion design and micro-interactions?",
            "How do you conduct competitive analysis for design projects?",
            "Describe your approach to creating user journey maps.",
            "How do you handle design handoff to developers?",
            "What's your experience with A/B testing for design decisions?",
            "How do you approach designing for accessibility and inclusion?",
            "Describe your experience with design research and validation.",
            "How do you create and maintain design documentation?",
            "What's your approach to cross-browser and cross-platform compatibility?",
            "How do you handle design feedback and iteration cycles?"
        ],
        "Behavioral": [
            "Tell me about a time when you received difficult feedback on your design.",
            "Describe a situation where you had to compromise on a design decision.",
            "How do you advocate for the user when there are business constraints?",
            "Tell me about a design challenge you faced and how you overcame it.",
            "How do you collaborate with developers to implement your designs?",
            "Describe a time when user research contradicted your initial design assumptions.",
            "Tell me about a project where you had to work with tight deadlines.",
            "How do you handle conflicting feedback from different stakeholders?",
            "Describe a situation where you had to design for a user group you weren't familiar with.",
            "Tell me about a time you had to defend your design decisions.",
            "How do you approach working with stakeholders who don't understand UX?",
            "Describe a time when you had to pivot your design approach mid-project.",
            "Tell me about a situation where technical constraints limited your design options.",
            "How do you handle situations where business goals conflict with user needs?",
            "Describe a time when you successfully influenced a product decision through design.",
            "Tell me about a project where you had to work with limited resources.",
            "How do you approach learning about new user groups or industries?",
            "Describe a time when you had to present your design to senior executives.",
            "Tell me about a situation where you had to work with an existing design system.",
            "How do you handle criticism of your design work?"
        ],
        "Role-specific": [
            "How do you measure the success of a design?",
            "Describe how you stay current with design trends and best practices.",
            "How do you balance aesthetics with usability?",
            "Explain your approach to responsive design.",
            "How would you improve the user experience of our product?",
            "What's your experience with design systems and component libraries?",
            "How do you approach user onboarding and first-time user experiences?",
            "Describe your experience with e-commerce or conversion-focused design.",
            "How do you handle designing for different user personas and use cases?",
            "What's your approach to creating design specifications and guidelines?",
            "How do you collaborate with product managers and stakeholders?",
            "Describe your experience with design workshops and facilitation.",
            "How do you approach designing for international or multicultural audiences?",
            "What's your experience with voice user interfaces or emerging technologies?",
            "How do you handle design version control and collaboration?",
            "Describe your approach to creating design presentations and storytelling.",
            "How do you ensure your designs align with brand guidelines?",
            "What's your experience with design leadership and mentoring?",
            "How do you approach designing for different business models?",
            "Describe your experience with design operations and process improvement."
        ]
    },
    "IT Support": {
        "Technical": [
            "Explain the difference between hardware and software troubleshooting.",
            "How would you approach a user who can't connect to the internet?",
            "Describe your experience with ticketing systems.",
            "What steps would you take to secure a workstation?",
            "How do you prioritize multiple support requests?",
            "Explain how you would troubleshoot a slow computer.",
            "What's your experience with network troubleshooting?",
            "How do you approach mobile device support and management?",
            "Describe your experience with Active Directory and user management.",
            "How would you troubleshoot email connectivity issues?",
            "What's your approach to software installation and deployment?",
            "How do you handle printer and peripheral device issues?",
            "Describe your experience with backup and recovery procedures.",
            "How would you troubleshoot VPN connectivity problems?",
            "What's your experience with cloud services support?",
            "How do you approach virus and malware removal?",
            "Describe your experience with remote desktop and support tools.",
            "How would you handle a server outage or critical system failure?",
            "What's your approach to password management and security?",
            "How do you troubleshoot audio and video conferencing issues?",
            "Describe your experience with software licensing and compliance.",
            "How would you approach migrating user data to a new system?",
            "What's your experience with mobile device management (MDM)?",
            "How do you handle browser and web application issues?",
            "Describe your approach to monitoring system performance and health."
        ],
        "Behavioral": [
            "Tell me about a time when you had to explain a technical issue to a non-technical user.",
            "Describe a situation where you went above and beyond for a user.",
            "How do you handle frustrated or angry users?",
            "Tell me about a time when you couldn't solve a technical problem immediately.",
            "How do you stay patient when dealing with repetitive support issues?",
            "Describe a time when you had to work under pressure to resolve a critical issue.",
            "Tell me about a situation where you had to learn a new technology quickly.",
            "How do you handle multiple urgent requests at the same time?",
            "Describe a time when you had to escalate an issue to a higher level.",
            "Tell me about a situation where you prevented a major issue from occurring.",
            "How do you approach working with users who resist technology changes?",
            "Describe a time when you had to work with a difficult colleague or vendor.",
            "Tell me about a situation where you improved a support process.",
            "How do you handle situations where you don't know the answer immediately?",
            "Describe a time when you had to work overtime to resolve an issue.",
            "Tell me about a situation where you had to communicate bad news to users.",
            "How do you approach building rapport with new users or departments?",
            "Describe a time when you had to train someone on a new system.",
            "Tell me about a situation where you had to work independently without supervision.",
            "How do you maintain your composure during high-stress situations?"
        ],
        "Role-specific": [
            "What remote support tools are you familiar with?",
            "How do you document your troubleshooting steps?",
            "Describe your approach to user training and education.",
            "How do you keep up with new technologies and support techniques?",
            "What's your experience with supporting remote workers?",
            "How do you approach preventive maintenance and system monitoring?",
            "Describe your experience with help desk metrics and SLA management.",
            "How do you handle escalation procedures and communication?",
            "What's your approach to knowledge base creation and maintenance?",
            "How do you ensure data security while providing support?",
            "Describe your experience with asset management and inventory tracking.",
            "How do you approach vendor relationships and support coordination?",
            "What's your experience with change management and communication?",
            "How do you handle emergency response and business continuity?",
            "Describe your approach to user account provisioning and deprovisioning.",
            "How do you ensure compliance with IT policies and procedures?",
            "What's your experience with budget planning for IT support?",
            "How do you approach cross-training and knowledge sharing?",
            "Describe your experience with project work and implementations.",
            "How do you measure and improve customer satisfaction in support?"
        ]
    },
    "Cybersecurity": {
        "Technical": [
            "Explain the concept of defense in depth.",
            "What's the difference between authentication and authorization?",
            "How would you respond to a potential data breach?",
            "Describe common network vulnerabilities and how to mitigate them.",
            "What's your approach to vulnerability assessment?",
            "Explain the importance of patch management.",
            "How do you approach security incident response?",
            "What's your experience with penetration testing?",
            "Describe the CIA triad and its importance in security.",
            "How would you implement a zero-trust security model?",
            "What's your approach to security monitoring and SIEM tools?",
            "Explain the concept of threat modeling.",
            "How do you approach cloud security and configuration?",
            "What's your experience with identity and access management?",
            "Describe your approach to network segmentation and firewalls.",
            "How would you secure a remote workforce?",
            "What's your experience with encryption and key management?",
            "Explain the concept of security by design.",
            "How do you approach mobile device security?",
            "What's your experience with compliance frameworks (SOX, HIPAA, etc.)?",
            "Describe your approach to security risk assessment.",
            "How would you handle a ransomware attack?",
            "What's your experience with security automation and orchestration?",
            "Explain the concept of threat intelligence and its applications.",
            "How do you approach secure software development practices?"
        ],
        "Behavioral": [
            "Tell me about a time when you identified a security risk before it became an issue.",
            "How do you balance security needs with user convenience?",
            "Describe a situation where you had to convince management to invest in security measures.",
            "How do you stay current with evolving security threats?",
            "Tell me about a time when you had to respond to a security incident.",
            "Describe a situation where you had to work under pressure during a security crisis.",
            "How do you approach educating non-technical staff about security?",
            "Tell me about a time when you had to implement unpopular security policies.",
            "Describe a situation where you discovered a security vulnerability.",
            "How do you handle situations where security and business objectives conflict?",
            "Tell me about a time you had to coordinate with law enforcement or external agencies.",
            "Describe a situation where you had to learn about a new threat quickly.",
            "How do you approach building security awareness across an organization?",
            "Tell me about a time when you had to present security metrics to leadership.",
            "Describe a situation where you had to work with a third-party security vendor.",
            "How do you handle the stress of constant vigilance required in security?",
            "Tell me about a time when you had to update security policies or procedures.",
            "Describe a situation where you had to investigate a potential insider threat.",
            "How do you approach collaboration with other IT teams on security matters?",
            "Tell me about a time when you had to make a quick security decision."
        ],
        "Role-specific": [
            "What security tools and technologies are you experienced with?",
            "How would you implement a security awareness program?",
            "Describe your experience with compliance requirements (GDPR, HIPAA, etc.)",
            "What's your approach to security logging and monitoring?",
            "How would you conduct a security audit?",
            "Describe your experience with digital forensics and incident investigation.",
            "How do you approach security architecture and design reviews?",
            "What's your experience with business continuity and disaster recovery planning?",
            "How do you handle security vendor evaluation and management?",
            "Describe your approach to security metrics and reporting.",
            "How would you develop and test an incident response plan?",
            "What's your experience with security policy development and governance?",
            "How do you approach threat hunting and proactive security measures?",
            "Describe your experience with security training and certification programs.",
            "How would you secure cloud infrastructure and services?",
            "What's your approach to managing security across multiple locations?",
            "How do you handle security aspects of mergers and acquisitions?",
            "Describe your experience with security budget planning and justification.",
            "How would you approach implementing new security technologies?",
            "What's your experience with coordinating security across different business units?"
        ]
    
    }
}

COMMON_QUESTIONS = {
    "Background": [
        "Tell me more about yourself and why you're interested in this field.",
        "What interests you most about this role and our company?",
        "Walk me through your career journey and key achievements.",
        "What are your long-term career goals?",
        "How did you become interested in this field?",
        "What do you consider your greatest professional accomplishment?",
        "Tell me about a challenge that shaped your career direction.",
        "What motivates you in your work?",
        "How do you stay current with industry trends and developments?",
        "What unique perspective or experience do you bring to this role?"
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
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def analyze_resume_with_ai(resume_text):
    """Analyze resume using OpenAI to extract key information"""
    prompt = f"""
    Analyze the following resume and extract key information in JSON format:
    
    Resume Text:
    {resume_text}
    
    Please extract and return a JSON object with the following structure:
    {{
        "name": "candidate name",
        "email": "email address", 
        "phone": "phone number",
        "current_role": "current job title",
        "experience_years": "estimated years of experience",
        "skills": ["list", "of", "technical", "skills"],
        "education": ["degree", "institution"],
        "industries": ["list", "of", "industries", "worked", "in"],
        "job_roles": ["list", "of", "previous", "job", "titles"],
        "achievements": ["key", "achievements", "and", "accomplishments"],
        "technologies": ["programming", "languages", "tools", "platforms"],
        "certifications": ["professional", "certifications"],
        "recommended_fields": ["suggested", "job", "fields", "based", "on", "experience"],
        "strengths": ["identified", "strengths", "from", "resume"],
        "areas_for_improvement": ["potential", "skill", "gaps", "or", "areas", "to", "develop"]
    }}
    
    Only return the JSON object, no additional text.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        analysis_text = response.choices[0].message.content.strip()
        analysis_text = analysis_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(analysis_text)
    except Exception as e:
        st.error(f"Error analyzing resume: {str(e)}")
        return {}

def generate_personalized_questions(resume_analysis, target_field=None):
    """Generate personalized interview questions based on resume analysis"""
    if not target_field and resume_analysis.get("recommended_fields"):
        target_field = resume_analysis["recommended_fields"][0]
    
    if not target_field:
        target_field = "General"
    
    # Get base questions from the field
    base_questions = []
    if target_field in JOB_FIELDS:
        for category, questions in JOB_FIELDS[target_field].items():
            base_questions.extend([{"category": category, "question": q} for q in questions[:3]])
    
    # Generate personalized questions using AI
    prompt = f"""
    Based on the following resume analysis, generate 10 personalized interview questions for a {target_field} role:
    
    Resume Analysis:
    - Current Role: {resume_analysis.get('current_role', 'N/A')}
    - Experience: {resume_analysis.get('experience_years', 'N/A')} years
    - Skills: {', '.join(resume_analysis.get('skills', []))}
    - Technologies: {', '.join(resume_analysis.get('technologies', []))}
    - Industries: {', '.join(resume_analysis.get('industries', []))}
    - Achievements: {', '.join(resume_analysis.get('achievements', []))}
    
    Generate questions that:
    1. Reference specific skills/experience from their resume
    2. Are appropriate for their experience level
    3. Focus on their demonstrated strengths
    4. Address any gaps or areas for improvement
    5. Are relevant to the {target_field} field
    
    Return as a JSON array of objects with "category" and "question" fields:
    [
        {{"category": "Experience-based", "question": "specific question"}},
        ...
    ]
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        questions_text = response.choices[0].message.content.strip()
        questions_text = questions_text.replace("```json", "").replace("```", "").strip()
        personalized_questions = json.loads(questions_text)
        
        # Combine with some base questions
        all_questions = base_questions[:5] + personalized_questions[:10]
        return all_questions
        
    except Exception as e:
        st.error(f"Error generating personalized questions: {str(e)}")
        return base_questions[:10]

def generate_career_recommendations(resume_analysis):
    """Generate career recommendations based on resume analysis"""
    prompt = f"""
    Based on the following resume analysis, provide career recommendations and insights:
    
    Resume Analysis:
    - Current Role: {resume_analysis.get('current_role', 'N/A')}
    - Experience: {resume_analysis.get('experience_years', 'N/A')} years
    - Skills: {', '.join(resume_analysis.get('skills', []))}
    - Technologies: {', '.join(resume_analysis.get('technologies', []))}
    - Industries: {', '.join(resume_analysis.get('industries', []))}
    - Strengths: {', '.join(resume_analysis.get('strengths', []))}
    - Areas for Improvement: {', '.join(resume_analysis.get('areas_for_improvement', []))}
    
    Provide recommendations in JSON format:
    {{
        "suitable_roles": ["list of 5 specific job titles they'd be good for"],
        "growth_opportunities": ["potential career advancement paths"],
        "skill_recommendations": ["skills to develop for career growth"],
        "industry_insights": ["relevant industry trends and opportunities"],
        "salary_range": "estimated salary range for their experience level",
        "next_steps": ["actionable steps to advance their career"],
        "interview_focus_areas": ["areas to emphasize in interviews"]
    }}
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        recommendations_text = response.choices[0].message.content.strip()
        recommendations_text = recommendations_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(recommendations_text)
    except Exception as e:
        st.error(f"Error generating career recommendations: {str(e)}")
        return {}

def chatbot_response(user_message, resume_analysis):
    """Generate chatbot response based on user message and resume analysis"""
    context = f"""
    You are a career counselor and interview coach. You have access to the user's resume analysis:
    
    Resume Summary:
    - Name: {resume_analysis.get('name', 'User')}
    - Current Role: {resume_analysis.get('current_role', 'N/A')}
    - Experience: {resume_analysis.get('experience_years', 'N/A')} years
    - Skills: {', '.join(resume_analysis.get('skills', []))}
    - Recommended Fields: {', '.join(resume_analysis.get('recommended_fields', []))}
    - Strengths: {', '.join(resume_analysis.get('strengths', []))}
    
    Chat History:
    {chr(10).join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-5:]])}
    
    User Message: {user_message}
    
    Provide helpful, personalized career advice based on their background. Be conversational and supportive.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": context}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your request right now. Please try again later."

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
        
        # ADD THIS NEW BUTTON:
        if st.button("📄 Resume Analysis & Career Coach", key="resume_chatbot_button", use_container_width=True):
            st.session_state.setup_stage = "resume_chatbot"
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
elif st.session_state.setup_stage == "resume_chatbot":
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #333;
        border-radius: 5px;
        background-color: #1E1E1E;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #3498db;
        color: white;
        padding: 8px 12px;
        border-radius: 15px;
        margin: 5px 0;
        margin-left: 50px;
        text-align: right;
    }
    .bot-message {
        background-color: #2C2C2C;
        color: white;
        padding: 8px 12px;
        border-radius: 15px;
        margin: 5px 0;
        margin-right: 50px;
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 48px; color: white;">Resume Analysis & Career Coach</h1>
        <p style="font-size: 18px; color: #cccccc;">Upload your resume for personalized interview questions and career recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    if not st.session_state.resume_text:
        st.markdown("### 📄 Upload Your Resume")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose your resume file",
                type=['pdf', 'docx', 'txt'],
                help="Upload your resume in PDF, DOCX, or TXT format"
            )
            
            if uploaded_file is not None:
                # Extract text based on file type
                file_type = uploaded_file.type
                
                with st.spinner("Processing your resume..."):
                    if file_type == "application/pdf":
                        resume_text = extract_text_from_pdf(uploaded_file)
                    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        resume_text = extract_text_from_docx(uploaded_file)
                    else:  # txt file
                        resume_text = str(uploaded_file.read(), "utf-8")
                    
                    st.session_state.resume_text = resume_text
                    
                    # Analyze resume with AI
                    st.session_state.resume_analysis = analyze_resume_with_ai(resume_text)
                    
                    # Generate initial recommendations
                    st.session_state.career_recommendations = generate_career_recommendations(st.session_state.resume_analysis)
                    
                    st.success("Resume processed successfully!")
                    st.rerun()
        
        with col2:
            st.markdown("### Alternative Input")
            if st.button("Paste Resume Text", type="secondary"):
                st.session_state.manual_input = True
        
        # Manual text input option
        if st.session_state.get("manual_input", False):
            manual_text = st.text_area("Paste your resume text here:", height=200)
            if st.button("Analyze Resume") and manual_text:
                with st.spinner("Analyzing your resume..."):
                    st.session_state.resume_text = manual_text
                    st.session_state.resume_analysis = analyze_resume_with_ai(manual_text)
                    st.session_state.career_recommendations = generate_career_recommendations(st.session_state.resume_analysis)
                    st.success("Resume analyzed successfully!")
                    st.rerun()
    
    # Resume analysis and chat interface
    else:
        # Display resume summary
        if st.session_state.resume_analysis:
            with st.expander("📋 Resume Summary", expanded=False):
                analysis = st.session_state.resume_analysis
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {analysis.get('name', 'Not specified')}")
                    st.write(f"**Current Role:** {analysis.get('current_role', 'Not specified')}")
                    st.write(f"**Experience:** {analysis.get('experience_years', 'Not specified')} years")
                    st.write(f"**Skills:** {', '.join(analysis.get('skills', [])[:5])}")
                
                with col2:
                    st.write(f"**Recommended Fields:** {', '.join(analysis.get('recommended_fields', []))}")
                    st.write(f"**Technologies:** {', '.join(analysis.get('technologies', [])[:5])}")
                    st.write(f"**Industries:** {', '.join(analysis.get('industries', []))}")
        
        # Career recommendations
        if st.session_state.career_recommendations:
            with st.expander("🎯 Career Recommendations", expanded=False):
                recs = st.session_state.career_recommendations
                
                st.markdown("**Suitable Roles:**")
                for role in recs.get('suitable_roles', []):
                    st.write(f"• {role}")
                
                st.markdown("**Skills to Develop:**")
                for skill in recs.get('skill_recommendations', []):
                    st.write(f"• {skill}")
                
                st.markdown("**Next Steps:**")
                for step in recs.get('next_steps', []):
                    st.write(f"• {step}")
        
        # Quick action buttons
        st.markdown("### 🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Generate Interview Questions"):
                if st.session_state.resume_analysis.get('recommended_fields'):
                    target_field = st.session_state.resume_analysis['recommended_fields'][0]
                    st.session_state.personalized_questions = generate_personalized_questions(
                        st.session_state.resume_analysis, target_field
                    )
                    st.success(f"Generated {len(st.session_state.personalized_questions)} personalized questions!")
        
        with col2:
            if st.button("Start Practice Interview"):
                if st.session_state.personalized_questions:
                    st.session_state.questions = st.session_state.personalized_questions
                    st.session_state.selected_job_field = st.session_state.resume_analysis.get('recommended_fields', ['General'])[0]
                    st.session_state.current_question_idx = 0
                    st.session_state.answers = [""] * len(st.session_state.questions)
                    st.session_state.feedbacks = [""] * len(st.session_state.questions)
                    st.session_state.setup_stage = "interview"
                    st.session_state.interview_stage = "introduction"
                    st.rerun()
                else:
                    st.error("Please generate interview questions first!")
        
        with col3:
            if st.button("Salary Insights"):
                if st.session_state.career_recommendations.get('salary_range'):
                    st.info(f"Estimated salary range: {st.session_state.career_recommendations['salary_range']}")
                else:
                    st.info("Salary information not available")
        
        with col4:
            if st.button("Upload New Resume"):
                # Reset all resume-related session state
                st.session_state.resume_text = ""
                st.session_state.resume_analysis = {}
                st.session_state.career_recommendations = []
                st.session_state.chat_history = []
                st.session_state.personalized_questions = []
                st.rerun()
        
        # Display personalized questions if generated
        if st.session_state.personalized_questions:
            with st.expander("🎯 Your Personalized Interview Questions", expanded=False):
                for i, q_data in enumerate(st.session_state.personalized_questions, 1):
                    st.write(f"**{i}. ({q_data['category']})** {q_data['question']}")
        
        # Chat interface
        st.markdown("### 💬 Career Coach Chat")
        st.markdown("Ask me anything about your career, interview preparation, or professional development!")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Type your message:", placeholder="e.g., How can I improve my chances for a data analyst role?")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Send") and user_input:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Generate bot response
                bot_response = chatbot_response(user_input, st.session_state.resume_analysis)
                st.session_state.chat_history.append({"role": "bot", "content": bot_response})
                
                st.rerun()
        
        # Suggested questions for first-time users
        if not st.session_state.chat_history:
            st.markdown("**💡 Try asking:**")
            suggestions = [
                "What roles am I best suited for based on my experience?",
                "How can I improve my resume for my target role?",
                "What skills should I focus on developing?",
                "What salary range should I expect?",
                "How can I prepare for interviews in my field?"
            ]
            
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggestion_{hash(suggestion)}"):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    bot_response = chatbot_response(suggestion, st.session_state.resume_analysis)
                    st.session_state.chat_history.append({"role": "bot", "content": bot_response})
                    st.rerun()
    
    # Navigation buttons
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Welcome"):
            st.session_state.setup_stage = "welcome_page"
            st.rerun()
    
    with col2:
        if st.button("Continue to Interview Setup →"):
            st.session_state.setup_stage = "job_selection"
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
# Complete fixed implementation for the interview question phase
# Replace the entire Question phase section in your code with this implementation

# This is the correct order for your interview flow code
# Copy and paste this as a complete replacement for the interview flow section

# Interview Screen (introduction and questions)
elif st.session_state.setup_stage == "interview" and st.session_state.interview_stage == "introduction":
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
elif st.session_state.setup_stage == "interview" and st.session_state.interview_stage == "question":
    # Define all the necessary variables for the current question
    current_q_data = st.session_state.questions[st.session_state.current_question_idx]
    current_category = current_q_data["category"]
    current_question = current_q_data["question"]

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
            # Using the actual audio_recorder component directly
            audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=16000, key="audio_recorder")
            
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
    st.error("An error occurred in the application flow. Please use the Reset button below to restart.")
    
    # Keep the reset button as requested
    if st.button("Reset Application"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state.setup_stage = "welcome_page"
        st.session_state.interview_stage = "introduction"
        st.rerun()