import streamlit as st
import openai
import json
import PyPDF2
import docx
from io import BytesIO
import re
from datetime import datetime

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
        "interview_focus_areas": ["areas to emphasize in interviews"],
        "learning_resources": ["recommended courses, certifications, or resources"]
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

def generate_personalized_questions(resume_analysis, target_field=None):
    """Generate personalized interview questions based on resume analysis"""
    if not target_field and resume_analysis.get("recommended_fields"):
        target_field = resume_analysis["recommended_fields"][0]
    
    if not target_field:
        target_field = "General"
    
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
        {{"category": "Experience-based", "question": "specific question about their background"}},
        {{"category": "Technical", "question": "technical question based on their skills"}},
        {{"category": "Behavioral", "question": "behavioral question"}},
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
        
        return personalized_questions
        
    except Exception as e:
        st.error(f"Error generating personalized questions: {str(e)}")
        return []

def chatbot_response(user_message, resume_analysis):
    """Generate chatbot response based on user message and resume analysis"""
    context = f"""
    You are a professional career counselor and interview coach. You have access to the user's resume analysis:
    
    Resume Summary:
    - Name: {resume_analysis.get('name', 'User')}
    - Current Role: {resume_analysis.get('current_role', 'N/A')}
    - Experience: {resume_analysis.get('experience_years', 'N/A')} years
    - Skills: {', '.join(resume_analysis.get('skills', []))}
    - Technologies: {', '.join(resume_analysis.get('technologies', []))}
    - Recommended Fields: {', '.join(resume_analysis.get('recommended_fields', []))}
    - Strengths: {', '.join(resume_analysis.get('strengths', []))}
    - Areas for Improvement: {', '.join(resume_analysis.get('areas_for_improvement', []))}
    
    Recent Chat History:
    {chr(10).join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-5:]])}
    
    User Message: {user_message}
    
    Provide helpful, personalized career advice based on their background. Be conversational, supportive, and actionable.
    Focus on:
    - Career development advice
    - Interview preparation tips
    - Skill development recommendations
    - Industry insights
    - Job search strategies
    
    Keep responses concise but informative (2-3 paragraphs max).
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
        return f"I apologize, but I'm having trouble processing your request right now. Please try again later. Error: {str(e)}"

def run_career_coach():
    """Main function to run the career coach interface"""
    
    # Apply dark theme styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    
    /* Hide Streamlit elements */
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
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    .blue-accent {
        color: #3498db !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1E1E1E;
        color: white;
        border: 1px solid #3498db;
        border-radius: 4px;
        padding: 10px 15px;
        font-size: 16px;
        font-weight: normal;
        margin: 5px 0;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #3498db;
        transform: scale(1.02);
    }
    
    /* Primary button */
    .stButton > [data-testid="baseButton-primary"] {
        background-color: #3498db !important;
        color: white !important;
        border: none !important;
        font-weight: bold;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1E1E1E;
        border: 1px solid #3498db;
        border-radius: 4px;
    }
    
    /* Text areas and inputs */
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background-color: #1E1E1E !important;
        color: white !important;
        border-color: #3498db !important;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background-color: #1E1E1E;
        border: 1px solid #3498db;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 10px;
        margin: 10px 0;
        border-radius: 8px;
        border-left: 4px solid #3498db;
    }
    
    .user-message {
        background-color: #1E1E1E;
        margin-left: 20px;
    }
    
    .assistant-message {
        background-color: #2A2A2A;
        margin-right: 20px;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #3498db;
        border-radius: 4px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚀 Career Coach & Resume Analysis")
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["📄 Resume Upload", "💬 Career Chat", "🎯 Interview Prep"])
    
    with tab1:
        st.markdown("### Upload Your Resume")
        st.markdown("Upload your resume in PDF or DOCX format to get personalized career insights and recommendations.")
        
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx'],
            help="Supported formats: PDF, DOCX"
        )
        
        if uploaded_file is not None:
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = extract_text_from_docx(uploaded_file)
            else:
                st.error("Unsupported file format")
                return
            
            if resume_text:
                st.session_state.resume_text = resume_text
                
                # Show extracted text preview
                with st.expander("📋 Resume Text Preview"):
                    st.text_area("Extracted text:", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=200, disabled=True)
                
                # Analyze resume
                if st.button("🔍 Analyze Resume", type="primary"):
                    with st.spinner("Analyzing your resume..."):
                        analysis = analyze_resume_with_ai(resume_text)
                        if analysis:
                            st.session_state.resume_analysis = analysis
                            st.success("✅ Resume analysis complete!")
                            
                            # Display analysis results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### 👤 Profile Summary")
                                st.markdown(f"**Name:** {analysis.get('name', 'Not found')}")
                                st.markdown(f"**Current Role:** {analysis.get('current_role', 'Not specified')}")
                                st.markdown(f"**Experience:** {analysis.get('experience_years', 'Not specified')} years")
                                
                                if analysis.get('skills'):
                                    st.markdown("**Top Skills:**")
                                    for skill in analysis['skills'][:5]:
                                        st.markdown(f"• {skill}")
                            
                            with col2:
                                st.markdown("#### 🎯 Recommended Fields")
                                if analysis.get('recommended_fields'):
                                    for field in analysis['recommended_fields'][:3]:
                                        st.markdown(f"• {field}")
                                
                                st.markdown("#### 💪 Key Strengths")
                                if analysis.get('strengths'):
                                    for strength in analysis['strengths'][:3]:
                                        st.markdown(f"• {strength}")
                            
                            # Generate career recommendations
                            with st.spinner("Generating career recommendations..."):
                                recommendations = generate_career_recommendations(analysis)
                                if recommendations:
                                    st.session_state.career_recommendations = recommendations
                                    
                                    st.markdown("---")
                                    st.markdown("### 🚀 Career Recommendations")
                                    
                                    # Suitable roles
                                    if recommendations.get('suitable_roles'):
                                        st.markdown("#### 🎯 Suitable Roles for You")
                                        for role in recommendations['suitable_roles']:
                                            st.markdown(f"• {role}")
                                    
                                    # Skills to develop
                                    if recommendations.get('skill_recommendations'):
                                        st.markdown("#### 📚 Skills to Develop")
                                        for skill in recommendations['skill_recommendations']:
                                            st.markdown(f"• {skill}")
                                    
                                    # Next steps
                                    if recommendations.get('next_steps'):
                                        st.markdown("#### 📋 Next Steps")
                                        for step in recommendations['next_steps']:
                                            st.markdown(f"• {step}")
        
        else:
            st.info("👆 Upload your resume to get started with personalized career analysis!")
    
    with tab2:
        st.markdown("### 💬 Chat with Your Career Coach")
        
        if not st.session_state.get('resume_analysis'):
            st.warning("⚠️ Please upload and analyze your resume first to get personalized career advice!")
        else:
            st.markdown("Ask me anything about your career, interview preparation, or professional development!")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("#### 📝 Chat History")
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>You:</strong> {message['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>Career Coach:</strong> {message['content']}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Chat input
            user_input = st.text_area("Ask your career coach:", height=100, key="career_chat_input")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("💬 Send", type="primary"):
                    if user_input.strip():
                        # Add user message to history
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': user_input
                        })
                        
                        # Generate response
                        with st.spinner("Thinking..."):
                            response = chatbot_response(user_input, st.session_state.resume_analysis)
                            
                            # Add response to history
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': response
                            })
                        
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Quick questions
            st.markdown("#### 🔥 Quick Questions")
            quick_questions = [
                "How can I improve my resume?",
                "What salary should I expect?",
                "What skills should I learn next?",
                "How do I prepare for interviews?",
                "What are good career advancement opportunities?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(quick_questions):
                with cols[i % 2]:
                    if st.button(question, key=f"quick_q_{i}"):
                        # Add to chat history and generate response
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': question
                        })
                        
                        with st.spinner("Generating response..."):
                            response = chatbot_response(question, st.session_state.resume_analysis)
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': response
                            })
                        
                        st.rerun()
    
    with tab3:
        st.markdown("### 🎯 Personalized Interview Preparation")
        
        if not st.session_state.get('resume_analysis'):
            st.warning("⚠️ Please upload and analyze your resume first to get personalized interview questions!")
        else:
            st.markdown("Get interview questions tailored specifically to your background and experience!")
            
            # Field selection for targeted questions
            analysis = st.session_state.resume_analysis
            recommended_fields = analysis.get('recommended_fields', [])
            
            if recommended_fields:
                st.markdown("#### 🎯 Recommended Fields Based on Your Resume")
                selected_field = st.selectbox(
                    "Select a field for targeted interview questions:",
                    options=recommended_fields + ["Other"],
                    index=0
                )
                
                if selected_field == "Other":
                    selected_field = st.text_input("Enter the field you're targeting:")
                
                if st.button("🎤 Generate Personalized Questions", type="primary"):
                    with st.spinner("Creating personalized interview questions..."):
                        questions = generate_personalized_questions(analysis, selected_field)
                        
                        if questions:
                            st.session_state.personalized_questions = questions
                            st.success(f"✅ Generated {len(questions)} personalized questions!")
                            
                            # Display questions
                            st.markdown("#### 📝 Your Personalized Interview Questions")
                            for i, q_data in enumerate(questions, 1):
                                with st.expander(f"Question {i}: {q_data.get('category', 'General')}"):
                                    st.markdown(f"**Question:** {q_data.get('question', 'N/A')}")
                                    st.markdown(f"**Category:** {q_data.get('category', 'General')}")
                                    
                                    # Add a text area for practice answers
                                    practice_answer = st.text_area(
                                        "Practice your answer:",
                                        key=f"practice_answer_{i}",
                                        height=100
                                    )
                            
                            # Option to use these questions in the main interview
                            st.markdown("---")
                            if st.button("🚀 Start Interview with These Questions", type="primary"):
                                # Set up the personalized interview
                                st.session_state.questions = questions
                                st.session_state.selected_job_field = selected_field
                                st.session_state.current_question_idx = 0
                                st.session_state.answers = [""] * len(questions)
                                st.session_state.feedbacks = [""] * len(questions)
                                st.session_state.evaluations = []
                                st.session_state.interview_complete = False
                                st.session_state.show_feedback = False
                                st.session_state.question_spoken = False
                                st.session_state.interview_stage = "introduction"
                                st.session_state.setup_stage = "interview"
                                
                                st.success("🎯 Starting personalized interview! Redirecting...")
                                st.rerun()
                        else:
                            st.error("Failed to generate personalized questions. Please try again.")
            
            # Display existing personalized questions if available
            if st.session_state.get('personalized_questions'):
                st.markdown("---")
                st.markdown("#### 📋 Previously Generated Questions")
                st.info(f"You have {len(st.session_state.personalized_questions)} personalized questions ready!")
                
                if st.button("🔄 View/Edit Questions"):
                    for i, q_data in enumerate(st.session_state.personalized_questions, 1):
                        st.markdown(f"**{i}.** {q_data.get('question', 'N/A')} ({q_data.get('category', 'General')})")
    
    # Summary section at the bottom
    if st.session_state.get('resume_analysis') or st.session_state.get('career_recommendations'):
        st.markdown("---")
        st.markdown("### 📊 Career Summary Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.get('resume_analysis'):
                exp_years = st.session_state.resume_analysis.get('experience_years', '0')
                st.metric("Experience", f"{exp_years} years")
        
        with col2:
            if st.session_state.get('resume_analysis'):
                skills_count = len(st.session_state.resume_analysis.get('skills', []))
                st.metric("Skills Identified", f"{skills_count}")
        
        with col3:
            if st.session_state.get('career_recommendations'):
                roles_count = len(st.session_state.career_recommendations.get('suitable_roles', []))
                st.metric("Suitable Roles", f"{roles_count}")