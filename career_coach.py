import streamlit as st
import openai
import json
import PyPDF2
import docx
from io import BytesIO

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
    """Analyze resume using OpenAI - same format as main.py"""
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
    """Generate chatbot response - same format as main.py"""
    chat_history = st.session_state.get('cc_chat_history', [])
    
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
    {chr(10).join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])}
    
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

def run_career_coach():
    """Main function to run the career coach interface"""
    # Dark theme styling
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
    
    # Initialize session state with cc_ prefix to avoid conflicts
    if "cc_resume_text" not in st.session_state:
        st.session_state.cc_resume_text = ""
    if "cc_resume_analysis" not in st.session_state:
        st.session_state.cc_resume_analysis = {}
    if "cc_career_recommendations" not in st.session_state:
        st.session_state.cc_career_recommendations = []
    if "cc_chat_history" not in st.session_state:
        st.session_state.cc_chat_history = []
    if "cc_manual_input" not in st.session_state:
        st.session_state.cc_manual_input = False
    
    # File upload section
    if not st.session_state.cc_resume_text:
        st.markdown("### 📄 Upload Your Resume")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose your resume file",
                type=['pdf', 'docx', 'txt'],
                help="Upload your resume in PDF, DOCX, or TXT format"
            )
            
            if uploaded_file is not None:
                file_type = uploaded_file.type
                
                with st.spinner("Processing your resume..."):
                    if file_type == "application/pdf":
                        resume_text = extract_text_from_pdf(uploaded_file)
                    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        resume_text = extract_text_from_docx(uploaded_file)
                    else:  # txt file
                        resume_text = str(uploaded_file.read(), "utf-8")
                    
                    st.session_state.cc_resume_text = resume_text
                    st.session_state.cc_resume_analysis = analyze_resume_with_ai(resume_text)
                    st.session_state.cc_career_recommendations = generate_career_recommendations(st.session_state.cc_resume_analysis)
                    
                    st.success("Resume processed successfully!")
                    st.rerun()
        
        with col2:
            st.markdown("### Alternative Input")
            if st.button("Paste Resume Text", type="secondary"):
                st.session_state.cc_manual_input = True
        
        if st.session_state.cc_manual_input:
            manual_text = st.text_area("Paste your resume text here:", height=200)
            if st.button("Analyze Resume") and manual_text:
                with st.spinner("Analyzing your resume..."):
                    st.session_state.cc_resume_text = manual_text
                    st.session_state.cc_resume_analysis = analyze_resume_with_ai(manual_text)
                    st.session_state.cc_career_recommendations = generate_career_recommendations(st.session_state.cc_resume_analysis)
                    st.success("Resume analyzed successfully!")
                    st.rerun()
    
    else:
        # Display resume summary
        if st.session_state.cc_resume_analysis:
            with st.expander("📋 Resume Summary", expanded=False):
                analysis = st.session_state.cc_resume_analysis
                
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
        if st.session_state.cc_career_recommendations:
            with st.expander("🎯 Career Recommendations", expanded=False):
                recs = st.session_state.cc_career_recommendations
                
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
            if st.button("Salary Insights"):
                if st.session_state.cc_career_recommendations.get('salary_range'):
                    st.info(f"Estimated salary range: {st.session_state.cc_career_recommendations['salary_range']}")
                else:
                    st.info("Salary information not available")
        
        with col2:
            if st.button("Growth Opportunities"):
                if st.session_state.cc_career_recommendations.get('growth_opportunities'):
                    for opp in st.session_state.cc_career_recommendations['growth_opportunities']:
                        st.write(f"• {opp}")
                else:
                    st.info("Growth information not available")
        
        with col3:
            if st.button("Industry Insights"):
                if st.session_state.cc_career_recommendations.get('industry_insights'):
                    for insight in st.session_state.cc_career_recommendations['industry_insights']:
                        st.write(f"• {insight}")
                else:
                    st.info("Industry insights not available")
        
        with col4:
            if st.button("Upload New Resume"):
                st.session_state.cc_resume_text = ""
                st.session_state.cc_resume_analysis = {}
                st.session_state.cc_career_recommendations = []
                st.session_state.cc_chat_history = []
                st.rerun()
        
        # Chat interface
        st.markdown("### 💬 Career Coach Chat")
        st.markdown("Ask me anything about your career, interview preparation, or professional development!")
        
        # Display chat history
        if st.session_state.cc_chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.cc_chat_history:
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
                st.session_state.cc_chat_history.append({"role": "user", "content": user_input})
                bot_response = chatbot_response(user_input, st.session_state.cc_resume_analysis)
                st.session_state.cc_chat_history.append({"role": "bot", "content": bot_response})
                st.rerun()
        
        # Suggested questions
        if not st.session_state.cc_chat_history:
            st.markdown("**💡 Try asking:**")
            suggestions = [
                "What roles am I best suited for based on my experience?",
                "How can I improve my resume for my target role?",
                "What skills should I focus on developing?",
                "What salary range should I expect?",
                "How can I prepare for interviews in my field?"
            ]
            
            for suggestion in suggestions:
                if st.button(suggestion, key=f"cc_suggestion_{hash(suggestion)}"):
                    st.session_state.cc_chat_history.append({"role": "user", "content": suggestion})
                    bot_response = chatbot_response(suggestion, st.session_state.cc_resume_analysis)
                    st.session_state.cc_chat_history.append({"role": "bot", "content": bot_response})
                    st.rerun()