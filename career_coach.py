import streamlit as st
import json
import PyPDF2
import docx
from io import BytesIO
import requests
import time

def make_openai_request(data, max_retries=3):
    """Make OpenAI API request with retry logic for rate limits"""
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limit hit, wait and retry
                wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                st.info(f"Rate limit reached. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            else:
                st.error(f"OpenAI API error: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    
    st.error("Max retries reached. Please try again in a few minutes.")
    return None

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
    # Truncate resume text if too long to reduce token usage
    if len(resume_text) > 3000:
        resume_text = resume_text[:3000] + "..."
    
    prompt = f"""
    Extract key info from this resume in JSON format:
    
    {resume_text}
    
    Return JSON:
    {{
        "name": "name",
        "email": "email", 
        "phone": "phone",
        "current_role": "job title",
        "experience_years": "years",
        "skills": ["skills"],
        "strengths": ["strengths"],
        "areas_for_improvement": ["improvements"]
    }}
    """
    
    try:
        # Use helper function with retry logic
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,  # Reduced from 1000
            "temperature": 0.3
        }
        
        result = make_openai_request(data)
        
        if result:
            analysis_text = result["choices"][0]["message"]["content"].strip()
            analysis_text = analysis_text.replace("```json", "").replace("```", "").strip()
            return json.loads(analysis_text)
        else:
            return {}
            
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
        # Use helper function with retry logic
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an expert career counselor providing structured recommendations."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.5
        }
        
        result = make_openai_request(data)
        
        if result:
            recommendations_text = result["choices"][0]["message"]["content"].strip()
            recommendations_text = recommendations_text.replace("```json", "").replace("```", "").strip()
            return json.loads(recommendations_text)
        else:
            return {}
            
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
        # Use helper function with retry logic
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an expert interview coach providing conversational career advice."},
                {"role": "user", "content": context}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        result = make_openai_request(data)
        
        if result:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "I apologize, but I'm having trouble processing your request right now. Please try again in a few minutes."
            
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your request right now. Please try again later."

def run_career_coach():
    """Main function to run the career coach interface"""
    
    # Apply dark theme styling
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
    .stButton > [data-testid="baseButton-primary"] {
        background-color: #3498db !important;
        color: white !important;
        border: none !important;
        font-weight: bold;
    }
    h1, h2, h3, h4 {
        color: white !important;
    }
    .blue-accent {
        color: #3498db !important;
    }
    [data-testid="stTextInput"] > div > div > input {
        background-color: #1E1E1E !important;
        color: white !important;
        border-color: #3498db !important;
    }
    [data-testid="stTextArea"] textarea {
        background-color: #1E1E1E !important;
        color: white !important;
        border-color: #3498db !important;
    }
    .chat-message {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border: 1px solid #333;
    }
    .user-message {
        background-color: #2E4057;
        border-color: #3498db;
    }
    .bot-message {
        background-color: #1E1E1E;
        border-color: #555;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📄 Resume Analysis & Career Coach")
    
    # Resume upload section
    st.markdown("### Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Choose your resume file", 
        type=['pdf', 'docx', 'txt'],
        help="Upload your resume in PDF, DOCX, or TXT format"
    )
    
    if uploaded_file is not None:
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(uploaded_file)
        else:  # txt file
            resume_text = str(uploaded_file.read(), "utf-8")
        
        st.session_state.resume_text = resume_text
        
        if resume_text:
            st.success("✅ Resume uploaded successfully!")
            
            # Show a preview of the extracted text
            with st.expander("Preview extracted text"):
                st.text_area("Resume content:", resume_text, height=200, disabled=True)
            
            # Analyze resume button
            if st.button("🔍 Analyze Resume", type="primary"):
                with st.spinner("Analyzing your resume..."):
                    st.session_state.resume_analysis = analyze_resume_with_ai(resume_text)
                
                if st.session_state.resume_analysis:
                    st.success("✅ Resume analysis complete!")
                    st.rerun()
        else:
            st.error("❌ Could not extract text from the uploaded file. Please try a different format.")
    
    # Show analysis results if available
    if st.session_state.get('resume_analysis'):
        analysis = st.session_state.resume_analysis
        
        st.markdown("---")
        st.markdown("### 📊 Resume Analysis Results")
        
        # Basic info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Personal Information:**")
            st.write(f"• **Name:** {analysis.get('name', 'Not found')}")
            st.write(f"• **Current Role:** {analysis.get('current_role', 'Not specified')}")
            st.write(f"• **Experience:** {analysis.get('experience_years', 'Not specified')} years")
            
        with col2:
            st.markdown("**Contact Information:**")
            st.write(f"• **Email:** {analysis.get('email', 'Not found')}")
            st.write(f"• **Phone:** {analysis.get('phone', 'Not found')}")
        
        # Skills and technologies
        if analysis.get('skills'):
            st.markdown("**Technical Skills:**")
            skills_text = ", ".join(analysis['skills'])
            st.write(skills_text)
        
        if analysis.get('technologies'):
            st.markdown("**Technologies:**")
            tech_text = ", ".join(analysis['technologies'])
            st.write(tech_text)
        
        # Strengths and improvements
        col1, col2 = st.columns(2)
        
        with col1:
            if analysis.get('strengths'):
                st.markdown("**Strengths:**")
                for strength in analysis['strengths']:
                    st.write(f"• {strength}")
        
        with col2:
            if analysis.get('areas_for_improvement'):
                st.markdown("**Areas for Improvement:**")
                for area in analysis['areas_for_improvement']:
                    st.write(f"• {area}")
        
        # Career recommendations
        if st.button("💡 Get Career Recommendations"):
            with st.spinner("Generating career recommendations..."):
                st.session_state.career_recommendations = generate_career_recommendations(analysis)
            
            if st.session_state.career_recommendations:
                st.rerun()
    
    # Show career recommendations if available
    if st.session_state.get('career_recommendations'):
        recommendations = st.session_state.career_recommendations
        
        st.markdown("---")
        st.markdown("### 🚀 Career Recommendations")
        
        # Suitable roles
        if recommendations.get('suitable_roles'):
            st.markdown("**Recommended Job Roles:**")
            for role in recommendations['suitable_roles']:
                st.write(f"• {role}")
        
        # Growth opportunities
        if recommendations.get('growth_opportunities'):
            st.markdown("**Career Growth Opportunities:**")
            for opportunity in recommendations['growth_opportunities']:
                st.write(f"• {opportunity}")
        
        # Skill recommendations
        if recommendations.get('skill_recommendations'):
            st.markdown("**Skills to Develop:**")
            for skill in recommendations['skill_recommendations']:
                st.write(f"• {skill}")
        
        # Salary range
        if recommendations.get('salary_range'):
            st.markdown(f"**Estimated Salary Range:** {recommendations['salary_range']}")
        
        # Next steps
        if recommendations.get('next_steps'):
            st.markdown("**Next Steps:**")
            for step in recommendations['next_steps']:
                st.write(f"• {step}")
    
    # Chat interface
    if st.session_state.get('resume_analysis'):
        st.markdown("---")
        st.markdown("### 💬 Career Coaching Chat")
        st.markdown("Ask me anything about your career, interview preparation, or professional development!")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("**Chat History:**")
            for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message">**You:** {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message">**Coach:** {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Ask me anything about your career:", key="career_chat_input")
        
        if st.button("Send") and user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Generate bot response
            with st.spinner("Thinking..."):
                bot_response = chatbot_response(user_input, st.session_state.resume_analysis)
                st.session_state.chat_history.append({"role": "bot", "content": bot_response})
            
            st.rerun()
        
        # Quick action buttons
        st.markdown("**Quick Questions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("How can I improve my resume?"):
                st.session_state.chat_history.append({"role": "user", "content": "How can I improve my resume?"})
                bot_response = chatbot_response("How can I improve my resume?", st.session_state.resume_analysis)
                st.session_state.chat_history.append({"role": "bot", "content": bot_response})
                st.rerun()
        
        with col2:
            if st.button("What should I focus on in interviews?"):
                st.session_state.chat_history.append({"role": "user", "content": "What should I focus on in interviews?"})
                bot_response = chatbot_response("What should I focus on in interviews?", st.session_state.resume_analysis)
                st.session_state.chat_history.append({"role": "bot", "content": bot_response})
                st.rerun()
        
        with col3:
            if st.button("What are my career options?"):
                st.session_state.chat_history.append({"role": "user", "content": "What are my career options?"})
                bot_response = chatbot_response("What are my career options?", st.session_state.resume_analysis)
                st.session_state.chat_history.append({"role": "bot", "content": bot_response})
                st.rerun()
    
    else:
        st.info("👆 Upload your resume above to start using the career coach!")