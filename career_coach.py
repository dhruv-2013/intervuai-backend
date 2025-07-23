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
        # Updated for newer OpenAI library versions
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        analysis_text = response.choices[0].message.content.strip()
        analysis_text = analysis_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(analysis_text)
    except Exception as e:
        # Fallback to older API style if new one fails
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content.strip()
            analysis_text = analysis_text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(analysis_text)
        except Exception as e2:
            st.error(f"Error analyzing resume: {str(e2)}")
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
        # Updated for newer OpenAI library versions
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        recommendations_text = response.choices[0].message.content.strip()
        recommendations_text = recommendations_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(recommendations_text)
    except Exception as e:
        # Fallback to older API style if new one fails
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            
            recommendations_text = response.choices[0].message.content.strip()
            recommendations_text = recommendations_text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(recommendations_text)
        except Exception as e2:
            st.error(f"Error generating career recommendations: {str(e2)}")
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
        # Updated for newer OpenAI library versions
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": context}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback to older API style if new one fails
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": context}],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e2:
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