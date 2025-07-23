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
    """Analyze resume using OpenAI - Updated for v1.0.0+"""
    prompt = f"""
    Analyze this resume and extract key information in JSON format:
    
    {resume_text}
    
    Return only JSON:
    {{
        "name": "candidate name",
        "current_role": "current job title", 
        "experience_years": "estimated years",
        "skills": ["skill1", "skill2", "skill3"],
        "recommended_fields": ["field1", "field2"],
        "strengths": ["strength1", "strength2"]
    }}
    """
    
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
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

def chatbot_response(user_message, resume_analysis):
    """Generate simple chatbot response"""
    context = f"""
    You are a career coach. User's background:
    - Role: {resume_analysis.get('current_role', 'N/A')}
    - Experience: {resume_analysis.get('experience_years', 'N/A')} years
    - Skills: {', '.join(resume_analysis.get('skills', []))}
    
    User asks: {user_message}
    
    Give helpful career advice in 2-3 sentences.
    """
    
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": context}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, I'm having trouble right now. Please try again later."

def run_career_coach():
    """Main career coach interface - Simple version"""
    
    # Apply dark theme
    st.markdown("""
    <style>
    .stApp { background-color: #121212; color: white; }
    h1, h2, h3 { color: white !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚀 Career Coach")
    
    # Two simple tabs
    tab1, tab2 = st.tabs(["📄 Resume Analysis", "💬 Career Chat"])
    
    with tab1:
        st.markdown("### Upload Your Resume")
        
        uploaded_file = st.file_uploader("Choose your resume (PDF or DOCX)", type=['pdf', 'docx'])
        
        if uploaded_file is not None:
            # Extract text
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
            
            if resume_text:
                st.session_state.resume_text = resume_text
                
                # Show preview
                with st.expander("📋 Resume Preview"):
                    st.text_area("", resume_text[:500] + "...", height=150, disabled=True)
                
                # Analyze button
                if st.button("🔍 Analyze Resume", type="primary"):
                    with st.spinner("Analyzing..."):
                        analysis = analyze_resume_with_ai(resume_text)
                        if analysis:
                            st.session_state.resume_analysis = analysis
                            st.success("✅ Analysis complete!")
                            
                            # Show results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Profile:**")
                                st.write(f"Name: {analysis.get('name', 'N/A')}")
                                st.write(f"Role: {analysis.get('current_role', 'N/A')}")
                                st.write(f"Experience: {analysis.get('experience_years', 'N/A')} years")
                            
                            with col2:
                                st.markdown("**Top Skills:**")
                                for skill in analysis.get('skills', [])[:5]:
                                    st.write(f"• {skill}")
                            
                            st.markdown("**Recommended Fields:**")
                            for field in analysis.get('recommended_fields', []):
                                st.write(f"• {field}")
    
    with tab2:
        st.markdown("### Chat with Your Career Coach")
        
        if not st.session_state.get('resume_analysis'):
            st.warning("⚠️ Please analyze your resume first!")
        else:
            # Show chat history
            if st.session_state.chat_history:
                for msg in st.session_state.chat_history[-5:]:  # Show last 5 messages
                    if msg['role'] == 'user':
                        st.markdown(f"**You:** {msg['content']}")
                    else:
                        st.markdown(f"**Coach:** {msg['content']}")
                    st.markdown("---")
            
            # Chat input
            user_input = st.text_area("Ask your career coach:", height=100)
            
            if st.button("💬 Send", type="primary"):
                if user_input.strip():
                    # Add user message
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    
                    # Get response
                    with st.spinner("Thinking..."):
                        response = chatbot_response(user_input, st.session_state.resume_analysis)
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                    
                    st.rerun()
            
            # Quick questions
            st.markdown("**Quick Questions:**")
            quick_questions = [
                "How can I improve my resume?",
                "What salary should I expect?", 
                "What skills should I learn next?"
            ]
            
            for question in quick_questions:
                if st.button(question, key=f"quick_{hash(question)}"):
                    st.session_state.chat_history.append({'role': 'user', 'content': question})
                    with st.spinner("Thinking..."):
                        response = chatbot_response(question, st.session_state.resume_analysis)
                        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    st.rerun()