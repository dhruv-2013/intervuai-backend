import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import openai
import streamlit as st
import tempfile
import firebase_admin
from firebase_admin import credentials, storage

# Define configure_firebase_cors function BEFORE it's called
def configure_firebase_cors():
    """Configure CORS settings for Firebase Storage to allow the dashboard to fetch data"""
    try:
        # Get the storage bucket
        bucket = storage.bucket()
        
        # Set comprehensive CORS rules
        bucket.cors = [
            {
                # Include both the dashboard URL and localhost for testing
                "origin": [
                    "https://intervuai-dashboard.vercel.app", 
                    "http://localhost:3000",
                    # It's sometimes helpful to include the www variant and http version
                    "http://intervuai-dashboard.vercel.app",
                    "https://www.intervuai-dashboard.vercel.app"
                ],
                # Allow more HTTP methods - GET is required for fetching the JSON
                "method": ["GET", "HEAD", "OPTIONS"],
                # 3600 seconds = 1 hour cache time
                "maxAgeSeconds": 3600,
                # Include all necessary headers for CORS requests
                "responseHeader": [
                    "Content-Type", 
                    "Access-Control-Allow-Origin", 
                    "Access-Control-Allow-Methods",
                    "Access-Control-Allow-Headers",
                    "Content-Length"
                ]
            }
        ]
        bucket.patch()  # Apply the new rules explicitly
        # Apply the update to the bucket
        bucket.update()
        st.success("✅ CORS configuration updated for Firebase bucket")
        return True
    except Exception as e:
        st.error(f"Failed to update CORS: {e}")
        return False

# ——— FIREBASE ADMIN INIT ———
fb_creds_dict = json.loads(st.secrets["FIREBASE_CREDENTIALS_JSON"])

# Dump to a real temp file so Firebase‑Admin can read it
tf = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
tf.write(json.dumps(fb_creds_dict))
tf.flush()
cred_path = tf.name

# Initialize Firebase Admin exactly once
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            "storageBucket": "interview-agent-53543.firebasestorage.app"
        })
        st.write("✅ Firebase initialized from temp file")
        
        # Now call configure_firebase_cors AFTER Firebase is initialized
        configure_firebase_cors()
    except Exception as e:
        st.error(f"Firebase init failed: {e}")
        raise
# —————————————————————————

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# The enhanced answer evaluation function
def get_answer_evaluation(question, answer, job_field):
    """
    Evaluate the interview answer using OpenAI and return structured evaluation data
    for the frontend dashboard.
    
    Parameters:
    - question: The interview question
    - answer: The candidate's answer
    - job_field: The job field (e.g., "Software Engineering", "Data Science")
    
    Returns:
    - Dictionary containing structured evaluation data
    """
    prompt = f"""
    You are a strict interview coach specializing in {job_field} roles. 
    Analyze the following interview response with realistic professional standards:
    
    Question: {question}
    Answer: {answer}
    
    Provide a structured JSON response with the following format:
    {{
        "scores": {{
            "content": <score 1-10 for answer content and relevance>,
            "clarity": <score 1-10 for clarity and organization>,
            "technical_accuracy": <score 1-10 for technical correctness, if applicable>,
            "confidence": <score 1-10 for confidence and delivery>,
            "overall": <overall score 1-10>
        }},
        "feedback": {{
            "strengths": [<list of 2-3 key strengths>],
            "areas_for_improvement": [<list of 2-3 areas to improve>],
            "missing_elements": [<list of key points that should have been included>]
        }},
        "skills_demonstrated": [<list of 3-5 skills demonstrated in the answer>],
        "skill_levels": {{
            <skill name>: <proficiency level 15-75>,
            <skill name>: <proficiency level 15-75>,
            ...
        }},
        "improved_answer": "<brief example of an improved answer (2-3 sentences)>",
        "keywords": [<list of important keywords that should appear in a strong answer>]
    }}
    
    Ensure the response is valid JSON and all scores are integers.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert interview coach providing structured evaluation data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.4,
        )
        
        # Parse the JSON response
        evaluation_data = json.loads(response["choices"][0]["message"]["content"])
        
        # Apply realistic scoring based on answer length
        word_count = len(answer.split()) if answer else 0
        if word_count < 20:
            score_factor = 0.5
        elif word_count < 40:
            score_factor = 0.7
        else:
            score_factor = 1.0
        
        # Adjust scores
        if "scores" in evaluation_data:
            for key in evaluation_data["scores"]:
                evaluation_data["scores"][key] = max(1, int(evaluation_data["scores"][key] * score_factor))
        
        # Cap skill levels
        if "skill_levels" in evaluation_data:
            for skill in evaluation_data["skill_levels"]:
                evaluation_data["skill_levels"][skill] = min(75, max(15, int(evaluation_data["skill_levels"][skill] * score_factor)))
        
        # Add metadata
        evaluation_data["question"] = question
        evaluation_data["answer"] = answer
        evaluation_data["job_field"] = job_field
        evaluation_data["timestamp"] = datetime.now().isoformat()
        
        return evaluation_data
    
    except Exception as e:
        print(f"Error generating evaluation: {str(e)}")
        # Return a basic evaluation structure if there's an error
        return {
            "error": str(e),
            "question": question,
            "answer": answer,
            "job_field": job_field,
            "timestamp": datetime.now().isoformat(),
            "scores": {
                "content": 2,
                "clarity": 2,
                "technical_accuracy": 2,
                "confidence": 2,
                "overall": 2
            },
            "feedback": {
                "strengths": ["Unable to analyze strengths due to error"],
                "areas_for_improvement": ["Unable to analyze areas for improvement due to error"],
                "missing_elements": ["Unable to analyze missing elements due to error"]
            },
            "skills_demonstrated": ["Technical knowledge", "Communication"],
            "skill_levels": {"Technical knowledge": 25, "Communication": 25},
            "improved_answer": "Unable to generate improved answer due to error."
        }

def save_evaluation_data(evaluations, interviewee_name):
    """
    Save the evaluation data to a JSON file and upload it to Firebase Storage.
    Falls back to local storage if Firebase is not available.
    
    Parameters:
    - evaluations: List of evaluation dictionaries
    - interviewee_name: Name of the interviewee
    
    Returns:
    - Public URL of the uploaded evaluation JSON file or direct data object
    """
    try:
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{interviewee_name.lower().replace(' ', '_')}_{timestamp}.json"
        
        # Create the full evaluation data package
        career_profile = {
            "interviewee": interviewee_name,
            "timestamp": datetime.now().isoformat(),
            "responses": evaluations,
            "aggregate_scores": calculate_aggregate_scores(evaluations),
            "skill_assessment": aggregate_skill_assessment(evaluations),
            "career_insights": generate_career_insights(evaluations)
        }
        
        # Create temp file for Firebase upload
        temp_dir = tempfile.gettempdir()
        file_path = Path(temp_dir) / filename
        
        # Write career_profile to temp file
        with open(file_path, "w") as f:
            json.dump(career_profile, f, indent=2)
        
        # Try Firebase upload
        try:
            # Make sure Firebase is initialized
            if not firebase_admin._apps:
                st.warning("Firebase not initialized. Initializing now...")
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    "storageBucket": "interview-agent-53543.firebasestorage.app"
                })
                
                # Configure CORS immediately after initialization
                configure_firebase_cors()
            
            bucket = storage.bucket()
            blob = bucket.blob(f"evaluations/{filename}")
            blob.upload_from_filename(str(file_path))
            
            # Add content type metadata
            blob.content_type = "application/json"
            
            # Add CORS headers in the metadata
            metadata = {
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f"inline; filename={filename}"
            }
            blob.metadata = metadata
            blob.patch()
            
            # Make the file publicly accessible
            blob.make_public()
            public_url = blob.public_url
            
            # Clean up temp file
            os.remove(file_path)

            
            # Generate dashboard URL with Firebase data link
            dashboard_url = f"https://intervuai-dashboard.vercel.app/?data={public_url}"
            #st.success(f"({dashboard_url})")
            
            return public_url
            
        except Exception as firebase_error:
            st.warning(f"Firebase upload failed: {str(firebase_error)}. Using direct data approach.")
            
            # Create a Base64 encoded data version for direct embedding
            # Note: This is only suitable for smaller data sizes
            try:
                # Create a minimal version of the data to reduce size
                compact_data = {
                    "interviewee": career_profile["interviewee"],
                    "timestamp": career_profile["timestamp"],
                    "responses": career_profile["responses"],
                    "aggregate_scores": career_profile["aggregate_scores"],
                    "skill_assessment": career_profile["skill_assessment"],
                    "career_insights": career_profile["career_insights"]
                }
                
                # Encode as base64
                import base64
                json_str = json.dumps(compact_data)
                encoded_bytes = base64.b64encode(json_str.encode('utf-8'))
                encoded_str = encoded_bytes.decode('utf-8')
                
                # Create dashboard URL with encoded data
                dashboard_url = f"https://intervuai-dashboard.vercel.app/?encoded_data={encoded_str}"
                
                # Check URL length - if too long, this won't work
                if len(dashboard_url) > 4000:  # Most browsers have limits around 4-8K
                    raise ValueError("URL too long for direct data embedding")
                
                st.success("✅ Created direct data link")
                st.success(f"({dashboard_url})")
                
                # Also provide a download button for the full data
                json_str = json.dumps(career_profile, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download Profile Data</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                return dashboard_url
                
            except Exception as encoding_error:
                st.warning(f"Direct data encoding failed: {str(encoding_error)}. Providing download option.")
                
                # Provide download button for the data file
                with open(file_path, "r") as f:
                    json_data = f.read()
                
                json_str = json.dumps(career_profile, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                
                st.markdown("### Your career profile is ready")
                st.info("To view your dashboard, download the file and upload it on the dashboard page.")
                
                href = f'<a href="data:application/json;base64,{b64}" download="{filename}" class="button">Download Career Profile Data</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                dashboard_link = "https://intervuai-dashboard.vercel.app/"
                st.markdown(f"[Open Dashboard]({dashboard_link}) and upload the file you downloaded")
                
                # Clean up temp file
                os.remove(file_path)
                
                return {
                    "status": "download_only",
                    "data": career_profile
                }
                
    except Exception as e:
        st.error(f"Error in save_evaluation_data: {str(e)}")
        return None

def calculate_aggregate_scores(evaluations):
    """Calculate aggregate scores across all evaluations"""
    if not evaluations:
        return {}
    
    # Extract all score categories
    all_score_keys = set()
    for eval_data in evaluations:
        if "scores" in eval_data:
            all_score_keys.update(eval_data["scores"].keys())
    
    # Calculate averages with penalties
    aggregates = {}
    num_questions = len(evaluations)
    
    for key in all_score_keys:
        scores = [eval_data["scores"].get(key, 0) for eval_data in evaluations if "scores" in eval_data]
        if scores:
            avg_score = sum(scores) / len(scores)
            # Apply penalty for limited questions
            if num_questions < 3:
                avg_score *= 0.7
            elif num_questions < 5:
                avg_score *= 0.85
            aggregates[key] = round(max(1, avg_score), 1)
    
    return aggregates

def aggregate_skill_assessment(evaluations):
    """Aggregate skill assessments across all evaluations"""
    all_skills = {}
    demonstrated_skills = []
    
    for eval_data in evaluations:
        # Collect skills demonstrated
        if "skills_demonstrated" in eval_data:
            demonstrated_skills.extend(eval_data["skills_demonstrated"])
        
        # Collect skill levels
        if "skill_levels" in eval_data:
            for skill, level in eval_data["skill_levels"].items():
                if skill not in all_skills:
                    all_skills[skill] = []
                all_skills[skill].append(level)
    
    # Calculate realistic skill gaps
    skill_assessment = []
    for skill, levels in all_skills.items():
        avg_level = round(sum(levels) / len(levels))
        # Apply larger skill gaps
        if avg_level < 30:
            gap = 45
        elif avg_level < 50:
            gap = 35
        else:
            gap = 25
        skill_assessment.append({
            "name": skill,
            "current": avg_level,
            "desired": min(95, avg_level + gap)
        })
    
    # Add frequency count for demonstrated skills
    skill_frequency = {}
    for skill in demonstrated_skills:
        skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
    
    # Sort by frequency
    top_demonstrated = [{"name": k, "count": v} for k, v in 
                        sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)]
    
    return {
        "assessed_levels": skill_assessment,
        "demonstrated_skills": top_demonstrated[:8]  # Return top 8 most demonstrated
    }

def generate_career_insights(evaluations):
    """Generate career insights based on evaluations"""
    # Extract job field if available
    job_field = "General"
    if evaluations and "job_field" in evaluations[0]:
        job_field = evaluations[0]["job_field"]
    
    # Calculate realistic compatibility based on performance
    avg_score = 0
    if evaluations:
        scores = [eval_data.get("scores", {}).get("overall", 3) for eval_data in evaluations]
        avg_score = sum(scores) / len(scores)
    
    base_compatibility = min(75, max(35, int((avg_score / 10) * 70)))
    
    # Simulated career paths with realistic compatibility
    career_paths = []
    if "Software" in job_field:
        career_paths = [
            {
                "name": "Software Developer → Senior Developer → Tech Lead", 
                "compatibility": min(base_compatibility + 5, 75),
                "description": "This path leverages your technical skills with increasing leadership responsibility.",
                "keySkills": ["Coding", "Problem Solving", "System Design", "Code Review"]
            },
            {
                "name": "Developer → DevOps Engineer → Infrastructure Architect", 
                "compatibility": max(35, base_compatibility - 10),
                "description": "This path focuses on infrastructure, automation and deployment.",
                "keySkills": ["CI/CD", "Cloud Services", "Automation", "Infrastructure"]
            }
        ]
    elif "Data" in job_field:
        career_paths = [
            {
                "name": "Data Analyst → Senior Analyst → Analytics Manager", 
                "compatibility": min(base_compatibility + 3, 73),
                "description": "This path leverages your analytical skills with increasing leadership responsibility.",
                "keySkills": ["SQL", "Data Visualization", "Statistical Analysis", "Business Acumen"]
            },
            {
                "name": "Data Scientist → ML Engineer → AI Specialist", 
                "compatibility": max(38, base_compatibility - 8),
                "description": "This technical path requires deeper focus on machine learning and algorithm development.",
                "keySkills": ["Python", "Machine Learning", "Deep Learning", "Algorithm Design"]
            }
        ]
    elif "Project" in job_field:
        career_paths = [
            {
                "name": "Project Coordinator → Project Manager → Program Manager", 
                "compatibility": min(base_compatibility + 3, 71),
                "description": "This path focuses on managing increasingly complex projects and programs.",
                "keySkills": ["Project Planning", "Stakeholder Management", "Risk Management", "Team Leadership"]
            },
            {
                "name": "Project Manager → PMO Specialist → PMO Director", 
                "compatibility": max(36, base_compatibility - 12),
                "description": "This path focuses on developing and implementing project management standards across an organization.",
                "keySkills": ["Process Improvement", "Methodology Development", "Strategic Planning", "Portfolio Management"]
            }
        ]
    elif "UX" in job_field or "UI" in job_field:
        career_paths = [
            {
                "name": "UX/UI Designer → Senior Designer → Design Lead", 
                "compatibility": min(base_compatibility + 4, 73),
                "description": "This path focuses on creating exceptional user experiences with growing leadership responsibilities.",
                "keySkills": ["User Research", "Wireframing", "Prototyping", "User Testing"]
            },
            {
                "name": "UX Designer → UX Researcher → User Experience Director", 
                "compatibility": max(38, base_compatibility - 8),
                "description": "This path focuses on deep user understanding and research to guide product development.",
                "keySkills": ["User Research", "Usability Testing", "Data Analysis", "Persona Development"]
            }
        ]
    else:
        career_paths = [
            {
                "name": f"{job_field} Specialist → Senior Specialist → Team Lead", 
                "compatibility": base_compatibility,
                "description": "This path follows a natural progression of expertise and leadership in your field.",
                "keySkills": ["Technical Knowledge", "Communication", "Problem Solving", "Leadership"]
            }
        ]
    
    # Create a work environment preference profile based on answer analysis
    seed = hash(str(evaluations)) % 100
    np.random.seed(seed)
    
    work_environment = [
        {"name": "Collaborative", "value": 45 + np.random.randint(-10, 20)},
        {"name": "Autonomous", "value": 40 + np.random.randint(-10, 25)},
        {"name": "Fast-paced", "value": 35 + np.random.randint(-10, 20)},
        {"name": "Structured", "value": 40 + np.random.randint(-10, 25)},
        {"name": "Creative", "value": 30 + np.random.randint(-10, 25)},
        {"name": "Data-driven", "value": 50 + np.random.randint(-10, 20)}
    ]
    
    # Ensure realistic range
    for item in work_environment:
        item["value"] = max(15, min(item["value"], 80))
    
    # Create development recommendations
    recommendations = [
        {
            "area": "Technical Skills",
            "recommendation": "Enhance specific technical skills relevant to your desired role",
            "resources": [
                "Online courses on current technologies",
                "Technical certification programs",
                "Hands-on project work"
            ]
        },
        {
            "area": "Communication",
            "recommendation": "Strengthen ability to communicate complex ideas clearly",
            "resources": [
                "Communication workshops",
                "Presentation practice",
                "Technical writing exercises"
            ]
        },
        {
            "area": "Career Growth",
            "recommendation": "Build a stronger professional network in your field",
            "resources": [
                "Industry meetups and conferences",
                "Online professional communities",
                "Informational interviews with leaders"
            ]
        }
    ]
    
    return {
        "careerPaths": career_paths,
        "workEnvironment": work_environment,
        "development": recommendations
    }

# For testing purposes
if __name__ == "__main__":
    sample_question = "Tell me about a time you had to work under pressure to meet a deadline."
    sample_answer = "Last year, our team needed to deliver a critical update within three weeks. I organized everyone into focused workstreams, created daily check-ins, and personally handled the complex technical tasks. We prioritized key features, automated testing, and maintained clear stakeholder communication. We delivered on time with all core requirements met."
    job_field = "Software Engineering"
    
    result = get_answer_evaluation(sample_question, sample_answer, job_field)
    print(json.dumps(result, indent=2))
    
    # Upload to Firebase
    public_url = save_evaluation_data([result], "Test User")
    #print(f"\n✅ Uploaded to Firebase:\n{public_url}")
    
    # Generate the React dashboard URL
    dashboard_url = f"https://intervuai-dashboard.vercel.app/?data={public_url}"
    #print(f"\n🌐 View your Career Dashboard:\n{dashboard_url}")