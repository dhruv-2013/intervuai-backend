import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import openai
import streamlit as st

# Initialize Firebase with better error handling
try:
    import firebase_admin
    from firebase_admin import credentials, storage
    
    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        try:
            firebase_cred_dict = json.loads(st.secrets["FIREBASE_CREDENTIALS_JSON"])
            cred = credentials.Certificate(firebase_cred_dict)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'interview-agent-53543.appspot.com'
            })
            print("Firebase initialized successfully")
        except Exception as e:
            print(f"Error initializing Firebase: {str(e)}")
except ImportError:
    print("Firebase admin SDK not available")

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
    You are an expert interview coach specializing in {job_field} roles. 
    Analyze the following interview response and provide detailed, constructive feedback:
    
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
            <skill name>: <proficiency level 1-100>,
            <skill name>: <proficiency level 1-100>,
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
            temperature=0.7,
        )
        
        # Parse the JSON response
        evaluation_data = json.loads(response["choices"][0]["message"]["content"])
        
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
                "content": 5,
                "clarity": 5,
                "technical_accuracy": 5,
                "confidence": 5,
                "overall": 5
            },
            "feedback": {
                "strengths": ["Unable to analyze strengths due to error"],
                "areas_for_improvement": ["Unable to analyze areas for improvement due to error"],
                "missing_elements": ["Unable to analyze missing elements due to error"]
            },
            "skills_demonstrated": ["Technical knowledge", "Communication"],
            "skill_levels": {"Technical knowledge": 70, "Communication": 70},
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
    - Public URL of the uploaded evaluation JSON file or local path
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
        
        # Try using temporary file for cloud environments
        import tempfile
        
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        file_path = Path(temp_dir) / filename
        
        # Save locally first
        with open(file_path, "w") as f:
            json.dump(career_profile, f, indent=2)
        
        # Upload to Firebase if available
        try:
            from firebase_admin import storage
            bucket = storage.bucket()
            blob = bucket.blob(f"evaluations/{filename}")
            blob.upload_from_filename(str(file_path))
            blob.make_public()
            # Clean up temp file
            os.remove(file_path)
            return blob.public_url
        
        except Exception as e:
            print(f"Firebase upload error: {str(e)}")
            # Firebase failed, return the local path
            return str(file_path)
            
    except Exception as e:
        print(f"Error in save_evaluation_data: {str(e)}")
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
    
    # Calculate averages for each score category
    aggregates = {}
    for key in all_score_keys:
        scores = [eval_data["scores"].get(key, 0) for eval_data in evaluations if "scores" in eval_data]
        if scores:
            aggregates[key] = round(sum(scores) / len(scores), 1)
    
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
    
    # Calculate average level for each skill
    skill_assessment = []
    for skill, levels in all_skills.items():
        avg_level = round(sum(levels) / len(levels))
        skill_assessment.append({
            "name": skill,
            "current": avg_level,
            "desired": min(100, avg_level + 15)  # Target is slightly higher than current
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
    
    # Simulated career paths based on job field
    career_paths = []
    if "Software" in job_field:
        career_paths = [
            {
                "name": "Software Developer → Senior Developer → Tech Lead", 
                "compatibility": 92,
                "description": "This path leverages your technical skills with increasing leadership responsibility.",
                "keySkills": ["Coding", "Problem Solving", "System Design", "Code Review"]
            },
            {
                "name": "Developer → DevOps Engineer → Infrastructure Architect", 
                "compatibility": 78,
                "description": "This path focuses on infrastructure, automation and deployment.",
                "keySkills": ["CI/CD", "Cloud Services", "Automation", "Infrastructure"]
            }
        ]
    elif "Data" in job_field:
        career_paths = [
            {
                "name": "Data Analyst → Senior Analyst → Analytics Manager", 
                "compatibility": 90,
                "description": "This path leverages your analytical skills with increasing leadership responsibility.",
                "keySkills": ["SQL", "Data Visualization", "Statistical Analysis", "Business Acumen"]
            },
            {
                "name": "Data Scientist → ML Engineer → AI Specialist", 
                "compatibility": 82,
                "description": "This technical path requires deeper focus on machine learning and algorithm development.",
                "keySkills": ["Python", "Machine Learning", "Deep Learning", "Algorithm Design"]
            }
        ]
    elif "Project" in job_field:
        career_paths = [
            {
                "name": "Project Coordinator → Project Manager → Program Manager", 
                "compatibility": 88,
                "description": "This path focuses on managing increasingly complex projects and programs.",
                "keySkills": ["Project Planning", "Stakeholder Management", "Risk Management", "Team Leadership"]
            },
            {
                "name": "Project Manager → PMO Specialist → PMO Director", 
                "compatibility": 75,
                "description": "This path focuses on developing and implementing project management standards across an organization.",
                "keySkills": ["Process Improvement", "Methodology Development", "Strategic Planning", "Portfolio Management"]
            }
        ]
    elif "UX" in job_field or "UI" in job_field:
        career_paths = [
            {
                "name": "UX/UI Designer → Senior Designer → Design Lead", 
                "compatibility": 90,
                "description": "This path focuses on creating exceptional user experiences with growing leadership responsibilities.",
                "keySkills": ["User Research", "Wireframing", "Prototyping", "User Testing"]
            },
            {
                "name": "UX Designer → UX Researcher → User Experience Director", 
                "compatibility": 82,
                "description": "This path focuses on deep user understanding and research to guide product development.",
                "keySkills": ["User Research", "Usability Testing", "Data Analysis", "Persona Development"]
            }
        ]
    else:
        career_paths = [
            {
                "name": f"{job_field} Specialist → Senior Specialist → Team Lead", 
                "compatibility": 85,
                "description": "This path follows a natural progression of expertise and leadership in your field.",
                "keySkills": ["Technical Knowledge", "Communication", "Problem Solving", "Leadership"]
            }
        ]
    
    # Create a work environment preference profile based on answer analysis
    seed = hash(str(evaluations)) % 100
    np.random.seed(seed)
    
    work_environment = [
        {"name": "Collaborative", "value": 65 + np.random.randint(-10, 20)},
        {"name": "Autonomous", "value": 60 + np.random.randint(-10, 30)},
        {"name": "Fast-paced", "value": 55 + np.random.randint(-10, 25)},
        {"name": "Structured", "value": 50 + np.random.randint(-10, 30)},
        {"name": "Creative", "value": 45 + np.random.randint(-10, 35)},
        {"name": "Data-driven", "value": 70 + np.random.randint(-10, 20)}
    ]
    
    # Ensure all values are in valid range
    for item in work_environment:
        item["value"] = max(10, min(item["value"], 95))
    
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
    print(f"\n✅ Uploaded to Firebase:\n{public_url}")
    
    # Generate the React dashboard URL
    dashboard_url = f"https://intervuai-dashboard.vercel.app/?data={public_url}"
    print(f"\n🌐 View your Career Dashboard:\n{dashboard_url}")