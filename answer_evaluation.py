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

def evaluate_answer_quality(answer):
    """
    Evaluate the quality and completeness of an answer to determine base scoring range
    
    Returns:
    - quality_score: 1-5 (1=very poor, 5=excellent)
    - completeness_factor: 0.3-1.0 (multiplier for final scores)
    """
    if not answer or len(answer.strip()) < 10:
        return 1, 0.3
    
    answer_length = len(answer.strip())
    word_count = len(answer.split())
    
    # Quality indicators
    quality_indicators = 0
    
    # Length and substance check
    if word_count >= 50:
        quality_indicators += 1
    if word_count >= 100:
        quality_indicators += 1
    
    # Structure indicators
    if any(phrase in answer.lower() for phrase in ["for example", "specifically", "in particular", "such as"]):
        quality_indicators += 1
    
    # Professional language
    if any(phrase in answer.lower() for phrase in ["experience", "responsibility", "achieved", "managed", "developed"]):
        quality_indicators += 1
    
    # Problem-solving indicators
    if any(phrase in answer.lower() for phrase in ["challenge", "solution", "approach", "result", "outcome"]):
        quality_indicators += 1
    
    # Determine quality score (1-5)
    if quality_indicators >= 4 and word_count >= 80:
        quality_score = 5
        completeness_factor = 1.0
    elif quality_indicators >= 3 and word_count >= 60:
        quality_score = 4
        completeness_factor = 0.85
    elif quality_indicators >= 2 and word_count >= 40:
        quality_score = 3
        completeness_factor = 0.70
    elif quality_indicators >= 1 and word_count >= 20:
        quality_score = 2
        completeness_factor = 0.55
    else:
        quality_score = 1
        completeness_factor = 0.40
    
    return quality_score, completeness_factor

# The enhanced answer evaluation function with realistic scoring
def get_answer_evaluation(question, answer, job_field):
    """
    Evaluate the interview answer using OpenAI with realistic, stricter scoring
    
    Parameters:
    - question: The interview question
    - answer: The candidate's answer
    - job_field: The job field (e.g., "Software Engineering", "Data Science")
    
    Returns:
    - Dictionary containing structured evaluation data with realistic scores
    """
    
    # First, evaluate the basic quality of the answer
    quality_score, completeness_factor = evaluate_answer_quality(answer)
    
    prompt = f"""
    You are a strict, experienced interviewer specializing in {job_field} roles. 
    Evaluate the following interview response with realistic, professional standards:
    
    Question: {question}
    Answer: {answer}
    
    SCORING GUIDELINES (BE STRICT):
    - Score 8-10: Exceptional answers with specific examples, clear structure, and deep insights
    - Score 6-7: Good answers with relevant content but may lack depth or examples
    - Score 4-5: Average answers that address the question but lack detail or sophistication
    - Score 2-3: Weak answers that partially address the question with minimal detail
    - Score 1: Poor answers that don't adequately address the question
    
    The answer quality appears to be: {quality_score}/5 (consider this in your evaluation)
    
    Provide a structured JSON response with realistic, strict scoring:
    {{
        "scores": {{
            "content": <score 1-10 for answer content and relevance - be strict>,
            "clarity": <score 1-10 for clarity and organization - be strict>,
            "technical_accuracy": <score 1-10 for technical correctness if applicable - be strict>,
            "confidence": <score 1-10 for confidence and delivery - be strict>,
            "overall": <overall score 1-10 - should be realistic average of other scores>
        }},
        "feedback": {{
            "strengths": [<list of 2-3 genuine strengths, if any exist>],
            "areas_for_improvement": [<list of 3-4 specific areas to improve>],
            "missing_elements": [<list of key elements that should have been included>]
        }},
        "skills_demonstrated": [<list of 2-5 skills actually demonstrated - be conservative>],
        "skill_levels": {{
            <skill name>: <realistic proficiency level 10-85 based on actual demonstration>,
            <skill name>: <realistic proficiency level 10-85 based on actual demonstration>
        }},
        "improved_answer": "<example of a significantly better answer showing what good looks like>",
        "keywords": [<important keywords that should appear in strong answers>],
        "answer_completeness": <percentage 20-95 of how complete the answer was>
    }}
    
    Be honest and realistic in your evaluation. Don't inflate scores.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a strict interview evaluator. Provide realistic, honest assessments. Most candidates score in the 40-70% range, with only exceptional answers scoring higher."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.3,  # Lower temperature for more consistent, less generous scoring
        )
        
        # Parse the JSON response
        evaluation_data = json.loads(response["choices"][0]["message"]["content"])
        
        # Apply completeness factor to make scores more realistic
        if "scores" in evaluation_data:
            for score_key in evaluation_data["scores"]:
                original_score = evaluation_data["scores"][score_key]
                # Apply completeness factor and ensure minimum realistic scoring
                adjusted_score = max(1, int(original_score * completeness_factor))
                evaluation_data["scores"][score_key] = adjusted_score
        
        # Adjust skill levels to be more realistic (cap at 85 for single question)
        if "skill_levels" in evaluation_data:
            for skill in evaluation_data["skill_levels"]:
                current_level = evaluation_data["skill_levels"][skill]
                # Apply realistic capping and completeness factor
                realistic_level = min(85, max(10, int(current_level * completeness_factor)))
                evaluation_data["skill_levels"][skill] = realistic_level
        
        # Add metadata
        evaluation_data["question"] = question
        evaluation_data["answer"] = answer
        evaluation_data["job_field"] = job_field
        evaluation_data["timestamp"] = datetime.now().isoformat()
        evaluation_data["quality_assessment"] = {
            "base_quality": quality_score,
            "completeness_factor": completeness_factor
        }
        
        return evaluation_data
    
    except Exception as e:
        print(f"Error generating evaluation: {str(e)}")
        # Return a realistic low-score evaluation structure if there's an error
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
                "strengths": ["Unable to identify strengths due to evaluation error"],
                "areas_for_improvement": ["Unable to analyze due to error - recommend providing more detailed responses"],
                "missing_elements": ["Unable to analyze missing elements due to error"]
            },
            "skills_demonstrated": ["Basic communication"],
            "skill_levels": {"Basic communication": 25},
            "improved_answer": "Unable to generate improved answer due to error.",
            "answer_completeness": 20,
            "quality_assessment": {
                "base_quality": 1,
                "completeness_factor": 0.3
            }
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
            "aggregate_scores": calculate_realistic_aggregate_scores(evaluations),
            "skill_assessment": aggregate_realistic_skill_assessment(evaluations),
            "career_insights": generate_realistic_career_insights(evaluations)
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

def calculate_realistic_aggregate_scores(evaluations):
    """Calculate realistic aggregate scores with penalties for incomplete data"""
    if not evaluations:
        return {}
    
    # Extract all score categories
    all_score_keys = set()
    for eval_data in evaluations:
        if "scores" in eval_data:
            all_score_keys.update(eval_data["scores"].keys())
    
    # Calculate averages for each score category
    aggregates = {}
    num_questions = len(evaluations)
    
    for key in all_score_keys:
        scores = [eval_data["scores"].get(key, 0) for eval_data in evaluations if "scores" in eval_data]
        if scores:
            base_average = sum(scores) / len(scores)
            
            # Apply penalty for insufficient data (less than 3 questions)
            if num_questions < 3:
                confidence_penalty = 0.7  # 30% penalty
                penalized_score = base_average * confidence_penalty
            elif num_questions < 5:
                confidence_penalty = 0.85  # 15% penalty
                penalized_score = base_average * confidence_penalty
            else:
                penalized_score = base_average
            
            aggregates[key] = round(max(1, penalized_score), 1)
    
    # Add metadata about data completeness
    aggregates["data_completeness"] = {
        "questions_answered": num_questions,
        "confidence_level": "High" if num_questions >= 5 else "Medium" if num_questions >= 3 else "Low",
        "reliability_note": f"Scores based on {num_questions} question(s). More questions recommended for accurate assessment."
    }
    
    return aggregates

def aggregate_realistic_skill_assessment(evaluations):
    """Aggregate skill assessments with realistic skill gaps and penalties for limited data"""
    all_skills = {}
    demonstrated_skills = []
    num_questions = len(evaluations)
    
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
    
    # Calculate realistic skill assessment
    skill_assessment = []
    for skill, levels in all_skills.items():
        avg_level = sum(levels) / len(levels)
        
        # Apply confidence penalty for limited data
        if num_questions < 3:
            confidence_factor = 0.75  # Reduce confidence in skill level
            current_level = int(avg_level * confidence_factor)
        elif num_questions < 5:
            confidence_factor = 0.90
            current_level = int(avg_level * confidence_factor)
        else:
            current_level = int(avg_level)
        
        # Calculate realistic skill gap (30-50% gap instead of just 15%)
        if current_level < 40:
            skill_gap = 45  # Large gap for low skills
        elif current_level < 60:
            skill_gap = 35  # Medium gap
        else:
            skill_gap = 25  # Smaller gap for demonstrated skills
        
        desired_level = min(95, current_level + skill_gap)
        
        skill_assessment.append({
            "name": skill,
            "current": max(10, current_level),
            "desired": desired_level,
            "gap_percentage": round(((desired_level - current_level) / desired_level) * 100, 1),
            "confidence": "High" if num_questions >= 5 else "Medium" if num_questions >= 3 else "Low"
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
        "demonstrated_skills": top_demonstrated[:8],
        "assessment_note": f"Assessment based on {num_questions} question(s). Skill gaps calculated realistically for career development."
    }

def generate_realistic_career_insights(evaluations):
    """Generate realistic career insights with honest assessments"""
    # Extract job field if available
    job_field = "General"
    if evaluations and "job_field" in evaluations[0]:
        job_field = evaluations[0]["job_field"]
    
    num_questions = len(evaluations)
    
    # Calculate average overall score for realistic compatibility
    avg_score = 0
    if evaluations:
        overall_scores = [eval_data.get("scores", {}).get("overall", 3) for eval_data in evaluations]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 3
    
    # Base compatibility on actual performance (more realistic)
    base_compatibility = min(90, max(40, int((avg_score / 10) * 85)))  # Convert 1-10 score to 40-85% compatibility
    
    # Apply penalty for limited data
    if num_questions < 3:
        data_penalty = 0.8
    elif num_questions < 5:
        data_penalty = 0.9
    else:
        data_penalty = 1.0
    
    realistic_compatibility = int(base_compatibility * data_penalty)
    
    # Career paths with realistic compatibility based on performance
    career_paths = []
    if "Software" in job_field:
        career_paths = [
            {
                "name": "Software Developer → Senior Developer → Tech Lead", 
                "compatibility": min(realistic_compatibility + 5, 90),
                "description": "This path leverages your technical skills with increasing leadership responsibility.",
                "keySkills": ["Coding", "Problem Solving", "System Design", "Code Review"],
                "development_time": "2-4 years" if realistic_compatibility > 60 else "3-6 years"
            },
            {
                "name": "Developer → DevOps Engineer → Infrastructure Architect", 
                "compatibility": max(40, realistic_compatibility - 10),
                "description": "This path focuses on infrastructure, automation and deployment.",
                "keySkills": ["CI/CD", "Cloud Services", "Automation", "Infrastructure"],
                "development_time": "2-5 years" if realistic_compatibility > 55 else "4-7 years"
            }
        ]
    elif "Data" in job_field:
        career_paths = [
            {
                "name": "Data Analyst → Senior Analyst → Analytics Manager", 
                "compatibility": min(realistic_compatibility + 3, 88),
                "description": "This path leverages your analytical skills with increasing leadership responsibility.",
                "keySkills": ["SQL", "Data Visualization", "Statistical Analysis", "Business Acumen"],
                "development_time": "2-4 years" if realistic_compatibility > 60 else "3-5 years"
            },
            {
                "name": "Data Scientist → ML Engineer → AI Specialist", 
                "compatibility": max(45, realistic_compatibility - 8),
                "description": "This technical path requires deeper focus on machine learning and algorithm development.",
                "keySkills": ["Python", "Machine Learning", "Deep Learning", "Algorithm Design"],
                "development_time": "3-5 years" if realistic_compatibility > 55 else "4-7 years"
            }
        ]
    else:
        career_paths = [
            {
                "name": f"{job_field} Specialist → Senior Specialist → Team Lead", 
                "compatibility": realistic_compatibility,
                "description": "This path follows a natural progression of expertise and leadership in your field.",
                "keySkills": ["Technical Knowledge", "Communication", "Problem Solving", "Leadership"],
                "development_time": "2-4 years" if realistic_compatibility > 60 else "3-6 years"
            }
        ]
    
    # More realistic work environment assessment
    seed = hash(str(evaluations)) % 100
    np.random.seed(seed)
    
    # Base values on actual performance rather than inflated numbers
    base_values = {
        "Collaborative": 45 + int(avg_score * 2),
        "Autonomous": 40 + int(avg_score * 2.5),
        "Fast-paced": 35 + int(avg_score * 2),
        "Structured": 50 + int(avg_score * 1.5),
        "Creative": 30 + int(avg_score * 2),
        "Data-driven": 45 + int(avg_score * 2)
    }
    
    work_environment = []
    for name, base_value in base_values.items():
        # Add some realistic variance
        variance = np.random.randint(-15, 15)
        final_value = max(20, min(base_value + variance, 85))  # More realistic range
        work_environment.append({"name": name, "value": final_value})
    
    # More targeted development recommendations based on actual weaknesses
    recommendations = [
        {
            "area": "Technical Skills",
            "recommendation": "Focus on developing core technical competencies through structured learning",
            "priority": "High" if avg_score < 6 else "Medium",
            "resources": [
                "Structured online courses with hands-on projects",
                "Technical certification programs",
                "Code review practice and peer learning"
            ]
        },
        {
            "area": "Communication & Presentation",
            "recommendation": "Improve ability to articulate complex ideas clearly and confidently",
            "priority": "High" if avg_score < 5 else "Medium",
            "resources": [
                "Communication workshops and public speaking practice",
                "Technical presentation opportunities",
                "Writing clear documentation and explanations"
            ]
        },
        {
            "area": "Professional Experience",
            "recommendation": "Gain more diverse experience and build a stronger professional network",
            "priority": "High" if num_questions < 3 else "Medium",
            "resources": [
                "Industry meetups and professional associations",
                "Informational interviews with industry professionals",
                "Contributing to open source projects or professional communities"
            ]
        }
    ]
    
    return {
        "careerPaths": career_paths,
        "workEnvironment": work_environment,
        "development": recommendations,
        "assessment_reliability": {
            "data_points": num_questions,
            "confidence_level": "High" if num_questions >= 5 else "Medium" if num_questions >= 3 else "Low",
            "recommendation": "Complete at least 5 interview questions for a comprehensive assessment" if num_questions < 5 else "Good data foundation for career planning"
        }
    }

# For testing purposes
if __name__ == "__main__":
    # Test with a short, poor answer
    sample_question = "Tell me about a time you had to work under pressure to meet a deadline."
    poor_answer = "I worked hard and met the deadline."
    job_field = "Software Engineering"
    
    print("=== Testing with Poor Answer ===")
    result_poor = get_answer_evaluation(sample_question, poor_answer, job_field)
    print(f"Overall Score: {result_poor['scores']['overall']}/10")
    print(f"Quality Assessment: {result_poor['quality_assessment']}")
    print(json.dumps(result_poor, indent=2))
    
    # Test with a good answer
    good_answer = "Last year, our team needed to deliver a critical customer-facing update within three weeks due to a security vulnerability. I took the lead by first organizing our team into focused workstreams based on expertise areas. I created daily standup meetings to track progress and identify blockers early. Personally, I handled the most complex technical tasks involving database migrations and API updates. We prioritized core security features over nice-to-have enhancements, implemented automated testing to catch regressions quickly, and maintained clear communication with stakeholders about our progress and any trade-offs. Through this systematic approach and team coordination, we delivered the update two days ahead of schedule with all critical security requirements met and zero production issues."
    
    print("\n=== Testing with Good Answer ===")
    result_good = get_answer_evaluation(sample_question, good_answer, job_field)
    print(f"Overall Score: {result_good['scores']['overall']}/10")
    print(f"Quality Assessment: {result_good['quality_assessment']}")
    
    # Test aggregate scoring with single question
    print("\n=== Testing Aggregate Scores (Single Question) ===")
    aggregates = calculate_realistic_aggregate_scores([result_poor])
    print(f"Aggregate Score: {aggregates['overall']}")
    print(f"Data Completeness: {aggregates['data_completeness']}")
    
    # Test skill assessment
    skills = aggregate_realistic_skill_assessment([result_poor])
    print(f"\nSkill Gaps: {[skill['gap_percentage'] for skill in skills['assessed_levels']]}")