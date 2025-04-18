# firebase_utils.py
import json
import os
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, storage
import streamlit as st

def initialize_firebase():
    """Initialize Firebase with proper error handling"""
    try:
        # Exit if already initialized
        if firebase_admin._apps:
            return True
        
        # Load credentials from Streamlit secrets
        if not hasattr(st, 'secrets') or "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in st.secrets:
            print("No Firebase credentials found in Streamlit secrets")
            return False
        
        # Parse credentials JSON
        cred_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
        if isinstance(cred_json, str):
            cred_dict = json.loads(cred_json)
        else:
            cred_dict = cred_json
        
        # Initialize Firebase
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'durable-stack-453203-c6.appspot.com'
        })
        
        # Verify bucket access
        bucket = storage.bucket()
        print(f"Firebase initialized with bucket: {bucket.name}")
        return True
            
    except Exception as e:
        print(f"Firebase initialization error: {str(e)}")
        return False

def upload_file_to_firebase(file_path, destination_path=None):
    """Upload a file to Firebase Storage"""
    try:
        # Initialize Firebase if needed
        if not firebase_admin._apps:
            if not initialize_firebase():
                return None
        
        # Get bucket and upload
        bucket = storage.bucket()
        if not destination_path:
            destination_path = Path(file_path).name
        
        blob = bucket.blob(destination_path)
        blob.upload_from_filename(str(file_path))
        blob.make_public()
        return blob.public_url
        
    except Exception as e:
        print(f"Firebase upload error: {str(e)}")
        return None