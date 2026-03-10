"""
Firebase Authentication Module
Verifies Firebase ID tokens and extracts user info.
"""

import os
import json
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from fastapi import Request, HTTPException
from database import upsert_user

# ── Initialize Firebase Admin ────────────────────────────────
firebase_cred_env = os.getenv("FIREBASE_CREDENTIALS")

try:
    if firebase_cred_env:
        # Load from environment variable (Render)
        cred_dict = json.loads(firebase_cred_env)
        _cred = credentials.Certificate(cred_dict)
    else:
        # Load from local file (Local Development)
        _cred_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "firebase-service-account.json"
        )
        _cred = credentials.Certificate(_cred_path)

    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(_cred, {
            "storageBucket": "ai-research-copilot-511d7.firebasestorage.app"
        })
except Exception as e:
    print(f"Warning: Failed to initialize Firebase: {e}")


async def get_current_user(request: Request) -> dict:
    """
    FastAPI dependency that extracts and verifies the Firebase ID token
    from the Authorization header. Returns user info dict.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split("Bearer ")[1]

    try:
        decoded = firebase_auth.verify_id_token(token)
        uid = decoded["uid"]
        email = decoded.get("email", "")
        name = decoded.get("name", decoded.get("email", "User"))

        # Automatically insert/update the user in our local database
        # to ensure foreign key constraints (like research_history) don't fail
        upsert_user(uid, email, name)

        return {
            "uid": uid,
            "email": email,
            "name": name,
        }
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")
