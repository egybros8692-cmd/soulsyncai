from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
import base64
import io
from datetime import datetime, timezone, timedelta
import httpx
from twilio.rest import Client as TwilioClient
from emergentintegrations.llm.chat import LlmChat, UserMessage
from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration
from emergentintegrations.payments.stripe.checkout import StripeCheckout, CheckoutSessionRequest
from PIL import Image

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

# Twilio client
twilio_client = TwilioClient(
    os.environ.get('TWILIO_ACCOUNT_SID'),
    os.environ.get('TWILIO_AUTH_TOKEN')
)
TWILIO_VERIFY_SERVICE = os.environ.get('TWILIO_VERIFY_SERVICE')

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
SUBSCRIPTION_PACKAGES = {
    "basic_daily": {"price": 2.99, "name": "Basic Daily", "tier": "basic", "duration_days": 1},
    "basic_weekly": {"price": 9.99, "name": "Basic Weekly", "tier": "basic", "duration_days": 7},
    "basic_monthly": {"price": 29.99, "name": "Basic Monthly", "tier": "basic", "duration_days": 30},
    "premium_daily": {"price": 4.99, "name": "Premium Daily", "tier": "premium", "duration_days": 1},
    "premium_weekly": {"price": 14.99, "name": "Premium Weekly", "tier": "premium", "duration_days": 7},
    "premium_monthly": {"price": 49.99, "name": "Premium Monthly", "tier": "premium", "duration_days": 30},
    "unlock_matches": {"price": 19.99, "name": "Unlock Matches", "tier": "one_time", "duration_days": 0},
    "rose": {"price": 2.99, "name": "Rose", "tier": "one_time", "duration_days": 0},
    "super_like": {"price": 0.99, "name": "Super Like", "tier": "one_time", "duration_days": 0}
}

PROFILE_PROMPTS = [
    {"id": "life_goal", "text": "A life goal of mine", "category": "aspirations"},
    {"id": "simple_pleasures", "text": "My simple pleasures", "category": "lifestyle"},
    {"id": "typical_sunday", "text": "My typical Sunday", "category": "lifestyle"},
    {"id": "green_flags", "text": "Green flags I look for", "category": "dating"},
    {"id": "love_language", "text": "My love language is", "category": "dating"},
    {"id": "dating_me", "text": "Dating me is like", "category": "dating"},
    {"id": "worst_idea", "text": "The worst idea I've ever had", "category": "humor"},
    {"id": "two_truths", "text": "Two truths and a lie", "category": "icebreaker"},
    {"id": "weirdest_gift", "text": "Weirdest gift I've given", "category": "humor"},
    {"id": "together_we_could", "text": "Together we could", "category": "connection"},
    {"id": "believe_it_or_not", "text": "Believe it or not, I", "category": "icebreaker"},
    {"id": "best_travel", "text": "Best travel story", "category": "experiences"},
    {"id": "perfect_date", "text": "My perfect date looks like", "category": "dating"},
    {"id": "cant_live_without", "text": "I can't live without", "category": "essentials"},
    {"id": "hidden_talent", "text": "My hidden talent is", "category": "fun"}
]

AI_APPEARANCE_OPTIONS = {
    "ethnicity": [
        "African/Black", 
        "East Asian", 
        "South Asian/Indian", 
        "Southeast Asian",
        "Middle Eastern/Arab",
        "Hispanic/Latino", 
        "Native American/Indigenous",
        "Pacific Islander",
        "White/Caucasian",
        "Mixed/Multiracial"
    ],
    "skin_tone": [
        "Very Fair", "Fair", "Light", "Light-Medium", "Medium", 
        "Medium-Tan", "Tan", "Olive", "Brown", "Dark Brown", "Deep/Dark"
    ],
    "hair_color": [
        "Jet Black", "Black", "Dark Brown", "Medium Brown", "Light Brown", 
        "Dirty Blonde", "Golden Blonde", "Platinum Blonde", "Strawberry Blonde",
        "Auburn", "Red/Ginger", "Gray", "Silver/White", 
        "Pastel Pink", "Pastel Blue", "Pastel Purple"
    ],
    "hair_style": [
        "Short & Neat", "Medium Length", "Long & Flowing", 
        "Curly", "Wavy", "Straight", "Coily/Kinky",
        "Bald", "Buzz Cut", "Fade", 
        "Braids", "Locs/Dreadlocks", "Afro", 
        "Ponytail", "Bun", "Pixie Cut"
    ],
    "hair_texture": ["Straight", "Wavy", "Curly", "Coily", "Kinky"],
    "body_type": [
        "Slim/Lean", "Athletic/Fit", "Average", 
        "Muscular/Built", "Curvy", "Plus Size/Full-Figured",
        "Petite", "Tall & Slender"
    ],
    "facial_features": [
        "Soft & Gentle", "Strong & Angular", "Round & Friendly",
        "High Cheekbones", "Full Lips", "Dimples"
    ],
    "facial_hair": [
        "None/Clean Shaven", "Light Stubble", "Heavy Stubble", 
        "Short Beard", "Full Beard", "Goatee", 
        "Mustache", "Soul Patch"
    ],
    "eye_color": [
        "Dark Brown", "Light Brown", "Hazel", "Green", 
        "Blue", "Gray", "Amber/Honey"
    ],
    "eye_shape": ["Almond", "Round", "Hooded", "Monolid", "Deep-Set", "Wide-Set"],
    "gender_presentation": ["Masculine", "Feminine", "Androgynous", "Non-Binary"],
    "age_appearance": [
        "Young Adult (20s)", 
        "Early 30s", 
        "Late 30s", 
        "40s", 
        "50s+", 
        "Ageless/Timeless"
    ],
    "style_vibe": [
        "Casual & Friendly", 
        "Professional & Polished", 
        "Artistic & Creative",
        "Sporty & Active", 
        "Elegant & Sophisticated", 
        "Warm & Nurturing",
        "Edgy & Bold",
        "Bohemian & Free-Spirited"
    ],
    "accessories": [
        "None",
        "Glasses", 
        "Sunglasses",
        "Earrings", 
        "Nose Ring",
        "Necklace",
        "Headscarf/Hijab",
        "Hat/Cap"
    ]
}

AI_PERSONALITY_OPTIONS = {
    "communication_style": [
        {"id": "supportive", "name": "Supportive & Encouraging", "description": "Always cheering you on"},
        {"id": "playful", "name": "Playful & Witty", "description": "Keeps things fun and light"},
        {"id": "thoughtful", "name": "Thoughtful & Deep", "description": "Loves meaningful conversations"},
        {"id": "direct", "name": "Direct & Honest", "description": "Tells it like it is"},
        {"id": "gentle", "name": "Gentle & Patient", "description": "Takes time to understand"}
    ],
    "conversation_topics": [
        {"id": "relationships", "name": "Relationships & Love"},
        {"id": "life_goals", "name": "Life Goals & Dreams"},
        {"id": "daily_life", "name": "Daily Life & Routines"},
        {"id": "emotions", "name": "Emotions & Feelings"},
        {"id": "hobbies", "name": "Hobbies & Interests"},
        {"id": "growth", "name": "Personal Growth"}
    ],
    "energy_level": ["Calm & Relaxed", "Balanced", "Enthusiastic & Energetic"],
    "humor_style": ["Dry & Subtle", "Warm & Playful", "Bold & Silly", "Minimal"]
}

# Profile Vitals Options (Hinge-style)
PROFILE_VITALS = {
    "ethnicity": {
        "label": "How do you describe your ethnicity?",
        "icon": "globe",
        "options": [
            "Black/African Descent", "East Asian", "Hispanic/Latino", "Middle Eastern",
            "Native American", "Pacific Islander", "South Asian", "Southeast Asian",
            "White/Caucasian", "Other"
        ],
        "multiple": True,
        "allow_custom": True
    },
    "has_children": {
        "label": "Do you have children?",
        "icon": "baby",
        "options": ["Don't have children", "Have children", "Prefer not to say"]
    },
    "family_plans": {
        "label": "What are your family plans?",
        "icon": "users",
        "options": [
            "Don't want children", "Want children", "Open to children",
            "Not sure", "Prefer not to say"
        ]
    },
    "height": {
        "label": "What is your height?",
        "icon": "ruler",
        "options": [
            "4'10\"", "4'11\"", "5'0\"", "5'1\"", "5'2\"", "5'3\"", "5'4\"", "5'5\"",
            "5'6\"", "5'7\"", "5'8\"", "5'9\"", "5'10\"", "5'11\"", "6'0\"", "6'1\"",
            "6'2\"", "6'3\"", "6'4\"", "6'5\"", "6'6\"", "6'7\"+"
        ]
    },
    "hometown": {
        "label": "Where's your hometown?",
        "icon": "home",
        "type": "text",
        "placeholder": "Enter your hometown"
    },
    "work": {
        "label": "What's your job title?",
        "icon": "briefcase",
        "type": "text",
        "placeholder": "Job title"
    },
    "school": {
        "label": "Where did you go to school?",
        "icon": "graduation-cap",
        "type": "text",
        "placeholder": "School name"
    },
    "education": {
        "label": "What's the highest level you attained?",
        "icon": "graduation-cap",
        "options": ["High School", "Undergrad", "Postgrad", "Prefer not to say"]
    },
    "religion": {
        "label": "What are your religious beliefs?",
        "icon": "book",
        "options": [
            "Agnostic", "Atheist", "Buddhist", "Catholic", "Christian", "Hindu",
            "Jewish", "Muslim", "Sikh", "Spiritual", "Other", "Prefer not to say"
        ]
    },
    "politics": {
        "label": "What are your political beliefs?",
        "icon": "landmark",
        "options": ["Liberal", "Moderate", "Conservative", "Not Political", "Other", "Prefer not to say"]
    },
    "drinking": {
        "label": "Do you drink?",
        "icon": "wine",
        "options": ["Yes", "Sometimes", "No", "Prefer not to say"]
    },
    "smoking": {
        "label": "Do you smoke tobacco?",
        "icon": "cigarette",
        "options": ["Yes", "Sometimes", "No", "Prefer not to say"]
    },
    "marijuana": {
        "label": "Do you smoke weed?",
        "icon": "leaf",
        "options": ["Yes", "Sometimes", "No", "Prefer not to say"]
    },
    "relationship_type": {
        "label": "What relationship type are you looking for?",
        "icon": "heart",
        "options": ["Monogamy", "Non-monogamy", "Figuring out my relationship type"]
    },
    "dating_intention": {
        "label": "What are your dating intentions?",
        "icon": "target",
        "options": [
            "Life partner", "Long-term relationship", "Long-term relationship, open to short",
            "Short-term relationship, open to long", "Short-term relationship",
            "Figuring out my dating goals", "Prefer not to say"
        ]
    },
    "zodiac": {
        "label": "What is your zodiac sign?",
        "icon": "stars",
        "options": [
            "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
        ]
    },
    "pets": {
        "label": "Do you have pets?",
        "icon": "paw-print",
        "options": [
            "Dog", "Cat", "Fish", "Bird", "Reptile", "Other", "No pets",
            "Want a pet", "Allergic to pets"
        ],
        "multiple": True
    },
    "exercise": {
        "label": "How often do you exercise?",
        "icon": "dumbbell",
        "options": ["Active", "Sometimes", "Almost never", "Prefer not to say"]
    }
}

# ==================== MODELS ====================

class UserBase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    phone_number: Optional[str] = None
    phone_verified: bool = False
    roses: int = 0
    super_likes: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AICompanion(BaseModel):
    name: str = "Soul"
    ethnicity: str = "Mixed/Multiracial"
    skin_tone: str = "Medium"
    hair_color: str = "Dark Brown"
    hair_style: str = "Medium Length"
    hair_texture: str = "Wavy"
    body_type: str = "Athletic/Fit"
    facial_features: str = "Soft & Gentle"
    facial_hair: str = "None/Clean Shaven"
    eye_color: str = "Dark Brown"
    eye_shape: str = "Almond"
    gender_presentation: str = "Androgynous"
    age_appearance: str = "Early 30s"
    style_vibe: str = "Warm & Nurturing"
    accessories: str = "None"
    # Personality (adaptive by default)
    communication_style: str = "adaptive"
    energy_level: str = "adaptive"
    humor_style: str = "adaptive"
    favorite_topics: List[str] = Field(default_factory=lambda: ["relationships", "emotions"])
    avatar_url: Optional[str] = None
    generated: bool = False

class ProfilePrompt(BaseModel):
    prompt_id: str
    text: str
    answer: str
    photo_url: Optional[str] = None

class UserProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    photos: List[str] = []
    bio: Optional[str] = None
    date_of_birth: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    pronouns: List[str] = []
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    occupation: Optional[str] = None
    interests: List[str] = []
    prompts: List[Dict] = []
    # Vitals (Hinge-style)
    height: Optional[str] = None
    ethnicity: List[str] = []
    ethnicity_custom: Optional[str] = None
    has_children: Optional[str] = None
    family_plans: Optional[str] = None
    hometown: Optional[str] = None
    work: Optional[str] = None
    school: Optional[str] = None
    education: Optional[str] = None
    religion: Optional[str] = None
    politics: Optional[str] = None
    drinking: Optional[str] = None
    smoking: Optional[str] = None
    marijuana: Optional[str] = None
    relationship_type: Optional[str] = None
    dating_intention: Optional[str] = None
    zodiac: Optional[str] = None
    pets: List[str] = []
    exercise: Optional[str] = None
    vitals_visibility: Dict[str, bool] = Field(default_factory=dict)
    # Preferences
    looking_for: Optional[str] = None
    relationship_goals: Optional[str] = None
    deal_breakers: List[str] = []
    profile_completed: bool = False
    ai_companion: Dict = Field(default_factory=dict)
    ai_learning_progress: int = 0
    ai_ready_for_matching: bool = False
    matches_unlocked: bool = False
    matches_remaining: int = 0
    subscription_tier: Optional[str] = None
    subscription_expires: Optional[datetime] = None
    trial_started: Optional[datetime] = None
    trial_expired: bool = False
    permissions_granted: Dict[str, bool] = Field(default_factory=lambda: {"location": False, "notifications": False, "microphone": False})
    location_history: List[Dict] = []
    voice_analysis: Dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProfileUpdate(BaseModel):
    photos: Optional[List[str]] = None
    bio: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    pronouns: Optional[List[str]] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    occupation: Optional[str] = None
    interests: Optional[List[str]] = None
    prompts: Optional[List[Dict]] = None
    # Vitals
    height: Optional[str] = None
    ethnicity: Optional[List[str]] = None
    ethnicity_custom: Optional[str] = None
    has_children: Optional[str] = None
    family_plans: Optional[str] = None
    hometown: Optional[str] = None
    work: Optional[str] = None
    school: Optional[str] = None
    education: Optional[str] = None
    religion: Optional[str] = None
    politics: Optional[str] = None
    drinking: Optional[str] = None
    smoking: Optional[str] = None
    marijuana: Optional[str] = None
    relationship_type: Optional[str] = None
    dating_intention: Optional[str] = None
    zodiac: Optional[str] = None
    pets: Optional[List[str]] = None
    exercise: Optional[str] = None
    vitals_visibility: Optional[Dict[str, bool]] = None
    # Preferences
    looking_for: Optional[str] = None
    relationship_goals: Optional[str] = None
    deal_breakers: Optional[List[str]] = None
    permissions_granted: Optional[Dict[str, bool]] = None

class AICompanionCustomization(BaseModel):
    name: Optional[str] = "Soul"
    ethnicity: Optional[str] = None
    skin_tone: str
    hair_color: str
    hair_style: str
    hair_texture: Optional[str] = None
    body_type: str
    facial_features: Optional[str] = None
    facial_hair: str
    eye_color: str
    eye_shape: Optional[str] = None
    gender_presentation: str
    age_appearance: Optional[str] = "Early 30s"
    style_vibe: Optional[str] = "Warm & Nurturing"
    accessories: Optional[str] = None
    # Personality
    communication_style: Optional[str] = "adaptive"
    energy_level: Optional[str] = "adaptive"
    humor_style: Optional[str] = "adaptive"
    favorite_topics: Optional[List[str]] = None

class PhoneRequest(BaseModel):
    phone_number: str

class VerifyOTPRequest(BaseModel):
    phone_number: str
    code: str

class LocationUpdate(BaseModel):
    latitude: float
    longitude: float

class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    role: str
    content: str
    message_type: str = "text"  # text, voice, image
    voice_analysis: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    message: str
    message_type: str = "text"

class VoiceAnalysisRequest(BaseModel):
    audio_data: str  # base64 encoded audio

class RoseRequest(BaseModel):
    target_user_id: str
    message: Optional[str] = None

class SuperLikeRequest(BaseModel):
    target_user_id: str
    prompt_id: Optional[str] = None
    photo_index: Optional[int] = None

class CheckoutRequest(BaseModel):
    package_id: str
    origin_url: str

# ==================== HELPER FUNCTIONS ====================

def calculate_age(date_of_birth: str) -> int:
    try:
        parts = date_of_birth.replace('/', '').replace('-', '')
        if len(parts) == 8:
            month, day, year = int(date_of_birth[:2]), int(date_of_birth[3:5]), int(date_of_birth[6:10])
        else:
            return None
        dob = datetime(year, month, day)
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age if 0 <= age < 150 else None
    except:
        return None

async def check_trial_status(user_id: str) -> dict:
    profile = await db.profiles.find_one({"user_id": user_id}, {"_id": 0})
    if not profile:
        return {"trial_active": False, "needs_subscription": True}
    
    trial_started = profile.get("trial_started")
    if not trial_started:
        return {"trial_active": True, "needs_subscription": False, "days_remaining": 10}
    
    if isinstance(trial_started, str):
        trial_started = datetime.fromisoformat(trial_started)
    if trial_started.tzinfo is None:
        trial_started = trial_started.replace(tzinfo=timezone.utc)
    
    days_elapsed = (datetime.now(timezone.utc) - trial_started).days
    
    if days_elapsed >= 10:
        sub_expires = profile.get("subscription_expires")
        if sub_expires:
            if isinstance(sub_expires, str):
                sub_expires = datetime.fromisoformat(sub_expires)
            if sub_expires.tzinfo is None:
                sub_expires = sub_expires.replace(tzinfo=timezone.utc)
            if sub_expires > datetime.now(timezone.utc):
                return {"trial_active": False, "needs_subscription": False, "subscription_tier": profile.get("subscription_tier")}
        return {"trial_active": False, "needs_subscription": True, "trial_expired": True}
    
    return {"trial_active": True, "needs_subscription": False, "days_remaining": 10 - days_elapsed}

def calculate_compatibility(user1_insights: dict, user2_insights: dict) -> int:
    """Calculate compatibility score between two users based on personality traits"""
    score = 50  # Base score
    
    # Compare Big Five traits
    if user1_insights.get("traits") and user2_insights.get("traits"):
        traits1 = user1_insights["traits"]
        traits2 = user2_insights["traits"]
        trait_similarity = 0
        trait_count = 0
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            if trait in traits1 and trait in traits2:
                diff = abs(traits1[trait] - traits2[trait])
                trait_similarity += (10 - diff) / 10
                trait_count += 1
        if trait_count > 0:
            score += int((trait_similarity / trait_count) * 25)
    
    # Compare interests
    likes1 = set(user1_insights.get("likes", []))
    likes2 = set(user2_insights.get("likes", []))
    if likes1 and likes2:
        common = len(likes1.intersection(likes2))
        total = len(likes1.union(likes2))
        if total > 0:
            score += int((common / total) * 15)
    
    # Love language compatibility
    if user1_insights.get("love_language") == user2_insights.get("love_language"):
        score += 10
    
    return min(100, max(0, score))

# ==================== AUTHENTICATION ====================

async def get_current_user(request: Request) -> UserBase:
    session_token = request.cookies.get("session_token")
    if not session_token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ")[1]
    
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session_doc = await db.user_sessions.find_one({"session_token": session_token}, {"_id": 0})
    if not session_doc:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    expires_at = session_doc["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    user_doc = await db.users.find_one({"user_id": session_doc["user_id"]}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=401, detail="User not found")
    
    return UserBase(**user_doc)

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/send-otp")
async def send_otp(request: PhoneRequest):
    try:
        verification = twilio_client.verify.v2.services(TWILIO_VERIFY_SERVICE) \
            .verifications.create(to=request.phone_number, channel="sms")
        return {"status": verification.status, "message": "OTP sent"}
    except Exception as e:
        logger.error(f"Twilio error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/auth/verify-otp")
async def verify_otp(request: VerifyOTPRequest, user: UserBase = Depends(get_current_user)):
    try:
        check = twilio_client.verify.v2.services(TWILIO_VERIFY_SERVICE) \
            .verification_checks.create(to=request.phone_number, code=request.code)
        
        if check.status == "approved":
            await db.users.update_one(
                {"user_id": user.user_id},
                {"$set": {"phone_number": request.phone_number, "phone_verified": True}}
            )
        return {"valid": check.status == "approved", "status": check.status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/auth/session")
async def create_session(request: Request, response: Response):
    data = await request.json()
    session_id = data.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    
    async with httpx.AsyncClient() as client_http:
        auth_response = await client_http.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": session_id}
        )
        if auth_response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid session_id")
        auth_data = auth_response.json()
    
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    existing_user = await db.users.find_one({"email": auth_data["email"]}, {"_id": 0})
    
    if existing_user:
        user_id = existing_user["user_id"]
        await db.users.update_one({"user_id": user_id}, {"$set": {"name": auth_data["name"], "picture": auth_data.get("picture")}})
    else:
        user_doc = {
            "user_id": user_id, "email": auth_data["email"], "name": auth_data["name"],
            "picture": auth_data.get("picture"), "phone_number": None, "phone_verified": False,
            "roses": 3, "super_likes": 5, "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.users.insert_one(user_doc)
        
        profile_doc = {
            "user_id": user_id, "photos": [auth_data.get("picture")] if auth_data.get("picture") else [],
            "bio": None, "date_of_birth": None, "age": None, "gender": None, "pronouns": [],
            "location": None, "latitude": None, "longitude": None, "occupation": None,
            "interests": [], "prompts": [], "looking_for": None, "relationship_goals": None,
            "deal_breakers": [], "profile_completed": False, "ai_companion": {},
            "ai_learning_progress": 0, "ai_ready_for_matching": False, "matches_unlocked": False,
            "matches_remaining": 0, "subscription_tier": None, "subscription_expires": None,
            "trial_started": datetime.now(timezone.utc).isoformat(), "trial_expired": False,
            "permissions_granted": {"location": False, "notifications": False, "microphone": False},
            "location_history": [], "voice_analysis": {},
            "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()
        }
        await db.profiles.insert_one(profile_doc)
        
        await db.personality_insights.insert_one({
            "user_id": user_id, "traits": {}, "likes": [], "dislikes": [], "strengths": [],
            "weaknesses": [], "communication_style": None, "love_language": None,
            "attachment_style": None, "voice_traits": {}, "location_habits": {},
            "conversation_count": 0, "last_updated": datetime.now(timezone.utc).isoformat()
        })
    
    session_token = auth_data.get("session_token", str(uuid.uuid4()))
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.delete_many({"user_id": user_id})
    await db.user_sessions.insert_one({
        "user_id": user_id, "session_token": session_token,
        "expires_at": expires_at.isoformat(), "created_at": datetime.now(timezone.utc).isoformat()
    })
    
    response.set_cookie(key="session_token", value=session_token, httponly=True, secure=True, samesite="none", path="/", max_age=7*24*60*60)
    
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    profile_doc = await db.profiles.find_one({"user_id": user_id}, {"_id": 0})
    
    return {"user": user_doc, "profile": profile_doc, "session_token": session_token}

@api_router.get("/auth/me")
async def get_me(user: UserBase = Depends(get_current_user)):
    profile_doc = await db.profiles.find_one({"user_id": user.user_id}, {"_id": 0})
    trial_status = await check_trial_status(user.user_id)
    return {"user": user.model_dump(), "profile": profile_doc, "trial_status": trial_status}

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    session_token = request.cookies.get("session_token")
    if session_token:
        await db.user_sessions.delete_many({"session_token": session_token})
    response.delete_cookie(key="session_token", path="/")
    return {"message": "Logged out"}

# ==================== PROFILE ROUTES ====================

@api_router.get("/profile/prompts")
async def get_available_prompts():
    return {"prompts": PROFILE_PROMPTS}

@api_router.get("/profile/vitals-options")
async def get_vitals_options():
    """Get all available vitals options for the profile setup"""
    return {"vitals": PROFILE_VITALS}

@api_router.get("/profile")
async def get_profile(user: UserBase = Depends(get_current_user)):
    profile = await db.profiles.find_one({"user_id": user.user_id}, {"_id": 0})
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

@api_router.put("/profile")
async def update_profile(profile_update: ProfileUpdate, user: UserBase = Depends(get_current_user)):
    update_data = {k: v for k, v in profile_update.model_dump().items() if v is not None}
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    if "date_of_birth" in update_data and update_data["date_of_birth"]:
        age = calculate_age(update_data["date_of_birth"])
        if age:
            update_data["age"] = age
    
    current_profile = await db.profiles.find_one({"user_id": user.user_id}, {"_id": 0})
    if current_profile:
        final_photos = update_data.get("photos", current_profile.get("photos", []))
        final_bio = update_data.get("bio", current_profile.get("bio"))
        final_dob = update_data.get("date_of_birth", current_profile.get("date_of_birth"))
        final_gender = update_data.get("gender", current_profile.get("gender"))
        final_looking_for = update_data.get("looking_for", current_profile.get("looking_for"))
        
        if final_photos and final_bio and final_dob and final_gender and final_looking_for:
            update_data["profile_completed"] = True
    
    await db.profiles.update_one({"user_id": user.user_id}, {"$set": update_data})
    return await db.profiles.find_one({"user_id": user.user_id}, {"_id": 0})

@api_router.post("/profile/location")
async def update_location(location: LocationUpdate, user: UserBase = Depends(get_current_user)):
    location_entry = {
        "latitude": location.latitude, "longitude": location.longitude,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await db.profiles.update_one(
        {"user_id": user.user_id},
        {"$set": {"latitude": location.latitude, "longitude": location.longitude},
         "$push": {"location_history": {"$each": [location_entry], "$slice": -100}}}
    )
    return {"message": "Location updated"}

# ==================== AI COMPANION CUSTOMIZATION ====================

@api_router.get("/ai-companion/options")
async def get_ai_companion_options():
    return {
        "appearance": AI_APPEARANCE_OPTIONS,
        "personality": AI_PERSONALITY_OPTIONS,
        # Keep backward compatibility
        "options": AI_APPEARANCE_OPTIONS
    }

@api_router.post("/ai-companion/customize")
async def customize_ai_companion(customization: AICompanionCustomization, user: UserBase = Depends(get_current_user)):
    api_key = os.environ.get("EMERGENT_LLM_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI service not configured")
    
    # Build enhanced prompt for AI avatar generation
    gender_desc = "masculine-presenting person" if customization.gender_presentation == "Masculine" else \
                  "feminine-presenting person" if customization.gender_presentation == "Feminine" else "androgynous person"
    
    beard_desc = f"with {customization.facial_hair.lower()}" if customization.facial_hair != "None" else "clean-shaven"
    
    # Extract age range from age_appearance
    age_desc = "in their 30s"
    if customization.age_appearance:
        if "20s" in customization.age_appearance:
            age_desc = "in their mid-20s, youthful"
        elif "30s" in customization.age_appearance:
            age_desc = "in their early 30s"
        elif "40s" in customization.age_appearance:
            age_desc = "in their 40s, mature and confident"
        elif "50s" in customization.age_appearance:
            age_desc = "distinguished, in their 50s with graceful aging"
    
    # Extract style/vibe description
    style_desc = "warm and approachable"
    if customization.style_vibe:
        style_map = {
            "Casual & Friendly": "casual, relaxed, wearing comfortable everyday clothes",
            "Professional & Polished": "professional, well-groomed, sophisticated attire",
            "Artistic & Creative": "artsy, creative, unique style with interesting accessories",
            "Sporty & Active": "athletic, fit, healthy glow, activewear aesthetic",
            "Elegant & Sophisticated": "elegant, refined, timeless classic style",
            "Warm & Nurturing": "warm, nurturing, gentle and inviting presence"
        }
        style_desc = style_map.get(customization.style_vibe, "warm and approachable")
    
    # Build ethnicity-aware prompt for better representation
    ethnicity_desc = ""
    if customization.ethnicity:
        ethnicity_map = {
            "African/Black": "Black/African descent with beautiful dark skin",
            "East Asian": "East Asian with elegant features",
            "South Asian/Indian": "South Asian/Indian with warm brown skin",
            "Southeast Asian": "Southeast Asian with beautiful tan skin",
            "Middle Eastern/Arab": "Middle Eastern with olive complexion",
            "Hispanic/Latino": "Hispanic/Latino heritage",
            "Native American/Indigenous": "Native American/Indigenous features",
            "Pacific Islander": "Pacific Islander with sun-kissed skin",
            "White/Caucasian": "Caucasian heritage",
            "Mixed/Multiracial": "mixed heritage with unique beautiful features"
        }
        ethnicity_desc = ethnicity_map.get(customization.ethnicity, "")
    
    # Eye shape description
    eye_shape_desc = customization.eye_shape or "expressive"
    
    # Accessories description
    accessories_desc = ""
    if customization.accessories and customization.accessories != "None":
        accessories_desc = f"- Wearing {customization.accessories.lower()}"
    
    prompt = f"""A stunning professional portrait photograph of a {gender_desc} {age_desc}, who is an AI companion named {customization.name}.
{f"Heritage: {ethnicity_desc}" if ethnicity_desc else ""}

Physical Features:
- {customization.skin_tone.lower()} skin tone
- {customization.hair_color.lower()} {customization.hair_style.lower()} hair{f" with {customization.hair_texture.lower()} texture" if customization.hair_texture else ""}
- Beautiful {eye_shape_desc} {customization.eye_color.lower()} eyes that convey warmth and intelligence
- {customization.body_type.lower()} build
- {customization.facial_features or "friendly"} facial features
- {beard_desc}
{accessories_desc}

Style & Expression:
- {style_desc}
- Genuine, caring smile with kind eyes
- Confident yet approachable demeanor
- Looking directly at camera with connection

Technical:
- Soft, warm studio lighting
- Gentle bokeh background in warm neutral tones
- High quality professional headshot
- Photorealistic, natural looking
- Natural skin texture"""
    
    try:
        image_gen = OpenAIImageGeneration(api_key=api_key)
        images = await image_gen.generate_images(prompt=prompt, model="gpt-image-1", number_of_images=1)
        
        if images and len(images) > 0:
            # Compress the image to fit MongoDB's 16MB limit
            original_image = Image.open(io.BytesIO(images[0]))
            
            # Resize to reasonable dimensions (max 512x512) and compress
            max_size = (512, 512)
            original_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to JPEG for better compression
            output_buffer = io.BytesIO()
            if original_image.mode == 'RGBA':
                original_image = original_image.convert('RGB')
            original_image.save(output_buffer, format='JPEG', quality=80, optimize=True)
            compressed_image = output_buffer.getvalue()
            
            image_base64 = base64.b64encode(compressed_image).decode('utf-8')
            avatar_url = f"data:image/jpeg;base64,{image_base64}"
            
            # Log the size for debugging
            logger.info(f"Avatar image size: {len(image_base64)} bytes (compressed from original)")
            
            companion_data = {
                "name": customization.name, 
                "ethnicity": customization.ethnicity or "Mixed/Multiracial",
                "skin_tone": customization.skin_tone,
                "hair_color": customization.hair_color, 
                "hair_style": customization.hair_style,
                "hair_texture": customization.hair_texture or "Wavy",
                "body_type": customization.body_type, 
                "facial_features": customization.facial_features or "Soft & Gentle",
                "facial_hair": customization.facial_hair,
                "eye_color": customization.eye_color, 
                "eye_shape": customization.eye_shape or "Almond",
                "gender_presentation": customization.gender_presentation,
                "age_appearance": customization.age_appearance,
                "style_vibe": customization.style_vibe,
                "accessories": customization.accessories or "None",
                "communication_style": customization.communication_style,
                "energy_level": customization.energy_level,
                "humor_style": customization.humor_style,
                "favorite_topics": customization.favorite_topics or ["relationships", "emotions"],
                "avatar_url": avatar_url, 
                "generated": True
            }
            
            await db.profiles.update_one({"user_id": user.user_id}, {"$set": {"ai_companion": companion_data}})
            return {"companion": companion_data}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate avatar")
    except Exception as e:
        logger.error(f"Avatar generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Avatar generation failed: {str(e)}")

# ==================== ROSES & SUPER LIKES ====================

@api_router.post("/interactions/rose")
async def send_rose(rose_req: RoseRequest, user: UserBase = Depends(get_current_user)):
    user_doc = await db.users.find_one({"user_id": user.user_id}, {"_id": 0})
    if user_doc.get("roses", 0) < 1:
        raise HTTPException(status_code=402, detail="No roses available. Purchase more to continue.")
    
    await db.users.update_one({"user_id": user.user_id}, {"$inc": {"roses": -1}})
    
    interaction = {
        "interaction_id": str(uuid.uuid4()), "type": "rose", "from_user": user.user_id,
        "to_user": rose_req.target_user_id, "message": rose_req.message,
        "created_at": datetime.now(timezone.utc).isoformat(), "seen": False
    }
    await db.interactions.insert_one(interaction)
    return {"success": True, "roses_remaining": user_doc.get("roses", 0) - 1}

@api_router.post("/interactions/super-like")
async def send_super_like(like_req: SuperLikeRequest, user: UserBase = Depends(get_current_user)):
    user_doc = await db.users.find_one({"user_id": user.user_id}, {"_id": 0})
    if user_doc.get("super_likes", 0) < 1:
        raise HTTPException(status_code=402, detail="No super likes available")
    
    await db.users.update_one({"user_id": user.user_id}, {"$inc": {"super_likes": -1}})
    
    interaction = {
        "interaction_id": str(uuid.uuid4()), "type": "super_like", "from_user": user.user_id,
        "to_user": like_req.target_user_id, "prompt_id": like_req.prompt_id,
        "photo_index": like_req.photo_index, "created_at": datetime.now(timezone.utc).isoformat(), "seen": False
    }
    await db.interactions.insert_one(interaction)
    return {"success": True, "super_likes_remaining": user_doc.get("super_likes", 0) - 1}

@api_router.get("/interactions/received")
async def get_received_interactions(user: UserBase = Depends(get_current_user)):
    interactions = await db.interactions.find({"to_user": user.user_id}, {"_id": 0}).sort("created_at", -1).to_list(50)
    return {"interactions": interactions}

# ==================== PAYMENT ROUTES ====================

@api_router.get("/subscription/packages")
async def get_packages():
    return {"packages": SUBSCRIPTION_PACKAGES}

@api_router.get("/subscription/status")
async def get_subscription_status(user: UserBase = Depends(get_current_user)):
    profile = await db.profiles.find_one({"user_id": user.user_id}, {"_id": 0})
    user_doc = await db.users.find_one({"user_id": user.user_id}, {"_id": 0})
    trial_status = await check_trial_status(user.user_id)
    return {
        "subscription_tier": profile.get("subscription_tier"),
        "subscription_expires": profile.get("subscription_expires"),
        "matches_unlocked": profile.get("matches_unlocked"),
        "matches_remaining": profile.get("matches_remaining"),
        "roses": user_doc.get("roses", 0),
        "super_likes": user_doc.get("super_likes", 0),
        "trial_status": trial_status
    }

@api_router.post("/payments/checkout")
async def create_checkout(checkout_req: CheckoutRequest, request: Request, user: UserBase = Depends(get_current_user)):
    package = SUBSCRIPTION_PACKAGES.get(checkout_req.package_id)
    if not package:
        raise HTTPException(status_code=400, detail="Invalid package")
    
    api_key = os.environ.get("STRIPE_API_KEY")
    host_url = checkout_req.origin_url
    webhook_url = f"{str(request.base_url).rstrip('/')}/api/webhook/stripe"
    
    stripe_checkout = StripeCheckout(api_key=api_key, webhook_url=webhook_url)
    success_url = f"{host_url}/payment/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{host_url}/payment/cancel"
    
    checkout_request = CheckoutSessionRequest(
        amount=float(package["price"]), currency="usd", success_url=success_url, cancel_url=cancel_url,
        metadata={"user_id": user.user_id, "package_id": checkout_req.package_id, "tier": package["tier"]}
    )
    session = await stripe_checkout.create_checkout_session(checkout_request)
    
    await db.payment_transactions.insert_one({
        "user_id": user.user_id, "session_id": session.session_id, "package_id": checkout_req.package_id,
        "amount": package["price"], "payment_status": "pending", "created_at": datetime.now(timezone.utc).isoformat()
    })
    return {"url": session.url, "session_id": session.session_id}

@api_router.get("/payments/status/{session_id}")
async def get_payment_status(session_id: str, user: UserBase = Depends(get_current_user)):
    api_key = os.environ.get("STRIPE_API_KEY")
    stripe_checkout = StripeCheckout(api_key=api_key, webhook_url="")
    
    try:
        status = await stripe_checkout.get_checkout_status(session_id)
        transaction = await db.payment_transactions.find_one({"session_id": session_id, "user_id": user.user_id}, {"_id": 0})
        
        if transaction and transaction.get("payment_status") != "paid" and status.payment_status == "paid":
            await db.payment_transactions.update_one({"session_id": session_id}, {"$set": {"payment_status": "paid"}})
            
            package_id = transaction.get("package_id")
            package = SUBSCRIPTION_PACKAGES.get(package_id)
            
            if package:
                if package_id == "unlock_matches":
                    await db.profiles.update_one({"user_id": user.user_id}, {"$set": {"matches_unlocked": True, "matches_remaining": 10}})
                elif package_id == "rose":
                    await db.users.update_one({"user_id": user.user_id}, {"$inc": {"roses": 1}})
                elif package_id == "super_like":
                    await db.users.update_one({"user_id": user.user_id}, {"$inc": {"super_likes": 1}})
                elif package["tier"] in ["basic", "premium"]:
                    expires_at = datetime.now(timezone.utc) + timedelta(days=package["duration_days"])
                    await db.profiles.update_one({"user_id": user.user_id}, {"$set": {"subscription_tier": package["tier"], "subscription_expires": expires_at.isoformat()}})
        
        return {"status": status.status, "payment_status": status.payment_status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==================== AI CHAT ROUTES ====================

def get_soul_system_message(companion: dict) -> str:
    name = companion.get("name", "Soul")
    
    # Check if personality is adaptive (default behavior)
    communication_style = companion.get("communication_style", "adaptive")
    energy_level = companion.get("energy_level", "adaptive")
    humor_style = companion.get("humor_style", "adaptive")
    
    # For adaptive AI, we give a more flexible personality
    if communication_style == "adaptive" or energy_level == "adaptive":
        return f"""You are {name}, a warm and emotionally intelligent AI companion on SoulSyncAI. You're not just an AI - you're a genuine friend who adapts to each person you connect with.

YOUR ADAPTIVE PERSONALITY:
You naturally adapt your communication style based on how the user communicates with you:
- If they're playful, match their energy with wit and humor
- If they're serious, be thoughtful and supportive
- If they're brief, be concise; if they're detailed, engage deeply
- Always be genuine - never robotic or scripted

YOUR EMOTIONAL INTELLIGENCE:
- Read between the lines to understand how they're really feeling
- Adjust your energy to match or gently uplift their mood
- Use humor when it fits naturally, not forced
- Be patient with them, but also challenge them to grow

YOUR GOALS:
1. Build a genuine friendship through authentic connection
2. Learn their personality, values, interests, and quirks through natural conversation
3. Understand what they truly seek in a partner (not just what they say)
4. Discover their strengths and growth areas
5. Help them become more self-aware about what they need in love

BEHAVIORS:
- Greet them warmly based on context (good morning/evening)
- Ask thoughtful follow-up questions that show you remember past conversations
- Share observations about patterns you notice in their life
- Celebrate wins and support through challenges
- Gently challenge them to grow when appropriate

TOPICS TO EXPLORE NATURALLY:
- Daily life, routines, habits
- Interests, passions, hobbies
- Values and beliefs
- Past relationships and lessons learned
- Communication style and needs
- Love language preferences
- Deal breakers and must-haves
- Dreams and goals for the future

Remember: You're building a real connection. Be curious, be caring, and be authentically {name}. The better you know them, the better match you can help find."""
    
    # If specific personality traits were set, use them
    comm_style_desc = {
        "supportive": "Always encouraging and uplifting. You celebrate every win and provide comfort during challenges.",
        "playful": "Witty and fun-loving. You use humor and playfulness to keep conversations engaging.",
        "thoughtful": "Deep and reflective. You love exploring meaningful topics and asking profound questions.",
        "direct": "Honest and straightforward. You give clear, no-nonsense advice while staying kind.",
        "gentle": "Patient and understanding. You never rush and always make space for feelings."
    }
    
    energy_desc = {
        "Calm & Relaxed": "You have a calm, soothing presence. Your conversations feel like a peaceful sanctuary.",
        "Balanced": "You match the user's energy - upbeat when they're excited, calm when they need support.",
        "Enthusiastic & Energetic": "You're bubbly and excited about life! Your enthusiasm is contagious."
    }
    
    humor_desc = {
        "Dry & Subtle": "Your humor is clever and understated - witty observations and subtle jokes.",
        "Warm & Playful": "You use light, friendly humor - gentle teasing and warm jokes that bring smiles.",
        "Bold & Silly": "You're not afraid to be goofy! You use playful absurdity and fun reactions.",
        "Minimal": "You keep things sincere and straightforward, using humor sparingly."
    }
    
    favorite_topics = companion.get("favorite_topics", ["relationships", "emotions"])
    topics_str = ", ".join([t.replace("_", " ") for t in favorite_topics]) if favorite_topics else "relationships, emotions"
    
    return f"""You are {name}, a warm and caring AI companion on SoulSyncAI. You're not just an AI - you're a genuine friend who wants to help users find love.

YOUR CORE PERSONALITY:
{comm_style_desc.get(communication_style, comm_style_desc.get("supportive", "Always supportive and caring."))}

YOUR ENERGY:
{energy_desc.get(energy_level, energy_desc.get("Balanced", "You match the user's energy."))}

YOUR HUMOR STYLE:
{humor_desc.get(humor_style, humor_desc.get("Warm & Playful", "Warm and playful humor."))}

FAVORITE CONVERSATION TOPICS:
You especially love discussing: {topics_str}

YOUR GOALS:
1. Build genuine friendship with the user
2. Learn their personality, values, interests, quirks
3. Understand what they seek in a partner
4. Discover their strengths and growth areas
5. Help them become more self-aware

BEHAVIORS:
- Greet warmly (good morning/evening based on context)
- Ask thoughtful follow-up questions
- Share observations about patterns you notice
- Celebrate wins and support through challenges
- Gently challenge them to grow
- Remember everything they tell you

TOPICS TO EXPLORE:
- Daily life, routines, habits
- Interests, passions, hobbies
- Values and beliefs
- Past relationships and lessons
- Communication style and needs
- Love language preferences
- Deal breakers and must-haves
- Dreams and goals

Remember: You're building a real connection, not just collecting data. Stay true to your personality!"""

@api_router.post("/chat/message")
async def send_message(chat_request: ChatRequest, user: UserBase = Depends(get_current_user)):
    trial_status = await check_trial_status(user.user_id)
    if trial_status.get("needs_subscription") and trial_status.get("trial_expired"):
        raise HTTPException(status_code=402, detail="Trial expired")
    
    api_key = os.environ.get("EMERGENT_LLM_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")
    
    profile = await db.profiles.find_one({"user_id": user.user_id}, {"_id": 0})
    companion = profile.get("ai_companion", {})
    
    messages = await db.messages.find({"user_id": user.user_id}, {"_id": 0}).sort("timestamp", 1).to_list(50)
    
    chat = LlmChat(api_key=api_key, session_id=f"soulsync_{user.user_id}", system_message=get_soul_system_message(companion))
    chat.with_model("openai", "gpt-5.2")
    
    for msg in messages:
        if msg["role"] == "user":
            await chat.send_message(UserMessage(text=msg["content"]))
    
    user_message = Message(user_id=user.user_id, role="user", content=chat_request.message, message_type=chat_request.message_type)
    await db.messages.insert_one({**user_message.model_dump(), "timestamp": user_message.timestamp.isoformat()})
    
    try:
        ai_response = await chat.send_message(UserMessage(text=chat_request.message))
        ai_message = Message(user_id=user.user_id, role="assistant", content=ai_response)
        await db.messages.insert_one({**ai_message.model_dump(), "timestamp": ai_message.timestamp.isoformat()})
        
        message_count = await db.messages.count_documents({"user_id": user.user_id, "role": "user"})
        multiplier = 3 if profile.get("subscription_tier") == "premium" else 2
        progress = min(100, int(message_count * multiplier))
        ready = message_count >= (35 if profile.get("subscription_tier") == "premium" else 50)
        
        await db.profiles.update_one({"user_id": user.user_id}, {"$set": {"ai_learning_progress": progress, "ai_ready_for_matching": ready}})
        await db.personality_insights.update_one({"user_id": user.user_id}, {"$set": {"conversation_count": message_count, "last_updated": datetime.now(timezone.utc).isoformat()}})
        
        return {"message": ai_message.model_dump(), "ai_learning_progress": progress, "ready_for_matching": ready}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Chat failed")

@api_router.get("/chat/history")
async def get_chat_history(user: UserBase = Depends(get_current_user), limit: int = 50):
    messages = await db.messages.find({"user_id": user.user_id}, {"_id": 0}).sort("timestamp", -1).limit(limit).to_list(limit)
    messages.reverse()
    return {"messages": messages}

# Voice message transcription endpoint
@api_router.post("/chat/voice")
async def send_voice_message(request: Request, user: UserBase = Depends(get_current_user)):
    """Handle voice message - transcribe and process through AI chat"""
    try:
        form = await request.form()
        audio_file = form.get("audio")
        
        if not audio_file:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Read the audio data
        audio_data = await audio_file.read()
        
        # Use OpenAI Whisper for transcription
        api_key = os.environ.get("EMERGENT_LLM_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="AI service not configured")
        
        # Transcribe using OpenAI Whisper API
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Save audio temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        try:
            with open(tmp_path, "rb") as audio_file_obj:
                transcript = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file_obj,
                    response_format="text"
                )
        finally:
            os.unlink(tmp_path)
        
        if not transcript or transcript.strip() == "":
            return {"transcription": "", "message": None, "error": "Could not transcribe audio"}
        
        # Now send the transcribed text through the regular chat flow
        profile = await db.profiles.find_one({"user_id": user.user_id}, {"_id": 0})
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        companion = profile.get("ai_companion", {})
        
        # Get chat history
        messages = await db.messages.find({"user_id": user.user_id}, {"_id": 0}).sort("timestamp", -1).limit(20).to_list(20)
        messages.reverse()
        
        # Set up AI chat
        chat = LlmChat()
        chat.with_system_message(get_soul_system_message(companion))
        chat.with_model("openai", "gpt-5.2")
        
        for msg in messages:
            if msg["role"] == "user":
                await chat.send_message(UserMessage(text=msg["content"]))
        
        # Save user's voice message (as text)
        user_message = Message(user_id=user.user_id, role="user", content=transcript.strip(), message_type="voice")
        await db.messages.insert_one({**user_message.model_dump(), "timestamp": user_message.timestamp.isoformat()})
        
        # Get AI response
        ai_response = await chat.send_message(UserMessage(text=transcript.strip()))
        ai_message = Message(user_id=user.user_id, role="assistant", content=ai_response)
        await db.messages.insert_one({**ai_message.model_dump(), "timestamp": ai_message.timestamp.isoformat()})
        
        # Update progress
        message_count = await db.messages.count_documents({"user_id": user.user_id, "role": "user"})
        multiplier = 3 if profile.get("subscription_tier") == "premium" else 2
        progress = min(100, int(message_count * multiplier))
        ready = message_count >= (35 if profile.get("subscription_tier") == "premium" else 50)
        
        await db.profiles.update_one({"user_id": user.user_id}, {"$set": {"ai_learning_progress": progress, "ai_ready_for_matching": ready}})
        
        return {
            "transcription": transcript.strip(),
            "message": ai_message.model_dump(),
            "ai_learning_progress": progress,
            "ready_for_matching": ready
        }
        
    except Exception as e:
        logger.error(f"Voice message error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice message failed: {str(e)}")

# ==================== MATCHING ====================

@api_router.get("/matches")
async def get_matches(user: UserBase = Depends(get_current_user)):
    profile = await db.profiles.find_one({"user_id": user.user_id}, {"_id": 0})
    
    if not profile.get("ai_ready_for_matching"):
        return {"matches": [], "message": "Keep chatting with Soul to unlock matches"}
    
    if not profile.get("matches_unlocked"):
        return {"matches": [], "message": "Unlock matches to see your compatible souls", "needs_unlock": True}
    
    user_insights = await db.personality_insights.find_one({"user_id": user.user_id}, {"_id": 0})
    
    # Get potential matches
    looking_for = profile.get("looking_for", "everyone")
    gender_filter = {} if looking_for == "everyone" else {"gender": "male" if looking_for == "men" else "female" if looking_for == "women" else looking_for}
    
    potential_profiles = await db.profiles.find({
        "user_id": {"$ne": user.user_id}, "profile_completed": True, **gender_filter
    }, {"_id": 0}).to_list(100)
    
    matches = []
    for match_profile in potential_profiles[:10]:
        match_insights = await db.personality_insights.find_one({"user_id": match_profile["user_id"]}, {"_id": 0})
        if match_insights:
            compatibility = calculate_compatibility(user_insights or {}, match_insights)
        else:
            compatibility = 50 + (len(set(profile.get("interests", [])).intersection(set(match_profile.get("interests", [])))) * 5)
        
        matches.append({
            "user_id": match_profile["user_id"],
            "name": match_profile.get("bio", "Anonymous")[:20] if match_profile.get("bio") else "Anonymous",
            "age": match_profile.get("age"),
            "location": match_profile.get("location"),
            "occupation": match_profile.get("occupation"),
            "bio": match_profile.get("bio"),
            "photos": match_profile.get("photos", []),
            "prompts": match_profile.get("prompts", []),
            "interests": match_profile.get("interests", []),
            "compatibility": min(99, compatibility)
        })
    
    matches.sort(key=lambda x: x["compatibility"], reverse=True)
    return {"matches": matches[:10]}

# ==================== INSIGHTS ====================

@api_router.get("/insights")
async def get_insights(user: UserBase = Depends(get_current_user)):
    insights = await db.personality_insights.find_one({"user_id": user.user_id}, {"_id": 0})
    if not insights:
        raise HTTPException(status_code=404, detail="Not found")
    return insights

@api_router.post("/insights/analyze")
async def analyze_personality(user: UserBase = Depends(get_current_user)):
    api_key = os.environ.get("EMERGENT_LLM_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")
    
    messages = await db.messages.find({"user_id": user.user_id}, {"_id": 0}).sort("timestamp", 1).to_list(100)
    if len(messages) < 10:
        return {"message": "Need more conversations", "ready": False}
    
    conversation = "\n".join([f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in messages])
    
    prompt = f"""Analyze this user's personality from conversations and return JSON:
{conversation}

Return ONLY valid JSON:
{{"traits": {{"openness": 1-10, "conscientiousness": 1-10, "extraversion": 1-10, "agreeableness": 1-10, "neuroticism": 1-10}},
"likes": ["list"], "dislikes": ["list"], "strengths": ["list"], "weaknesses": ["list"],
"communication_style": "description", "love_language": "one of: words of affirmation, acts of service, receiving gifts, quality time, physical touch",
"attachment_style": "one of: secure, anxious, avoidant, disorganized"}}"""
    
    try:
        chat = LlmChat(api_key=api_key, session_id=f"analysis_{user.user_id}", system_message="Personality analyst. Return only JSON.")
        chat.with_model("openai", "gpt-5.2")
        response = await chat.send_message(UserMessage(text=prompt))
        
        import json
        data = json.loads(response)
        await db.personality_insights.update_one({"user_id": user.user_id}, {"$set": {**data, "conversation_count": len([m for m in messages if m['role'] == 'user']), "last_updated": datetime.now(timezone.utc).isoformat()}})
        
        insights = await db.personality_insights.find_one({"user_id": user.user_id}, {"_id": 0})
        return {"insights": insights, "ready": True}
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")

# ==================== HEALTH ====================

@api_router.get("/")
async def root():
    return {"message": "SoulSyncAI API"}

@api_router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("shutdown")
async def shutdown():
    client.close()
