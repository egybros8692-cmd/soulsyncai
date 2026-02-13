"""
Backend API Tests for SoulSyncAI - Vitals Feature
Tests the new vitals options endpoint and profile update with vitals data
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestHealthEndpoint:
    """Health check endpoint tests"""
    
    def test_health_returns_200(self):
        """Test health endpoint returns healthy status"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        print("✅ Health endpoint returns 200 with healthy status")


class TestVitalsOptionsEndpoint:
    """Tests for GET /api/profile/vitals-options endpoint"""
    
    def test_vitals_options_returns_200(self):
        """Test vitals-options endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        assert response.status_code == 200
        print("✅ Vitals options endpoint returns 200")
    
    def test_vitals_options_contains_ethnicity(self):
        """Test vitals options contains ethnicity with multi-select and custom option"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        data = response.json()
        
        assert "vitals" in data
        assert "ethnicity" in data["vitals"]
        
        ethnicity = data["vitals"]["ethnicity"]
        assert ethnicity["label"] == "How do you describe your ethnicity?"
        assert ethnicity["multiple"] == True
        assert ethnicity["allow_custom"] == True
        assert "Black/African Descent" in ethnicity["options"]
        assert "East Asian" in ethnicity["options"]
        assert "Hispanic/Latino" in ethnicity["options"]
        assert "White/Caucasian" in ethnicity["options"]
        print("✅ Ethnicity vital has correct structure with multi-select and custom option")
    
    def test_vitals_options_contains_has_children(self):
        """Test vitals options contains has_children"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        data = response.json()
        
        assert "has_children" in data["vitals"]
        has_children = data["vitals"]["has_children"]
        assert has_children["label"] == "Do you have children?"
        assert "Don't have children" in has_children["options"]
        assert "Have children" in has_children["options"]
        print("✅ Has children vital has correct structure")
    
    def test_vitals_options_contains_family_plans(self):
        """Test vitals options contains family_plans"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        data = response.json()
        
        assert "family_plans" in data["vitals"]
        family_plans = data["vitals"]["family_plans"]
        assert family_plans["label"] == "What are your family plans?"
        assert "Don't want children" in family_plans["options"]
        assert "Want children" in family_plans["options"]
        assert "Open to children" in family_plans["options"]
        print("✅ Family plans vital has correct structure")
    
    def test_vitals_options_contains_hometown(self):
        """Test vitals options contains hometown (text input)"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        data = response.json()
        
        assert "hometown" in data["vitals"]
        hometown = data["vitals"]["hometown"]
        assert hometown["label"] == "Where's your hometown?"
        assert hometown["type"] == "text"
        assert "placeholder" in hometown
        print("✅ Hometown vital has correct text input structure")
    
    def test_vitals_options_contains_work(self):
        """Test vitals options contains work (text input)"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        data = response.json()
        
        assert "work" in data["vitals"]
        work = data["vitals"]["work"]
        assert work["label"] == "Where do you work?"
        assert work["type"] == "text"
        print("✅ Work vital has correct text input structure")
    
    def test_vitals_options_contains_drinking(self):
        """Test vitals options contains drinking"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        data = response.json()
        
        assert "drinking" in data["vitals"]
        drinking = data["vitals"]["drinking"]
        assert drinking["label"] == "Do you drink?"
        assert "Yes" in drinking["options"]
        assert "Sometimes" in drinking["options"]
        assert "No" in drinking["options"]
        print("✅ Drinking vital has correct structure")
    
    def test_vitals_options_contains_smoking(self):
        """Test vitals options contains smoking"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        data = response.json()
        
        assert "smoking" in data["vitals"]
        smoking = data["vitals"]["smoking"]
        assert smoking["label"] == "Do you smoke tobacco?"
        assert "Yes" in smoking["options"]
        assert "Sometimes" in smoking["options"]
        assert "No" in smoking["options"]
        print("✅ Smoking vital has correct structure")
    
    def test_vitals_options_contains_marijuana(self):
        """Test vitals options contains marijuana"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        data = response.json()
        
        assert "marijuana" in data["vitals"]
        marijuana = data["vitals"]["marijuana"]
        assert marijuana["label"] == "Do you smoke weed?"
        assert "Yes" in marijuana["options"]
        assert "Sometimes" in marijuana["options"]
        assert "No" in marijuana["options"]
        print("✅ Marijuana vital has correct structure")
    
    def test_vitals_options_contains_all_required_vitals(self):
        """Test all required vitals are present"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        data = response.json()
        
        required_vitals = [
            "ethnicity", "has_children", "family_plans", "height",
            "hometown", "work", "education", "drinking", "smoking",
            "marijuana", "relationship_type", "dating_intention",
            "religion", "politics", "zodiac", "pets", "exercise"
        ]
        
        for vital in required_vitals:
            assert vital in data["vitals"], f"Missing vital: {vital}"
        
        print(f"✅ All {len(required_vitals)} required vitals are present")


class TestProfilePromptsEndpoint:
    """Tests for GET /api/profile/prompts endpoint"""
    
    def test_prompts_returns_200(self):
        """Test prompts endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/profile/prompts")
        assert response.status_code == 200
        print("✅ Profile prompts endpoint returns 200")
    
    def test_prompts_contains_required_prompts(self):
        """Test prompts contains all required prompts"""
        response = requests.get(f"{BASE_URL}/api/profile/prompts")
        data = response.json()
        
        assert "prompts" in data
        prompts = data["prompts"]
        assert len(prompts) >= 15, f"Expected at least 15 prompts, got {len(prompts)}"
        
        # Check prompt structure
        for prompt in prompts:
            assert "id" in prompt
            assert "text" in prompt
            assert "category" in prompt
        
        # Check specific prompts exist
        prompt_ids = [p["id"] for p in prompts]
        assert "life_goal" in prompt_ids
        assert "love_language" in prompt_ids
        assert "dating_me" in prompt_ids
        
        print(f"✅ Profile prompts contains {len(prompts)} prompts with correct structure")


class TestProfileEndpointAuth:
    """Tests for profile endpoints that require authentication"""
    
    def test_profile_get_requires_auth(self):
        """Test GET /api/profile requires authentication"""
        response = requests.get(f"{BASE_URL}/api/profile")
        assert response.status_code == 401
        print("✅ GET /api/profile correctly requires authentication")
    
    def test_profile_put_requires_auth(self):
        """Test PUT /api/profile requires authentication"""
        response = requests.put(
            f"{BASE_URL}/api/profile",
            json={"ethnicity": ["East Asian"]}
        )
        assert response.status_code == 401
        print("✅ PUT /api/profile correctly requires authentication")


class TestSubscriptionPackages:
    """Tests for subscription packages endpoint"""
    
    def test_packages_returns_200(self):
        """Test subscription packages endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/subscription/packages")
        assert response.status_code == 200
        print("✅ Subscription packages endpoint returns 200")
    
    def test_packages_contains_required_packages(self):
        """Test packages contains all required subscription options"""
        response = requests.get(f"{BASE_URL}/api/subscription/packages")
        data = response.json()
        
        assert "packages" in data
        packages = data["packages"]
        
        # Check required packages exist
        assert "basic_daily" in packages
        assert "basic_weekly" in packages
        assert "basic_monthly" in packages
        assert "premium_daily" in packages
        assert "premium_weekly" in packages
        assert "premium_monthly" in packages
        assert "unlock_matches" in packages
        assert "rose" in packages
        assert "super_like" in packages
        
        # Check rose and super_like prices
        assert packages["rose"]["price"] == 2.99
        assert packages["super_like"]["price"] == 0.99
        
        print("✅ All subscription packages present with correct prices")


class TestAICompanionOptions:
    """Tests for AI companion customization options"""
    
    def test_ai_companion_options_returns_200(self):
        """Test AI companion options endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        print("✅ AI companion options endpoint returns 200")
    
    def test_ai_companion_options_contains_all_options(self):
        """Test AI companion options contains all customization options"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        data = response.json()
        
        assert "options" in data
        options = data["options"]
        
        required_options = [
            "skin_tone", "hair_color", "hair_style", "body_type",
            "facial_hair", "eye_color", "gender_presentation"
        ]
        
        for option in required_options:
            assert option in options, f"Missing option: {option}"
            assert len(options[option]) > 0, f"Empty options for: {option}"
        
        print(f"✅ All {len(required_options)} AI companion customization options present")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
