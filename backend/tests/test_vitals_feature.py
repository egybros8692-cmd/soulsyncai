"""
Test suite for SoulSyncAI Vitals Feature
Tests the new vitals fields: school, education, religion, politics
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestVitalsAPI:
    """Test /api/profile/vitals-options endpoint"""
    
    def test_health_check(self):
        """Verify API is healthy"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health check passed")
    
    def test_vitals_options_returns_18_vitals(self):
        """Verify API returns 18 total vitals"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        assert response.status_code == 200
        data = response.json()
        vitals = data.get("vitals", {})
        assert len(vitals) == 18, f"Expected 18 vitals, got {len(vitals)}"
        print(f"✅ API returns 18 vitals: {list(vitals.keys())}")
    
    def test_school_field_is_text_type(self):
        """Verify 'school' field exists with type: text"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        assert response.status_code == 200
        data = response.json()
        vitals = data.get("vitals", {})
        
        assert "school" in vitals, "school field missing from vitals"
        school = vitals["school"]
        assert school.get("type") == "text", f"school type should be 'text', got {school.get('type')}"
        assert school.get("label") == "Where did you go to school?"
        assert school.get("placeholder") == "School name"
        assert school.get("icon") == "graduation-cap"
        print("✅ School field is text type with correct label and placeholder")
    
    def test_education_options(self):
        """Verify 'education' options are High School, Undergrad, Postgrad, Prefer not to say"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        assert response.status_code == 200
        data = response.json()
        vitals = data.get("vitals", {})
        
        assert "education" in vitals, "education field missing from vitals"
        education = vitals["education"]
        expected_options = ["High School", "Undergrad", "Postgrad", "Prefer not to say"]
        actual_options = education.get("options", [])
        
        assert actual_options == expected_options, f"Expected {expected_options}, got {actual_options}"
        assert education.get("label") == "What's the highest level you attained?"
        print(f"✅ Education options correct: {actual_options}")
    
    def test_religion_options(self):
        """Verify 'religion' options include required values"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        assert response.status_code == 200
        data = response.json()
        vitals = data.get("vitals", {})
        
        assert "religion" in vitals, "religion field missing from vitals"
        religion = vitals["religion"]
        actual_options = religion.get("options", [])
        
        # Required options per test request
        required_options = ["Agnostic", "Atheist", "Buddhist", "Catholic", "Christian", "Hindu", "Jewish", "Muslim", "Sikh"]
        for opt in required_options:
            assert opt in actual_options, f"Missing religion option: {opt}"
        
        assert religion.get("label") == "What are your religious beliefs?"
        assert religion.get("icon") == "book"
        print(f"✅ Religion options include all required values: {actual_options}")
    
    def test_politics_options(self):
        """Verify 'politics' options include required values"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        assert response.status_code == 200
        data = response.json()
        vitals = data.get("vitals", {})
        
        assert "politics" in vitals, "politics field missing from vitals"
        politics = vitals["politics"]
        actual_options = politics.get("options", [])
        
        # Required options per test request
        required_options = ["Liberal", "Moderate", "Conservative", "Not Political", "Other"]
        for opt in required_options:
            assert opt in actual_options, f"Missing politics option: {opt}"
        
        assert politics.get("label") == "What are your political beliefs?"
        assert politics.get("icon") == "landmark"
        print(f"✅ Politics options include all required values: {actual_options}")
    
    def test_all_12_vitals_in_flow_exist(self):
        """Verify all 12 vitals in the flow exist in API response"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        assert response.status_code == 200
        data = response.json()
        vitals = data.get("vitals", {})
        
        # These are the 12 vitals shown in VitalsSetup flow
        vitals_in_flow = [
            "ethnicity", "has_children", "family_plans", "work", "school",
            "education", "hometown", "religion", "politics", "drinking",
            "smoking", "marijuana"
        ]
        
        for vital in vitals_in_flow:
            assert vital in vitals, f"Missing vital in API: {vital}"
        
        print(f"✅ All 12 vitals in flow exist in API response")
    
    def test_ethnicity_is_multiselect(self):
        """Verify ethnicity supports multiple selection"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        assert response.status_code == 200
        data = response.json()
        vitals = data.get("vitals", {})
        
        ethnicity = vitals.get("ethnicity", {})
        assert ethnicity.get("multiple") == True, "ethnicity should support multiple selection"
        assert ethnicity.get("allow_custom") == True, "ethnicity should allow custom input"
        print("✅ Ethnicity supports multi-select and custom input")
    
    def test_text_type_vitals(self):
        """Verify hometown, work, school are text type"""
        response = requests.get(f"{BASE_URL}/api/profile/vitals-options")
        assert response.status_code == 200
        data = response.json()
        vitals = data.get("vitals", {})
        
        text_vitals = ["hometown", "work", "school"]
        for vital_key in text_vitals:
            vital = vitals.get(vital_key, {})
            assert vital.get("type") == "text", f"{vital_key} should be text type"
            assert "placeholder" in vital, f"{vital_key} should have placeholder"
        
        print("✅ hometown, work, school are all text type with placeholders")


class TestProfileEndpoints:
    """Test profile-related endpoints"""
    
    def test_profile_requires_auth(self):
        """Verify GET /api/profile requires authentication"""
        response = requests.get(f"{BASE_URL}/api/profile")
        assert response.status_code == 401
        print("✅ GET /api/profile requires authentication (401)")
    
    def test_profile_update_requires_auth(self):
        """Verify PUT /api/profile requires authentication"""
        response = requests.put(f"{BASE_URL}/api/profile", json={"bio": "test"})
        assert response.status_code == 401
        print("✅ PUT /api/profile requires authentication (401)")
    
    def test_prompts_endpoint(self):
        """Verify GET /api/profile/prompts returns prompts"""
        response = requests.get(f"{BASE_URL}/api/profile/prompts")
        assert response.status_code == 200
        data = response.json()
        assert "prompts" in data
        assert len(data["prompts"]) > 0
        print(f"✅ GET /api/profile/prompts returns {len(data['prompts'])} prompts")


class TestOtherEndpoints:
    """Test other public endpoints"""
    
    def test_subscription_packages(self):
        """Verify subscription packages endpoint"""
        response = requests.get(f"{BASE_URL}/api/subscription/packages")
        assert response.status_code == 200
        data = response.json()
        assert "packages" in data
        packages = data["packages"]
        assert "rose" in packages
        assert "super_like" in packages
        print(f"✅ Subscription packages endpoint returns {len(packages)} packages")
    
    def test_ai_companion_options(self):
        """Verify AI companion customization options"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        data = response.json()
        assert "options" in data
        options = data["options"]
        expected_keys = ["skin_tone", "hair_color", "hair_style", "body_type", "facial_hair", "eye_color", "gender_presentation"]
        for key in expected_keys:
            assert key in options, f"Missing AI companion option: {key}"
        print(f"✅ AI companion options endpoint returns all customization options")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
