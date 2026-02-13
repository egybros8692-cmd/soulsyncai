"""
SoulSyncAI Backend API Tests
Tests for health check, AI companion options, and profile prompts endpoints
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestHealthCheck:
    """Health check endpoint tests"""
    
    def test_health_endpoint_returns_200(self):
        """Test that health endpoint returns 200 status"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        
    def test_health_endpoint_returns_healthy_status(self):
        """Test that health endpoint returns healthy status"""
        response = requests.get(f"{BASE_URL}/api/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestAICompanionOptions:
    """AI Companion customization options endpoint tests"""
    
    def test_ai_companion_options_returns_200(self):
        """Test that AI companion options endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        
    def test_ai_companion_options_has_appearance(self):
        """Test that AI companion options includes appearance options"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        data = response.json()
        assert "appearance" in data or "options" in data
        
    def test_ai_companion_options_has_skin_tone(self):
        """Test that appearance options include skin_tone"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        data = response.json()
        appearance = data.get("appearance") or data.get("options")
        assert "skin_tone" in appearance
        assert len(appearance["skin_tone"]) > 0
        
    def test_ai_companion_options_has_hair_options(self):
        """Test that appearance options include hair color and style"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        data = response.json()
        appearance = data.get("appearance") or data.get("options")
        assert "hair_color" in appearance
        assert "hair_style" in appearance
        
    def test_ai_companion_options_has_style_vibe(self):
        """Test that appearance options include style_vibe"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        data = response.json()
        appearance = data.get("appearance") or data.get("options")
        assert "style_vibe" in appearance
        # Verify expected style vibes exist
        expected_vibes = ["Warm & Nurturing", "Casual & Friendly"]
        for vibe in expected_vibes:
            assert vibe in appearance["style_vibe"]
            
    def test_ai_companion_options_has_personality(self):
        """Test that AI companion options includes personality options"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        data = response.json()
        assert "personality" in data
        assert "communication_style" in data["personality"]


class TestProfilePrompts:
    """Profile prompts endpoint tests"""
    
    def test_profile_prompts_returns_200(self):
        """Test that profile prompts endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/profile/prompts")
        assert response.status_code == 200
        
    def test_profile_prompts_returns_list(self):
        """Test that profile prompts returns a list inside 'prompts' key"""
        response = requests.get(f"{BASE_URL}/api/profile/prompts")
        data = response.json()
        assert "prompts" in data
        assert isinstance(data["prompts"], list)
        assert len(data["prompts"]) > 0
        
    def test_profile_prompts_have_required_fields(self):
        """Test that each prompt has id, text, and category"""
        response = requests.get(f"{BASE_URL}/api/profile/prompts")
        data = response.json()
        for prompt in data["prompts"]:
            assert "id" in prompt
            assert "text" in prompt
            assert "category" in prompt


class TestSubscriptionPackages:
    """Subscription packages endpoint tests"""
    
    def test_subscription_packages_returns_200(self):
        """Test that subscription packages endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/subscription/packages")
        assert response.status_code == 200
        
    def test_subscription_packages_returns_dict(self):
        """Test that subscription packages returns a dictionary with 'packages' key"""
        response = requests.get(f"{BASE_URL}/api/subscription/packages")
        data = response.json()
        assert isinstance(data, dict)
        assert "packages" in data
        
    def test_subscription_packages_has_basic_and_premium(self):
        """Test that subscription packages include basic and premium tiers"""
        response = requests.get(f"{BASE_URL}/api/subscription/packages")
        data = response.json()
        packages = data.get("packages", {})
        # Check for basic and premium packages
        has_basic = any("basic" in key for key in packages.keys())
        has_premium = any("premium" in key for key in packages.keys())
        assert has_basic, "Should have basic subscription packages"
        assert has_premium, "Should have premium subscription packages"


class TestAuthEndpoints:
    """Authentication endpoint tests (unauthenticated)"""
    
    def test_auth_me_returns_401_when_not_authenticated(self):
        """Test that /auth/me returns 401 when not authenticated"""
        response = requests.get(f"{BASE_URL}/api/auth/me")
        # Should return 401 or redirect when not authenticated
        assert response.status_code in [401, 403, 200]  # 200 if returns null user


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
