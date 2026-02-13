"""
Test AI Companion Options API - Enhanced Customization Features
Tests for appearance options (age_appearance, style_vibe) and personality options
(communication_style, energy_level, humor_style, conversation_topics)
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestAICompanionOptions:
    """Test AI Companion customization options endpoint"""
    
    def test_health_check(self):
        """Verify API is healthy"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health check passed")
    
    def test_ai_companion_options_returns_appearance_and_personality(self):
        """GET /api/ai-companion/options should return both appearance AND personality options"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        data = response.json()
        
        # Check both main keys exist
        assert "appearance" in data, "Missing 'appearance' key in response"
        assert "personality" in data, "Missing 'personality' key in response"
        print("✅ Response contains both 'appearance' and 'personality' keys")
    
    def test_age_appearance_options_exist(self):
        """Verify age_appearance options exist with correct values"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        data = response.json()
        
        appearance = data.get("appearance", {})
        assert "age_appearance" in appearance, "Missing 'age_appearance' in appearance options"
        
        age_options = appearance["age_appearance"]
        expected_options = ["Young Adult (20s)", "Adult (30s)", "Mature (40s)", "Distinguished (50s+)"]
        
        for expected in expected_options:
            assert expected in age_options, f"Missing age_appearance option: {expected}"
        
        assert len(age_options) == 4, f"Expected 4 age_appearance options, got {len(age_options)}"
        print(f"✅ age_appearance has all 4 options: {age_options}")
    
    def test_style_vibe_options_exist(self):
        """Verify style_vibe options exist with 6 options including 'Warm & Nurturing'"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        data = response.json()
        
        appearance = data.get("appearance", {})
        assert "style_vibe" in appearance, "Missing 'style_vibe' in appearance options"
        
        style_options = appearance["style_vibe"]
        expected_options = [
            "Casual & Friendly",
            "Professional & Polished",
            "Artistic & Creative",
            "Sporty & Active",
            "Elegant & Sophisticated",
            "Warm & Nurturing"
        ]
        
        for expected in expected_options:
            assert expected in style_options, f"Missing style_vibe option: {expected}"
        
        assert len(style_options) == 6, f"Expected 6 style_vibe options, got {len(style_options)}"
        print(f"✅ style_vibe has all 6 options including 'Warm & Nurturing'")
    
    def test_communication_style_has_5_options_with_structure(self):
        """Verify personality.communication_style has 5 options with id, name, description"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        data = response.json()
        
        personality = data.get("personality", {})
        assert "communication_style" in personality, "Missing 'communication_style' in personality options"
        
        comm_styles = personality["communication_style"]
        assert len(comm_styles) == 5, f"Expected 5 communication_style options, got {len(comm_styles)}"
        
        expected_ids = ["supportive", "playful", "thoughtful", "direct", "gentle"]
        
        for style in comm_styles:
            assert "id" in style, f"Missing 'id' in communication_style: {style}"
            assert "name" in style, f"Missing 'name' in communication_style: {style}"
            assert "description" in style, f"Missing 'description' in communication_style: {style}"
            assert isinstance(style["id"], str), f"'id' should be string: {style}"
            assert isinstance(style["name"], str), f"'name' should be string: {style}"
            assert isinstance(style["description"], str), f"'description' should be string: {style}"
        
        actual_ids = [s["id"] for s in comm_styles]
        for expected_id in expected_ids:
            assert expected_id in actual_ids, f"Missing communication_style id: {expected_id}"
        
        print(f"✅ communication_style has 5 options with id, name, description: {actual_ids}")
    
    def test_energy_level_has_3_options(self):
        """Verify personality.energy_level has 3 options"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        data = response.json()
        
        personality = data.get("personality", {})
        assert "energy_level" in personality, "Missing 'energy_level' in personality options"
        
        energy_levels = personality["energy_level"]
        expected_options = ["Calm & Relaxed", "Balanced", "Enthusiastic & Energetic"]
        
        assert len(energy_levels) == 3, f"Expected 3 energy_level options, got {len(energy_levels)}"
        
        for expected in expected_options:
            assert expected in energy_levels, f"Missing energy_level option: {expected}"
        
        print(f"✅ energy_level has 3 options: {energy_levels}")
    
    def test_humor_style_has_4_options(self):
        """Verify personality.humor_style has 4 options"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        data = response.json()
        
        personality = data.get("personality", {})
        assert "humor_style" in personality, "Missing 'humor_style' in personality options"
        
        humor_styles = personality["humor_style"]
        expected_options = ["Dry & Subtle", "Warm & Playful", "Bold & Silly", "Minimal"]
        
        assert len(humor_styles) == 4, f"Expected 4 humor_style options, got {len(humor_styles)}"
        
        for expected in expected_options:
            assert expected in humor_styles, f"Missing humor_style option: {expected}"
        
        print(f"✅ humor_style has 4 options: {humor_styles}")
    
    def test_conversation_topics_has_6_topics(self):
        """Verify personality.conversation_topics has 6 topics"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        data = response.json()
        
        personality = data.get("personality", {})
        assert "conversation_topics" in personality, "Missing 'conversation_topics' in personality options"
        
        topics = personality["conversation_topics"]
        assert len(topics) == 6, f"Expected 6 conversation_topics, got {len(topics)}"
        
        expected_ids = ["relationships", "life_goals", "daily_life", "emotions", "hobbies", "growth"]
        
        for topic in topics:
            assert "id" in topic, f"Missing 'id' in conversation_topic: {topic}"
            assert "name" in topic, f"Missing 'name' in conversation_topic: {topic}"
        
        actual_ids = [t["id"] for t in topics]
        for expected_id in expected_ids:
            assert expected_id in actual_ids, f"Missing conversation_topic id: {expected_id}"
        
        print(f"✅ conversation_topics has 6 topics: {actual_ids}")
    
    def test_backward_compatibility_options_key(self):
        """Verify backward compatibility - 'options' key still exists"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        data = response.json()
        
        assert "options" in data, "Missing backward compatibility 'options' key"
        
        # Verify options has same structure as appearance
        options = data["options"]
        assert "age_appearance" in options, "Missing 'age_appearance' in options (backward compat)"
        assert "style_vibe" in options, "Missing 'style_vibe' in options (backward compat)"
        
        print("✅ Backward compatibility 'options' key exists with age_appearance and style_vibe")
    
    def test_all_appearance_options_present(self):
        """Verify all appearance options are present"""
        response = requests.get(f"{BASE_URL}/api/ai-companion/options")
        assert response.status_code == 200
        data = response.json()
        
        appearance = data.get("appearance", {})
        required_keys = [
            "skin_tone", "hair_color", "hair_style", "body_type",
            "facial_hair", "eye_color", "gender_presentation",
            "age_appearance", "style_vibe"
        ]
        
        for key in required_keys:
            assert key in appearance, f"Missing appearance option: {key}"
            assert isinstance(appearance[key], list), f"'{key}' should be a list"
            assert len(appearance[key]) > 0, f"'{key}' should not be empty"
        
        print(f"✅ All {len(required_keys)} appearance options present: {required_keys}")


class TestAICompanionCustomizeEndpoint:
    """Test AI Companion customize endpoint (requires auth)"""
    
    def test_customize_requires_authentication(self):
        """POST /api/ai-companion/customize should require authentication"""
        response = requests.post(
            f"{BASE_URL}/api/ai-companion/customize",
            json={
                "name": "TestSoul",
                "skin_tone": "Medium",
                "hair_color": "Brown",
                "hair_style": "Medium",
                "body_type": "Athletic",
                "facial_hair": "None",
                "eye_color": "Brown",
                "gender_presentation": "Androgynous"
            }
        )
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("✅ POST /api/ai-companion/customize requires authentication (401)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
