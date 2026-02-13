#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime
import uuid

class SoulSyncAPITester:
    def __init__(self, base_url="https://aidate-soulmate.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.session_token = None
        self.user_id = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details
        })

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        test_headers = {'Content-Type': 'application/json'}
        
        if headers:
            test_headers.update(headers)
            
        if self.session_token:
            test_headers['Authorization'] = f'Bearer {self.session_token}'

        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        if self.session_token:
            print(f"   Auth: Bearer {self.session_token[:20]}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=test_headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=test_headers, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=test_headers, timeout=30)

            print(f"   Response: {response.status_code}")
            success = response.status_code == expected_status
            
            if success:
                self.log_test(name, True)
                try:
                    return True, response.json()
                except:
                    return True, response.text
            else:
                error_msg = f"Expected {expected_status}, got {response.status_code}"
                try:
                    error_details = response.json()
                    error_msg += f" - {error_details}"
                except:
                    error_msg += f" - {response.text[:200]}"
                self.log_test(name, False, error_msg)
                return False, {}

        except Exception as e:
            self.log_test(name, False, f"Exception: {str(e)}")
            return False, {}

    def test_health_endpoint(self):
        """Test health check endpoint"""
        return self.run_test("Health Check", "GET", "health", 200)

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root API", "GET", "", 200)

    def create_test_session(self):
        """Use the pre-created test user session"""
        print("\n🔧 Using test session...")
        
        # Use the test user we created in MongoDB
        self.session_token = "test_token_12345"
        self.user_id = "test_user_12345"
        
        print(f"   Session Token: {self.session_token}")
        print(f"   User ID: {self.user_id}")
        return True

    def test_auth_me(self):
        """Test getting current user"""
        if not self.session_token:
            self.log_test("Auth Me", False, "No session token")
            return False
        
        return self.run_test("Get Current User", "GET", "auth/me", 200)

    def test_profile_operations(self):
        """Test profile CRUD operations"""
        if not self.session_token:
            self.log_test("Profile Operations", False, "No session token")
            return False

        # Get profile
        success, profile = self.run_test("Get Profile", "GET", "profile", 200)
        
        if not success:
            return False

        # Update profile
        update_data = {
            "bio": "Test bio for SoulSync testing",
            "age": 28,
            "gender": "male",
            "location": "Test City, TC",
            "occupation": "Software Tester",
            "interests": ["Testing", "AI", "Technology"],
            "looking_for": "women",
            "relationship_goals": "long-term",
            "deal_breakers": ["Dishonesty"]
        }
        
        return self.run_test("Update Profile", "PUT", "profile", 200, data=update_data)

    def test_chat_message(self):
        """Test sending a chat message"""
        if not self.session_token:
            self.log_test("Chat Message", False, "No session token")
            return False

        message_data = {
            "message": "Hello Soul! This is a test message from the automated testing system."
        }
        
        return self.run_test("Send Chat Message", "POST", "chat/message", 200, data=message_data)

    def test_chat_history(self):
        """Test getting chat history"""
        if not self.session_token:
            self.log_test("Chat History", False, "No session token")
            return False

        return self.run_test("Get Chat History", "GET", "chat/history", 200)

    def test_insights(self):
        """Test getting personality insights"""
        if not self.session_token:
            self.log_test("Get Insights", False, "No session token")
            return False

        return self.run_test("Get Personality Insights", "GET", "insights", 200)

    def test_insights_analyze(self):
        """Test personality analysis"""
        if not self.session_token:
            self.log_test("Analyze Personality", False, "No session token")
            return False

        return self.run_test("Analyze Personality", "POST", "insights/analyze", 200)

    def test_subscription_packages(self):
        """Test getting subscription packages"""
        success, response = self.run_test("Get Subscription Packages", "GET", "subscription/packages", 200)
        
        if success and isinstance(response, dict):
            packages = response.get('packages', {})
            # Check for roses and super_like packages
            if 'rose' in packages and 'super_like' in packages:
                rose_pkg = packages['rose']
                super_like_pkg = packages['super_like']
                
                if rose_pkg.get('price') == 2.99 and super_like_pkg.get('price') == 0.99:
                    self.log_test("Rose & Super Like Packages", True)
                else:
                    self.log_test("Rose & Super Like Packages", False, f"Rose: ${rose_pkg.get('price')}, Super Like: ${super_like_pkg.get('price')}")
            else:
                self.log_test("Rose & Super Like Packages", False, "Packages not found")
        
        return success

    def test_subscription_status(self):
        """Test getting subscription status"""
        if not self.session_token:
            self.log_test("Subscription Status", False, "No session token")
            return False

        return self.run_test("Get Subscription Status", "GET", "subscription/status", 200)

    def test_phone_verification_send_otp(self):
        """Test sending OTP (will fail without valid phone, but should return proper error)"""
        if not self.session_token:
            self.log_test("Send OTP", False, "No session token")
            return False

        # Test with invalid phone number to check endpoint exists
        phone_data = {"phone_number": "+1234567890"}
        success, response = self.run_test("Send OTP (Invalid Phone)", "POST", "auth/send-otp", 400, data=phone_data)
        
        # We expect 400 for invalid phone, so this is actually success
        if not success:
            # Try to see if endpoint exists by checking the error
            try:
                # If we get a 400, the endpoint exists and is working
                return True
            except:
                return False
        return True

    def test_stripe_checkout(self):
        """Test Stripe checkout session creation"""
        if not self.session_token:
            self.log_test("Stripe Checkout", False, "No session token")
            return False

        checkout_data = {
            "package_id": "basic_monthly",
            "origin_url": "https://aidate-soulmate.preview.emergentagent.com"
        }
        
        return self.run_test("Create Stripe Checkout", "POST", "payments/checkout", 200, data=checkout_data)

    def test_profile_with_dob(self):
        """Test profile update with date of birth and age calculation"""
        if not self.session_token:
            self.log_test("Profile DOB Update", False, "No session token")
            return False

        # Update profile with DOB
        update_data = {
            "date_of_birth": "01/15/1995",  # MM/DD/YYYY format
            "bio": "Test bio with DOB",
            "gender": "male",
            "location": "Test City, TC",
            "occupation": "Software Tester"
        }
        
        success, response = self.run_test("Update Profile with DOB", "PUT", "profile", 200, data=update_data)
        
        if success and isinstance(response, dict):
            # Check if age was calculated correctly
            expected_age = 2026 - 1995  # Current year - birth year
            actual_age = response.get('age')
            if actual_age and abs(actual_age - expected_age) <= 1:  # Allow 1 year difference for birthday
                self.log_test("Age Calculation", True)
            else:
                self.log_test("Age Calculation", False, f"Expected ~{expected_age}, got {actual_age}")
        
        return success

    def test_ai_companion_options(self):
        """Test AI companion customization options"""
        success, response = self.run_test("Get AI Companion Options", "GET", "ai-companion/options", 200)
        
        if success and isinstance(response, dict):
            options = response.get('options', {})
            required_options = ['skin_tone', 'hair_color', 'hair_style', 'body_type', 'facial_hair', 'eye_color', 'gender_presentation']
            
            missing_options = [opt for opt in required_options if opt not in options]
            if not missing_options:
                self.log_test("AI Companion Options Complete", True)
            else:
                self.log_test("AI Companion Options Complete", False, f"Missing: {missing_options}")
        
        return success

    def test_profile_prompts(self):
        """Test profile prompts endpoint"""
        success, response = self.run_test("Get Profile Prompts", "GET", "profile/prompts", 200)
        
        if success and isinstance(response, dict):
            prompts = response.get('prompts', [])
            if len(prompts) >= 10:  # Should have multiple prompts
                self.log_test("Profile Prompts Available", True, f"Found {len(prompts)} prompts")
            else:
                self.log_test("Profile Prompts Available", False, f"Only {len(prompts)} prompts found")
        
        return success

    def test_roses_endpoint(self):
        """Test roses interaction endpoint"""
        if not self.session_token:
            self.log_test("Roses Endpoint", False, "No session token")
            return False

        # Test with invalid target user (should return proper error)
        rose_data = {
            "target_user_id": "invalid_user_123",
            "message": "Test rose message"
        }
        
        # We expect this to fail with 402 (no roses) or 404 (user not found)
        success, response = self.run_test("Send Rose (Invalid User)", "POST", "interactions/rose", 402, data=rose_data)
        
        # If we get 402, it means the endpoint exists and is checking roses balance
        return success

    def test_super_likes_endpoint(self):
        """Test super likes interaction endpoint"""
        if not self.session_token:
            self.log_test("Super Likes Endpoint", False, "No session token")
            return False

        # Test with invalid target user (should return proper error)
        super_like_data = {
            "target_user_id": "invalid_user_123",
            "prompt_id": "life_goal"
        }
        
        # We expect this to fail with 402 (no super likes) or 404 (user not found)
        success, response = self.run_test("Send Super Like (Invalid User)", "POST", "interactions/super-like", 402, data=super_like_data)
        
        # If we get 402, it means the endpoint exists and is checking super likes balance
        return success

    def test_logout(self):
        """Test logout"""
        if not self.session_token:
            self.log_test("Logout", False, "No session token")
            return False

        return self.run_test("Logout", "POST", "auth/logout", 200)

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting SoulSyncAI API Tests")
        print(f"   Base URL: {self.base_url}")
        print(f"   API URL: {self.api_url}")
        print("=" * 60)

        # Basic endpoint tests
        self.test_health_endpoint()
        self.test_root_endpoint()

        # Test subscription packages (public endpoint)
        self.test_subscription_packages()
        
        # Test AI companion options (public endpoint)
        self.test_ai_companion_options()
        
        # Test profile prompts (public endpoint)
        self.test_profile_prompts()

        # Authentication flow
        if self.create_test_session():
            self.test_auth_me()
            
            # Profile operations
            self.test_profile_operations()
            
            # Test DOB and age calculation
            self.test_profile_with_dob()
            
            # Subscription status (authenticated)
            self.test_subscription_status()
            
            # Phone verification (will test endpoint existence)
            self.test_phone_verification_send_otp()
            
            # Stripe checkout
            self.test_stripe_checkout()
            
            # Test roses and super likes endpoints
            self.test_roses_endpoint()
            self.test_super_likes_endpoint()
            
            # Chat functionality
            self.test_chat_message()
            self.test_chat_history()
            
            # Insights
            self.test_insights()
            # Note: Skipping analyze as it requires more conversations
            
            # Logout
            self.test_logout()
        else:
            print("❌ Failed to create test session, skipping authenticated tests")

        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed. Check the details above.")
            return 1

def main():
    tester = SoulSyncAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())