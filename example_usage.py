#!/usr/bin/env python3
"""
： API
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_chat():
    """"""
    url = f"{BASE_URL}/chat"
    
    # 
    test_messages = [
        " NIH Stage Model",
        "",
        "？"
    ]
    
    session_id = "test_session_001"
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'='*60}")
        print(f" {i}: {message}")
        print('='*60)
        
        payload = {
            "session_id": session_id,
            "message": message
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            print(f"\n: {data['reply']}")
            
            if data.get('debug'):
                print(f"\n:")
                print(json.dumps(data['debug'], indent=2, ensure_ascii=False))
        
        except requests.exceptions.ConnectionError:
            print(": 。（ python -m app.main）")
            break
        except Exception as e:
            print(f": {e}")

def test_session_info():
    """"""
    session_id = "test_session_001"
    url = f"{BASE_URL}/sessions/{session_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"\n:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f": {e}")

if __name__ == "__main__":
    print("NIH Stage Model AI Chatbot - ")
    print("="*60)
    
    # 
    test_chat()
    
    # 
    print(f"\n{'='*60}")
    print("")
    print('='*60)
    test_session_info()
