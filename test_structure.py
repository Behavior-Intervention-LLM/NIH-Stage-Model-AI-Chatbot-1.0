#!/usr/bin/env python3
"""

"""
import sys
import os

# 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """"""
    try:
        print(" core.types...")
        from app.core import types
        print("✓ core.types ")
        
        print(" core.state_store...")
        from app.core import state_store
        print("✓ core.state_store ")
        
        print(" core.memory...")
        from app.core import memory
        print("✓ core.memory ")
        
        print(" core.router...")
        from app.core import router
        print("✓ core.router ")
        
        print(" core.orchestrator...")
        from app.core import orchestrator
        print("✓ core.orchestrator ")
        
        print(" agents...")
        from app.agents import intent_agent, stage_agent, planner_agent, responder_agent
        print("✓ agents ")
        
        print(" tools...")
        from app.tools import db_tool, vector_tool
        print("✓ tools ")
        
        print("\n！✓")
        return True
        
    except ImportError as e:
        print(f"\n: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
