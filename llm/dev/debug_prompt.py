import sys
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Mock BaseClient and LLMResponse before importing evaluator if necessary, 
# but evaluator imports them from api_clients.
# Since we added project_root to sys.path, we can just let it import the real ones if they are simple.
# However, to be safe and avoid dependencies, validation errors or side effects, 
# I will try to rely on the existing code.

try:
    from llm.detection_evaluator import BugDetectionEvaluator
except ImportError as e:
    print(f"ImportError: {e}")
    # Try to import assuming we are in llm dir
    sys.path.append(str(current_dir))
    try:
        from detection_evaluator import BugDetectionEvaluator
    except ImportError as e2:
        print(f"ImportError 2: {e2}")
        sys.exit(1)

class MockClient:
    def __init__(self):
        self.model_name = "mock-model"
    
    def complete(self, prompt, system_prompt=None):
        return None 
    
    def __str__(self):
        return "MockClient"

def main():
    try:
        # Instantiate evaluator with the fixed detection prompt.
        client = MockClient()
        evaluator = BugDetectionEvaluator(client=client)
        
        # Create a dummy sample
        sample = {
            "code": "def quantum_func(q):\n    q.h(0)\n    return q.measure()",
            "label": 1,
            "sample_id": "test_sample_1",
            "is_quantum_related": True
        }
        
        system_prompt, user_prompt = evaluator._build_prompt(sample)
        
        print("\n" + "=" * 20 + " SYSTEM PROMPT " + "=" * 20)
        print(system_prompt)
        print("=" * 20 + " USER PROMPT " + "=" * 20)
        print(user_prompt)
        print("=" * 53)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
