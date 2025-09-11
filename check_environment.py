import sys

def check_groq_installation():
    """
    Checks if the groq library is installed and prints environment details.
    """
    print("-" * 50)
    # Print the exact path of the Python executable being used.
    print(f"🐍 Python Executable Path:\n   {sys.executable}")
    print("-" * 50)
    
    try:
        # Try to import groq and get its version.
        import groq
        print(f"✅ SUCCESS: The 'groq' library is installed.")
        print(f"   - Version: {groq.__version__}")
        print("\n💡 Now, run your Streamlit app from this SAME terminal window.")
        
    except ImportError:
        # If the import fails, it means the library is not in this environment.
        print(f"❌ ERROR: The 'groq' library is NOT installed in this Python environment.")
        print(f"\n👉 To fix this, run the following command in THIS terminal:")
        print(f"   {sys.executable} -m pip install groq")
        
    finally:
        print("-" * 50)

if __name__ == "__main__":
    check_groq_installation()
