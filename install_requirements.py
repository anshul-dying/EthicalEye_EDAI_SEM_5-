"""
Install requirements for Ethical Eye Extension
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages"""
    print("Installing requirements for Ethical Eye Extension...")
    
    # Core packages
    packages = [
        "torch>=1.12.0",
        "transformers>=4.21.0", 
        "scikit-learn>=1.1.0",
        "numpy>=1.21.0",
        "pandas>=1.4.0",
        "shap>=0.41.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "flask>=2.2.0",
        "flask-cors>=3.0.10",
        "nltk>=3.7",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "requests>=2.28.0"
    ]
    
    success_count = 0
    total_packages = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstallation completed: {success_count}/{total_packages} packages installed successfully")
    
    if success_count == total_packages:
        print("🎉 All packages installed successfully!")
        print("You can now run: python run_training_pipeline.py")
    else:
        print("⚠️ Some packages failed to install. Please check the errors above.")

if __name__ == "__main__":
    main()
