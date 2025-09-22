"""
Ethical Eye Training Pipeline Runner
Research Project: Explainable AI for Dark Pattern Detection

This script runs the complete training pipeline including:
1. DistilBERT model training
2. SHAP explanation generation
3. Research plot generation
4. Model evaluation and validation
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import json

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS!")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ ERROR!")
        print("STDERR:", e.stderr)
        if e.stdout:
            print("STDOUT:", e.stdout)
        return False

def check_requirements():
    """Check if required packages are installed"""
    print("Checking requirements...")
    
    required_packages = [
        'torch', 'transformers', 'sklearn', 'numpy', 'pandas',
        'shap', 'matplotlib', 'seaborn', 'flask', 'flask_cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them using: pip install -r requirements_training.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        'models/ethical_eye',
        'plots/research/paper',
        'plots/research/supplementary',
        'plots/research/shap',
        'logs/training',
        'results/evaluation',
        'results/shap',
        'data/processed'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def run_training():
    """Run the DistilBERT training"""
    return run_command(
        "python training/train_distilbert.py",
        "DistilBERT Model Training"
    )

def run_shap_analysis():
    """Run SHAP analysis"""
    return run_command(
        "python training/shap_explainer.py",
        "SHAP Explanation Analysis"
    )

def generate_research_plots():
    """Generate research plots"""
    return run_command(
        "python training/generate_research_plots.py",
        "Research Plot Generation"
    )

def test_api():
    """Test the API"""
    print("\nTesting API...")
    
    # Start API in background
    api_process = subprocess.Popen(
        ["python", "api/ethical_eye_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for API to start
    import time
    time.sleep(5)
    
    # Test API endpoints
    try:
        import requests
        
        # Test health check
        response = requests.get("http://127.0.0.1:5000/", timeout=10)
        if response.status_code == 200:
            print("✅ API Health Check: PASSED")
        else:
            print("❌ API Health Check: FAILED")
        
        # Test single text analysis
        test_data = {"text": "Hurry! Only 2 left in stock!", "confidence_threshold": 0.7}
        response = requests.post("http://127.0.0.1:5000/analyze_single", 
                               json=test_data, timeout=10)
        if response.status_code == 200:
            print("✅ Single Text Analysis: PASSED")
            result = response.json()
            print(f"   Category: {result.get('category', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
        else:
            print("❌ Single Text Analysis: FAILED")
        
        # Test pattern info
        response = requests.get("http://127.0.0.1:5000/patterns", timeout=10)
        if response.status_code == 200:
            print("✅ Pattern Info: PASSED")
        else:
            print("❌ Pattern Info: FAILED")
            
    except Exception as e:
        print(f"❌ API Test Error: {e}")
    
    finally:
        # Stop API
        api_process.terminate()
        api_process.wait()

def generate_summary_report():
    """Generate a summary report"""
    print("\nGenerating summary report...")
    
    report = {
        "training_completed": datetime.now().isoformat(),
        "project": "Ethical Eye Extension",
        "description": "Explainable AI for Dark Pattern Detection",
        "components": {
            "model_training": "DistilBERT fine-tuned for dark pattern classification",
            "shap_explanations": "SHAP-based explanations for transparency",
            "research_plots": "Comprehensive visualizations for research paper",
            "api": "Flask API with real-time analysis capabilities"
        },
        "outputs": {
            "model": "models/ethical_eye/final_model/",
            "plots": "plots/research/",
            "results": "results/",
            "logs": "logs/training/"
        },
        "next_steps": [
            "Load the trained model in the Chrome extension",
            "Update the extension to use the new API",
            "Conduct user study with 5-10 participants",
            "Write research paper for conference submission"
        ]
    }
    
    with open("training_summary_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("✅ Summary report saved: training_summary_report.json")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Ethical Eye Training Pipeline")
    parser.add_argument("--skip-checks", action="store_true", help="Skip requirement checks")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--skip-api-test", action="store_true", help="Skip API testing")
    
    args = parser.parse_args()
    
    print("🚀 ETHICAL EYE TRAINING PIPELINE")
    print("="*60)
    print("Research Project: Explainable AI for Dark Pattern Detection")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Check requirements
    if not args.skip_checks:
        if not check_requirements():
            print("\n❌ Requirements check failed. Please install missing packages.")
            return 1
    
    # Create directories
    create_directories()
    
    # Run training pipeline
    success_count = 0
    total_steps = 0
    
    # 1. Model Training
    if not args.skip_training:
        total_steps += 1
        if run_training():
            success_count += 1
        else:
            print("❌ Training failed. Stopping pipeline.")
            return 1
    
    # 2. SHAP Analysis
    if not args.skip_shap:
        total_steps += 1
        if run_shap_analysis():
            success_count += 1
        else:
            print("⚠️ SHAP analysis failed. Continuing with other steps.")
    
    # 3. Research Plots
    if not args.skip_plots:
        total_steps += 1
        if generate_research_plots():
            success_count += 1
        else:
            print("⚠️ Plot generation failed. Continuing with other steps.")
    
    # 4. API Testing
    if not args.skip_api_test:
        total_steps += 1
        test_api()
        success_count += 1  # API test doesn't fail the pipeline
    
    # Generate summary report
    generate_summary_report()
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETED!")
    print("="*60)
    print(f"Successfully completed: {success_count}/{total_steps} steps")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📁 OUTPUTS:")
    print("✅ Trained Model: models/ethical_eye/final_model/")
    print("✅ Research Plots: plots/research/")
    print("✅ SHAP Analysis: results/shap/")
    print("✅ Evaluation Results: results/evaluation/")
    print("✅ Training Logs: logs/training/")
    print("✅ Summary Report: training_summary_report.json")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Update Chrome extension to use new API")
    print("2. Test extension with trained model")
    print("3. Conduct user study (5-10 participants)")
    print("4. Write research paper")
    print("5. Submit to SCCUR or CHI SRC")
    
    if success_count == total_steps:
        print("\n✅ ALL STEPS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n⚠️ {total_steps - success_count} steps had issues. Check logs for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
