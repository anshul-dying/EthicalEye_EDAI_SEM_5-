"""
Test script for data balancing
"""

import pandas as pd
import sys
import os

# Add training directory to path
sys.path.append('training')
from data_balancer import AdvancedDataBalancer

def test_balancing():
    """Test the data balancing functionality"""
    
    # Load the actual data
    print("Loading datasets...")
    
    try:
        dark_patterns_df = pd.read_csv('LLM-Model/dark_patterns.csv')
        normie_df = pd.read_csv('LLM-Model/normie.csv')
        dataset_df = pd.read_csv('LLM-Model/dataset.csv')
        
        print("✅ Datasets loaded successfully")
        
        # Process data similar to training script
        dark_patterns_df = dark_patterns_df.dropna(subset=['Pattern String'])
        normie_df = normie_df[normie_df['classification'] == 0]
        dataset_df = dataset_df.dropna(subset=['text'])
        
        # Combine datasets
        combined_data = []
        
        # Add dark patterns
        for _, row in dark_patterns_df.iterrows():
            combined_data.append({
                'text': row['Pattern String'],
                'category': row['Pattern Category']
            })
        
        # Add normal patterns
        for _, row in normie_df.iterrows():
            combined_data.append({
                'text': row['Pattern String'],
                'category': 'Not Dark Pattern'
            })
        
        # Add dataset patterns
        for _, row in dataset_df.iterrows():
            combined_data.append({
                'text': row['text'],
                'category': row['Pattern Category']
            })
        
        # Create DataFrame
        df = pd.DataFrame(combined_data)
        df = df.drop_duplicates(subset=['text'])
        df = df[df['text'].str.len() >= 10]
        
        # Map categories
        category_mapping = {
            'Urgency': 'Urgency',
            'Scarcity': 'Scarcity', 
            'Social Proof': 'Social Proof',
            'Misdirection': 'Misdirection',
            'Forced Action': 'Forced Action',
            'Obstruction': 'Obstruction',
            'Sneaking': 'Sneaking',
            'Hidden Costs': 'Hidden Costs',
            'Not Dark Pattern': 'Not Dark Pattern'
        }
        
        df['category'] = df['category'].map(category_mapping).fillna('Not Dark Pattern')
        
        print(f"Original dataset size: {len(df)}")
        print("Original category distribution:")
        print(df['category'].value_counts())
        
        # Test balancing
        print("\nTesting data balancing...")
        balancer = AdvancedDataBalancer(random_state=42)
        
        # Test different methods
        methods = ['hybrid', 'augmentation', 'undersample']
        
        for method in methods:
            print(f"\n--- Testing {method} method ---")
            try:
                balanced_df = balancer.balance_dataset(df, target_samples_per_class=50, method=method)
                print(f"Balanced dataset size: {len(balanced_df)}")
                print("Balanced category distribution:")
                print(balanced_df['category'].value_counts())
                
                # Save sample
                balanced_df.to_csv(f'data/processed/test_balanced_{method}.csv', index=False)
                print(f"✅ {method} method successful - saved to test_balanced_{method}.csv")
                
            except Exception as e:
                print(f"❌ {method} method failed: {e}")
        
        print("\n🎉 Data balancing test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_balancing()
