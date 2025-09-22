"""
Advanced Data Balancing for Ethical Eye Extension
Research Project: Explainable AI for Dark Pattern Detection

This module implements advanced data balancing techniques including SMOTE,
text augmentation, and synthetic data generation for dark pattern classification.
"""

import pandas as pd
import numpy as np
from collections import Counter
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import nltk
from nltk.corpus import wordnet
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class AdvancedDataBalancer:
    """Advanced data balancing for text classification"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Text augmentation parameters
        self.synonym_dict = self._build_synonym_dictionary()
        self.insertion_words = [
            'exclusive', 'special', 'amazing', 'incredible', 'fantastic',
            'limited', 'rare', 'unique', 'premium', 'vip'
        ]
    
    def _build_synonym_dictionary(self):
        """Build a comprehensive synonym dictionary for dark pattern terms"""
        return {
            'hurry': ['rush', 'quick', 'fast', 'urgent', 'immediate'],
            'limited': ['restricted', 'scarce', 'few', 'rare', 'exclusive'],
            'only': ['just', 'merely', 'simply', 'barely', 'solely'],
            'left': ['remaining', 'available', 'stock', 'inventory'],
            'stock': ['inventory', 'supply', 'quantity', 'items'],
            'time': ['moment', 'period', 'duration', 'instant'],
            'offer': ['deal', 'promotion', 'discount', 'bargain'],
            'sale': ['discount', 'bargain', 'clearance', 'reduction'],
            'join': ['become', 'sign up', 'register', 'enroll'],
            'customers': ['users', 'buyers', 'clients', 'members'],
            'exclusive': ['special', 'unique', 'premium', 'vip'],
            'deal': ['offer', 'bargain', 'discount', 'promotion'],
            'save': ['discount', 'reduce', 'cut', 'lower'],
            'free': ['complimentary', 'gratis', 'no cost', 'zero cost'],
            'guarantee': ['promise', 'assurance', 'warranty', 'pledge'],
            'instant': ['immediate', 'instantaneous', 'quick', 'fast'],
            'secret': ['hidden', 'confidential', 'private', 'exclusive'],
            'proven': ['tested', 'verified', 'confirmed', 'validated'],
            'powerful': ['strong', 'effective', 'potent', 'influential'],
            'revolutionary': ['innovative', 'groundbreaking', 'cutting-edge', 'advanced']
        }
    
    def balance_dataset(self, df, target_samples_per_class=100, method='hybrid'):
        """
        Balance the dataset using various techniques
        
        Args:
            df: DataFrame with 'text' and 'category' columns
            target_samples_per_class: Target number of samples per class
            method: Balancing method ('smote', 'augmentation', 'hybrid', 'undersample')
        
        Returns:
            Balanced DataFrame
        """
        print(f"Original dataset size: {len(df)}")
        print(f"Original category distribution:")
        print(df['category'].value_counts())
        
        if method == 'smote':
            return self._balance_with_smote(df, target_samples_per_class)
        elif method == 'augmentation':
            return self._balance_with_augmentation(df, target_samples_per_class)
        elif method == 'hybrid':
            return self._balance_hybrid(df, target_samples_per_class)
        elif method == 'undersample':
            return self._balance_with_undersampling(df, target_samples_per_class)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _balance_with_smote(self, df, target_samples):
        """Balance using SMOTE on TF-IDF features"""
        print("Balancing with SMOTE...")
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['text'])
        
        # Reduce dimensionality for SMOTE
        svd = TruncatedSVD(n_components=100, random_state=self.random_state)
        X_reduced = svd.fit_transform(X)
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state, k_neighbors=1)
        X_balanced, y_balanced = smote.fit_resample(X_reduced, df['category'])
        
        # Reconstruct text features (approximation)
        X_reconstructed = svd.inverse_transform(X_balanced)
        
        # Create balanced DataFrame
        balanced_data = []
        for i, (features, category) in enumerate(zip(X_reconstructed, y_balanced)):
            # Find most similar original text
            similarity_scores = np.dot(X_reconstructed[i], X_reduced.T)
            most_similar_idx = np.argmax(similarity_scores)
            
            balanced_data.append({
                'text': df.iloc[most_similar_idx]['text'],
                'category': category
            })
        
        balanced_df = pd.DataFrame(balanced_data)
        print(f"SMOTE balanced dataset size: {len(balanced_df)}")
        return balanced_df
    
    def _balance_with_augmentation(self, df, target_samples):
        """Balance using text augmentation"""
        print("Balancing with text augmentation...")
        
        balanced_data = []
        
        for category in df['category'].unique():
            category_data = df[df['category'] == category].copy()
            current_count = len(category_data)
            
            print(f"Processing {category}: {current_count} samples")
            
            if current_count < target_samples:
                # Oversample with augmentation
                augmented_data = self._augment_category_data(category_data, target_samples)
                balanced_data.append(augmented_data)
            else:
                # Use original data
                balanced_data.append(category_data)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"Augmentation balanced dataset size: {len(balanced_df)}")
        return balanced_df
    
    def _balance_hybrid(self, df, target_samples):
        """Hybrid balancing: undersample majority, augment minority"""
        print("Balancing with hybrid method...")
        
        category_counts = df['category'].value_counts()
        median_count = int(category_counts.median())
        target_samples = max(min(target_samples, median_count * 2), 50)
        
        print(f"Target samples per category: {target_samples}")
        
        balanced_data = []
        
        for category in df['category'].unique():
            category_data = df[df['category'] == category].copy()
            current_count = len(category_data)
            
            if current_count > target_samples * 1.5:
                # Undersample
                category_data = category_data.sample(n=target_samples, random_state=self.random_state)
            elif current_count < target_samples:
                # Augment
                category_data = self._augment_category_data(category_data, target_samples)
            
            balanced_data.append(category_data)
            print(f"{category}: {len(category_data)} samples")
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"Hybrid balanced dataset size: {len(balanced_df)}")
        return balanced_df
    
    def _balance_with_undersampling(self, df, target_samples):
        """Balance by undersampling majority classes"""
        print("Balancing with undersampling...")
        
        # Find the minimum count among minority classes
        category_counts = df['category'].value_counts()
        min_count = category_counts.min()
        target_samples = min(target_samples, min_count)
        
        print(f"Target samples per category: {target_samples}")
        
        balanced_data = []
        for category in df['category'].unique():
            category_data = df[df['category'] == category].copy()
            if len(category_data) > target_samples:
                category_data = category_data.sample(n=target_samples, random_state=self.random_state)
            balanced_data.append(category_data)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"Undersampling balanced dataset size: {len(balanced_df)}")
        return balanced_df
    
    def _augment_category_data(self, category_data, target_count):
        """Augment data for a specific category"""
        current_count = len(category_data)
        if current_count >= target_count:
            return category_data
        
        samples_needed = target_count - current_count
        augmented_data = []
        
        for _ in range(samples_needed):
            # Select random sample to augment
            original_sample = category_data.sample(n=1, random_state=None).iloc[0]
            
            # Apply multiple augmentation techniques
            augmented_text = self._advanced_text_augmentation(original_sample['text'])
            
            # Create new sample
            new_sample = original_sample.copy()
            new_sample['text'] = augmented_text
            augmented_data.append(new_sample)
        
        # Combine original and augmented data
        augmented_df = pd.DataFrame(augmented_data)
        result_df = pd.concat([category_data, augmented_df], ignore_index=True)
        
        return result_df
    
    def _advanced_text_augmentation(self, text):
        """Apply advanced text augmentation techniques"""
        words = text.split()
        
        if len(words) <= 1:
            return text
        
        # Choose augmentation technique
        augmentation_type = random.choice([
            'synonym_replacement', 'word_shuffle', 'word_insertion', 
            'word_deletion', 'paraphrase', 'intensity_change'
        ])
        
        if augmentation_type == 'synonym_replacement':
            return self._synonym_replacement(words)
        elif augmentation_type == 'word_shuffle':
            return self._word_shuffle(words)
        elif augmentation_type == 'word_insertion':
            return self._word_insertion(words)
        elif augmentation_type == 'word_deletion':
            return self._word_deletion(words)
        elif augmentation_type == 'paraphrase':
            return self._paraphrase_text(words)
        elif augmentation_type == 'intensity_change':
            return self._intensity_change(words)
        
        return ' '.join(words)
    
    def _synonym_replacement(self, words):
        """Replace words with synonyms"""
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in self.synonym_dict:
                synonym = random.choice(self.synonym_dict[word_lower])
                # Preserve original capitalization and punctuation
                if word[0].isupper():
                    synonym = synonym.capitalize()
                if word.endswith(('.', ',', '!', '?')):
                    synonym += word[-1]
                words[i] = synonym
                break
        return ' '.join(words)
    
    def _word_shuffle(self, words):
        """Shuffle words while preserving structure"""
        if len(words) <= 3:
            return ' '.join(words)
        
        # Keep first and last words, shuffle middle
        if len(words) > 4:
            middle_words = words[1:-1]
            random.shuffle(middle_words)
            words = [words[0]] + middle_words + [words[-1]]
        
        return ' '.join(words)
    
    def _word_insertion(self, words):
        """Insert relevant words"""
        if len(words) <= 2:
            return ' '.join(words)
        
        insert_word = random.choice(self.insertion_words)
        insert_pos = random.randint(1, len(words)-1)
        words.insert(insert_pos, insert_word)
        
        return ' '.join(words)
    
    def _word_deletion(self, words):
        """Delete non-essential words"""
        if len(words) <= 3:
            return ' '.join(words)
        
        # Don't delete first or last word, or words with special characters
        deletable_indices = []
        for i in range(1, len(words)-1):
            word = words[i].lower()
            if not any(char in word for char in ['!', '?', '.', ',']) and len(word) > 2:
                deletable_indices.append(i)
        
        if deletable_indices:
            delete_idx = random.choice(deletable_indices)
            words.pop(delete_idx)
        
        return ' '.join(words)
    
    def _paraphrase_text(self, words):
        """Simple paraphrasing for dark pattern terms"""
        paraphrases = {
            'hurry': 'act fast',
            'limited time': 'time running out',
            'only': 'just',
            'left in stock': 'remaining',
            'exclusive offer': 'special deal',
            'don\'t miss': 'grab this',
            'act now': 'get it now',
            'limited quantity': 'few remaining'
        }
        
        text = ' '.join(words)
        for original, paraphrase in paraphrases.items():
            if original in text.lower():
                text = text.replace(original, paraphrase)
                break
        
        return text
    
    def _intensity_change(self, words):
        """Change intensity of words"""
        intensity_changes = {
            'hurry': 'rush',
            'limited': 'scarce',
            'exclusive': 'special',
            'amazing': 'great',
            'incredible': 'wonderful',
            'fantastic': 'excellent'
        }
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in intensity_changes:
                new_word = intensity_changes[word_lower]
                # Preserve original formatting
                if word[0].isupper():
                    new_word = new_word.capitalize()
                if word.endswith(('.', ',', '!', '?')):
                    new_word += word[-1]
                words[i] = new_word
                break
        
        return ' '.join(words)

def main():
    """Test the data balancer"""
    # Create sample data
    sample_data = {
        'text': [
            'Hurry! Only 2 left in stock!',
            'Limited time offer!',
            'Join 10,000+ customers',
            'Free shipping on orders over $50',
            'Welcome to our website',
            'Thank you for your purchase',
            'Contact us for support',
            'Don\'t miss this exclusive deal',
            'Act now before it\'s too late!',
            'Only 3 items remaining'
        ],
        'category': [
            'Urgency', 'Urgency', 'Social Proof', 'Not Dark Pattern',
            'Not Dark Pattern', 'Not Dark Pattern', 'Not Dark Pattern',
            'Scarcity', 'Urgency', 'Scarcity'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test balancer
    balancer = AdvancedDataBalancer()
    balanced_df = balancer.balance_dataset(df, target_samples_per_class=20, method='hybrid')
    
    print("\nBalanced dataset:")
    print(balanced_df['category'].value_counts())

if __name__ == "__main__":
    main()
