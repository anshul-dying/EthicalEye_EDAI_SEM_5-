"""
Layout-Based Feature Extraction Using HTML Structure

Extracts features for detecting visual misdirection:
- Button order
- Font size differences
- Hidden elements (via CSS)
- Visual hierarchy tricks
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from bs4 import BeautifulSoup
import cssutils
import warnings

# Suppress cssutils warnings
warnings.filterwarnings('ignore')
cssutils.log.setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """
    Analyzes HTML/CSS structure to detect visual misdirection patterns
    """
    
    def __init__(self):
        self.css_parser = cssutils.CSSParser()
    
    def analyze_html(self, html_content: str, css_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze HTML structure and CSS for dark pattern indicators
        
        Args:
            html_content: HTML string
            css_content: Optional CSS string (can also be embedded in HTML)
        
        Returns:
            Dictionary with extracted features and detected patterns
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract CSS from style tags if not provided separately
            if css_content is None:
                style_tags = soup.find_all('style')
                css_content = '\n'.join([tag.string or '' for tag in style_tags])
            
            # Extract features
            features = {
                'button_order': self._analyze_button_order(soup),
                'font_size_differences': self._analyze_font_sizes(soup, css_content),
                'hidden_elements': self._detect_hidden_elements(soup, css_content),
                'visual_hierarchy': self._analyze_visual_hierarchy(soup, css_content),
                'suspicious_patterns': []
            }
            
            # Detect patterns based on features
            patterns = self._detect_patterns(features)
            features['suspicious_patterns'] = patterns
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing HTML layout: {e}")
            return {
                'error': str(e),
                'button_order': [],
                'font_size_differences': {},
                'hidden_elements': [],
                'visual_hierarchy': {},
                'suspicious_patterns': []
            }
    
    def _analyze_button_order(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Analyze button order and detect misdirection patterns
        
        Returns:
            List of button information with order analysis
        """
        buttons = []
        
        # Find all button-like elements
        button_selectors = [
            soup.find_all('button'),
            soup.find_all('input', type='button'),
            soup.find_all('input', type='submit'),
            soup.find_all('a', class_=re.compile(r'button|btn|cta', re.I)),
            soup.find_all(attrs={'role': 'button'})
        ]
        
        all_buttons = []
        for button_list in button_selectors:
            all_buttons.extend(button_list)
        
        # Remove duplicates
        seen = set()
        unique_buttons = []
        for btn in all_buttons:
            btn_id = id(btn)
            if btn_id not in seen:
                seen.add(btn_id)
                unique_buttons.append(btn)
        
        # Extract button information
        for idx, btn in enumerate(unique_buttons):
            text = btn.get_text(strip=True)
            classes = ' '.join(btn.get('class', []))
            btn_id = btn.get('id', '')
            
            # Check for suspicious text patterns
            suspicious_keywords = [
                'no thanks', 'skip', 'continue', 'maybe later', 
                'not interested', 'decline', 'cancel'
            ]
            is_suspicious = any(kw in text.lower() for kw in suspicious_keywords)
            
            buttons.append({
                'index': idx,
                'text': text,
                'classes': classes,
                'id': btn_id,
                'tag': btn.name,
                'is_suspicious': is_suspicious,
                'position': self._get_element_position(btn)
            })
        
        return buttons
    
    def _analyze_font_sizes(self, soup: BeautifulSoup, css_content: str) -> Dict[str, Any]:
        """
        Analyze font size differences to detect visual hierarchy manipulation
        
        Returns:
            Dictionary with font size statistics
        """
        font_sizes = []
        
        # Parse CSS to get font-size rules
        css_rules = {}
        try:
            if css_content:
                sheet = self.css_parser.parseString(css_content)
                for rule in sheet:
                    if hasattr(rule, 'selectorText') and hasattr(rule, 'style'):
                        selector = rule.selectorText
                        if rule.style.getPropertyValue('font-size'):
                            size = rule.style.getPropertyValue('font-size')
                            css_rules[selector] = size
        except Exception as e:
            logger.debug(f"CSS parsing error: {e}")
        
        # Extract font sizes from inline styles and CSS
        elements = soup.find_all(['p', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'button'])
        
        for elem in elements:
            # Check inline style
            inline_style = elem.get('style', '')
            if 'font-size' in inline_style:
                match = re.search(r'font-size:\s*([\d.]+(?:px|em|pt|%))', inline_style)
                if match:
                    font_sizes.append({
                        'element': elem.name,
                        'text': elem.get_text(strip=True)[:50],
                        'size': match.group(1),
                        'source': 'inline'
                    })
            
            # Check CSS classes
            classes = elem.get('class', [])
            for cls in classes:
                for selector, size in css_rules.items():
                    if cls in selector or f'.{cls}' in selector:
                        font_sizes.append({
                            'element': elem.name,
                            'text': elem.get_text(strip=True)[:50],
                            'size': size,
                            'source': 'css',
                            'selector': selector
                        })
                        break
        
        # Analyze font size distribution
        if not font_sizes:
            return {
                'elements': [],
                'size_range': None,
                'size_variance': 0,
                'suspicious': False
            }
        
        # Extract numeric values (simplified - assumes px)
        numeric_sizes = []
        for item in font_sizes:
            size_str = item['size']
            match = re.search(r'([\d.]+)', size_str)
            if match:
                numeric_sizes.append(float(match.group(1)))
        
        if numeric_sizes:
            size_range = max(numeric_sizes) - min(numeric_sizes)
            size_variance = sum((x - sum(numeric_sizes)/len(numeric_sizes))**2 for x in numeric_sizes) / len(numeric_sizes)
            
            # Flag suspicious if there's extreme size difference (likely misdirection)
            suspicious = size_range > 20 or size_variance > 100
            
            return {
                'elements': font_sizes[:20],  # Limit to first 20
                'size_range': float(size_range),
                'size_variance': float(size_variance),
                'suspicious': suspicious,
                'min_size': float(min(numeric_sizes)),
                'max_size': float(max(numeric_sizes))
            }
        
        return {
            'elements': font_sizes,
            'size_range': None,
            'size_variance': 0,
            'suspicious': False
        }
    
    def _detect_hidden_elements(self, soup: BeautifulSoup, css_content: str) -> List[Dict[str, Any]]:
        """
        Detect hidden elements that might contain important information
        
        Returns:
            List of hidden elements with their properties
        """
        hidden_elements = []
        
        # Parse CSS for display:none, visibility:hidden, opacity:0
        hidden_selectors = set()
        try:
            if css_content:
                sheet = self.css_parser.parseString(css_content)
                for rule in sheet:
                    if hasattr(rule, 'selectorText') and hasattr(rule, 'style'):
                        selector = rule.selectorText
                        style = rule.style
                        
                        display = style.getPropertyValue('display')
                        visibility = style.getPropertyValue('visibility')
                        opacity = style.getPropertyValue('opacity')
                        
                        if display == 'none' or visibility == 'hidden' or opacity == '0':
                            hidden_selectors.add(selector)
        except Exception as e:
            logger.debug(f"CSS parsing error: {e}")
        
        # Check inline styles
        all_elements = soup.find_all(True)  # All elements
        
        for elem in all_elements:
            is_hidden = False
            hidden_reason = []
            
            # Check inline style
            inline_style = elem.get('style', '').lower()
            if 'display:none' in inline_style or 'display: none' in inline_style:
                is_hidden = True
                hidden_reason.append('display:none (inline)')
            if 'visibility:hidden' in inline_style or 'visibility: hidden' in inline_style:
                is_hidden = True
                hidden_reason.append('visibility:hidden (inline)')
            if 'opacity:0' in inline_style or 'opacity: 0' in inline_style:
                is_hidden = True
                hidden_reason.append('opacity:0 (inline)')
            
            # Check CSS classes
            classes = elem.get('class', [])
            for cls in classes:
                for selector in hidden_selectors:
                    if cls in selector or f'.{cls}' in selector:
                        is_hidden = True
                        hidden_reason.append(f'CSS: {selector}')
                        break
            
            # Check HTML5 hidden attribute
            if elem.get('hidden') is not None:
                is_hidden = True
                hidden_reason.append('hidden attribute')
            
            if is_hidden:
                text = elem.get_text(strip=True)
                # Only flag if element has meaningful content
                if text and len(text) > 3:
                    hidden_elements.append({
                        'tag': elem.name,
                        'text': text[:100],
                        'id': elem.get('id', ''),
                        'classes': ' '.join(classes),
                        'hidden_reasons': hidden_reason,
                        'is_suspicious': self._is_suspicious_hidden_element(elem, text)
                    })
        
        return hidden_elements
    
    def _is_suspicious_hidden_element(self, elem: Any, text: str) -> bool:
        """Check if hidden element contains suspicious content"""
        suspicious_keywords = [
            'subscription', 'recurring', 'auto-renew', 'cancel', 
            'terms', 'conditions', 'fee', 'charge', 'cost',
            'checkbox', 'check', 'agree', 'accept'
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in suspicious_keywords)
    
    def _analyze_visual_hierarchy(self, soup: BeautifulSoup, css_content: str) -> Dict[str, Any]:
        """
        Analyze visual hierarchy tricks (z-index, positioning, etc.)
        
        Returns:
            Dictionary with hierarchy analysis
        """
        hierarchy_features = {
            'z_index_layers': [],
            'absolute_positions': [],
            'fixed_positions': [],
            'overlapping_elements': []
        }
        
        # Parse CSS for z-index and positioning
        z_index_rules = {}
        position_rules = {}
        
        try:
            if css_content:
                sheet = self.css_parser.parseString(css_content)
                for rule in sheet:
                    if hasattr(rule, 'selectorText') and hasattr(rule, 'style'):
                        selector = rule.selectorText
                        style = rule.style
                        
                        z_idx = style.getPropertyValue('z-index')
                        position = style.getPropertyValue('position')
                        
                        if z_idx and z_idx != 'auto':
                            z_index_rules[selector] = z_idx
                        if position in ['absolute', 'fixed', 'sticky']:
                            position_rules[selector] = position
        except Exception as e:
            logger.debug(f"CSS parsing error: {e}")
        
        # Check inline styles
        elements = soup.find_all(True)
        
        for elem in elements:
            inline_style = elem.get('style', '')
            classes = elem.get('class', [])
            
            # Check z-index
            z_idx = None
            if 'z-index' in inline_style:
                match = re.search(r'z-index:\s*([\d-]+)', inline_style)
                if match:
                    z_idx = int(match.group(1))
            else:
                for cls in classes:
                    for selector, z_val in z_index_rules.items():
                        if cls in selector or f'.{cls}' in selector:
                            try:
                                z_idx = int(z_val)
                            except:
                                pass
                            break
            
            if z_idx is not None:
                hierarchy_features['z_index_layers'].append({
                    'tag': elem.name,
                    'text': elem.get_text(strip=True)[:50],
                    'z_index': z_idx,
                    'id': elem.get('id', ''),
                    'classes': ' '.join(classes)
                })
            
            # Check positioning
            position = None
            if 'position' in inline_style:
                match = re.search(r'position:\s*(\w+)', inline_style)
                if match:
                    position = match.group(1)
            else:
                for cls in classes:
                    for selector, pos in position_rules.items():
                        if cls in selector or f'.{cls}' in selector:
                            position = pos
                            break
            
            if position == 'absolute':
                hierarchy_features['absolute_positions'].append({
                    'tag': elem.name,
                    'text': elem.get_text(strip=True)[:50],
                    'id': elem.get('id', '')
                })
            elif position == 'fixed':
                hierarchy_features['fixed_positions'].append({
                    'tag': elem.name,
                    'text': elem.get_text(strip=True)[:50],
                    'id': elem.get('id', '')
                })
        
        # Sort z-index layers
        hierarchy_features['z_index_layers'].sort(key=lambda x: x['z_index'], reverse=True)
        
        return hierarchy_features
    
    def _get_element_position(self, elem: Any) -> Dict[str, Optional[int]]:
        """Get approximate position of element (if available from parsing)"""
        # This is a simplified version - in a real browser, you'd use getBoundingClientRect()
        return {
            'x': None,
            'y': None
        }
    
    def _detect_patterns(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect dark patterns based on extracted features
        
        Returns:
            List of detected patterns with confidence scores
        """
        patterns = []
        
        # Pattern 1: Misdirection via button order
        buttons = features.get('button_order', [])
        if len(buttons) >= 2:
            # Check if suspicious button (like "No thanks") appears before important action
            suspicious_buttons = [b for b in buttons if b.get('is_suspicious', False)]
            if suspicious_buttons:
                # If suspicious button is first or early, it's likely misdirection
                first_suspicious_idx = min(b['index'] for b in suspicious_buttons)
                if first_suspicious_idx < len(buttons) / 2:
                    patterns.append({
                        'pattern': 'Button Order Misdirection',
                        'confidence': 0.7,
                        'description': 'Suspicious buttons (e.g., "No thanks") appear prominently before main actions',
                        'evidence': f"Found {len(suspicious_buttons)} suspicious buttons in first {len(buttons)} buttons"
                    })
        
        # Pattern 2: Font size manipulation
        font_analysis = features.get('font_size_differences', {})
        if font_analysis.get('suspicious', False):
            patterns.append({
                'pattern': 'Font Size Manipulation',
                'confidence': 0.65,
                'description': 'Extreme font size differences detected, likely used for visual misdirection',
                'evidence': f"Font size range: {font_analysis.get('size_range', 0):.1f}px, variance: {font_analysis.get('size_variance', 0):.1f}"
            })
        
        # Pattern 3: Hidden important information
        hidden = features.get('hidden_elements', [])
        suspicious_hidden = [h for h in hidden if h.get('is_suspicious', False)]
        if suspicious_hidden:
            patterns.append({
                'pattern': 'Hidden Important Information',
                'confidence': 0.8,
                'description': 'Important information (subscriptions, fees, terms) is hidden via CSS',
                'evidence': f"Found {len(suspicious_hidden)} suspicious hidden elements"
            })
        
        # Pattern 4: Visual hierarchy manipulation
        hierarchy = features.get('visual_hierarchy', {})
        high_z_index = [z for z in hierarchy.get('z_index_layers', []) if z.get('z_index', 0) > 100]
        if high_z_index:
            patterns.append({
                'pattern': 'Visual Hierarchy Manipulation',
                'confidence': 0.6,
                'description': 'Unusually high z-index values detected, possibly used to overlay important content',
                'evidence': f"Found {len(high_z_index)} elements with z-index > 100"
            })
        
        return patterns

