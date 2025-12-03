// Results page script for Ethical Eye
console.log("Ethical Eye: Results page script loading...");

document.addEventListener('DOMContentLoaded', function() {
    console.log("Ethical Eye: DOM ready, loading results...");
    
    loadResults();
});

function loadResults() {
    const loading = document.getElementById('loading');
    const content = document.getElementById('content');
    const noResults = document.getElementById('no-results');
    
    // Get stored results from chrome.storage
    chrome.storage.local.get(['ethicalEyeFullResults', 'ethicalEyeResultsTimestamp'], function(data) {
        loading.style.display = 'none';
        
        if (!data.ethicalEyeFullResults || data.ethicalEyeFullResults.length === 0) {
            noResults.style.display = 'block';
            return;
        }
        
        const results = data.ethicalEyeFullResults;
        const timestamp = data.ethicalEyeResultsTimestamp;
        
        console.log("Ethical Eye: Loaded", results.length, "results");
        
        // Display summary
        displaySummary(results);
        
        // Display patterns
        displayPatterns(results);
        
        // Display timestamp
        if (timestamp) {
            const timestampEl = document.getElementById('timestamp');
            const date = new Date(timestamp);
            timestampEl.textContent = `Analysis completed: ${date.toLocaleString()}`;
        }
        
        content.style.display = 'block';
    });
}

function displaySummary(results) {
    const summaryContainer = document.getElementById('summary');
    
    // Count patterns by category
    const categoryCount = {};
    let totalConfidence = 0;
    
    results.forEach(result => {
        categoryCount[result.category] = (categoryCount[result.category] || 0) + 1;
        totalConfidence += result.confidence;
    });
    
    const avgConfidence = (totalConfidence / results.length * 100).toFixed(1);
    const categories = Object.keys(categoryCount).length;
    
    summaryContainer.innerHTML = `
        <div class="summary-card">
            <h3>${results.length}</h3>
            <p>Dark Patterns Detected</p>
        </div>
        <div class="summary-card">
            <h3>${categories}</h3>
            <p>Different Categories</p>
        </div>
        <div class="summary-card">
            <h3>${avgConfidence}%</h3>
            <p>Average Confidence</p>
        </div>
    `;
}

function displayPatterns(results) {
    const container = document.getElementById('patterns-container');
    
    container.innerHTML = '';
    
    results.forEach((result, index) => {
        const card = createPatternCard(result, index + 1);
        container.appendChild(card);
    });
}

function createPatternCard(result, index) {
    const card = document.createElement('div');
    card.className = 'pattern-card';
    
    // Create top words badges
    const topWordsHTML = result.top_words && result.top_words.length > 0
        ? result.top_words.map(([word, score]) => 
            `<div class="word-badge">
                <span>${word}</span>
                <span class="word-score">${score.toFixed(3)}</span>
            </div>`
        ).join('')
        : '<p style="color: #999; font-style: italic;">No SHAP data available</p>';
    
    // Create SHAP visualization
    const shapVisualization = createSHAPVisualization(result);
    
    card.innerHTML = `
        <div class="pattern-header">
            <div class="pattern-category">${index}. ${result.category}</div>
            <div class="pattern-confidence">${(result.confidence * 100).toFixed(1)}%</div>
        </div>
        
        <div class="pattern-text">${escapeHtml(result.text)}</div>
        
        ${result.pattern_description ? `
            <div class="pattern-description">
                <strong>Description:</strong> ${escapeHtml(result.pattern_description)}
            </div>
        ` : ''}
        
        ${result.explanation ? `
            <div class="pattern-explanation">
                <strong>AI Explanation:</strong> ${escapeHtml(result.explanation)}
            </div>
        ` : ''}
        
        <div class="shap-section">
            <h3>Top Contributing Words (SHAP Importance)</h3>
            <div class="top-words">
                ${topWordsHTML}
            </div>
            
            ${shapVisualization ? `
                <div class="shap-visualization">
                    <h4 style="margin-bottom: 10px; color: #666;">Token-level SHAP Values</h4>
                    ${shapVisualization}
                </div>
            ` : ''}
            
            ${!result.shap_computed && result.top_words && result.top_words.length > 0 ? `
                <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin-top: 10px; border-radius: 5px; font-size: 0.9em; color: #856404;">
                    <strong>Note:</strong> SHAP computation unavailable. Showing keyword-based indicators instead.
                </div>
            ` : ''}
        </div>
    `;
    
    return card;
}

function createSHAPVisualization(result) {
    if (!result.tokens || !result.shap_values || 
        result.tokens.length === 0 || result.shap_values.length === 0) {
        return null;
    }
    
    // Find max absolute value for normalization
    const maxAbsValue = Math.max(...result.shap_values.map(v => Math.abs(v)));
    
    if (maxAbsValue === 0) {
        return '<p style="color: #999; font-style: italic;">No SHAP values available</p>';
    }
    
    // Only show visualization if SHAP was actually computed (not keyword-based)
    if (!result.shap_computed) {
        return '<p style="color: #999; font-style: italic;">SHAP values not computed (using keyword-based fallback)</p>';
    }
    
    // Create visualization rows
    const rows = [];
    const minLength = Math.min(result.tokens.length, result.shap_values.length);
    
    for (let i = 0; i < minLength; i++) {
        const token = result.tokens[i];
        const value = result.shap_values[i];
        
        // Skip special tokens
        if (token.match(/^\[(CLS|SEP|PAD|UNK)\]$/) || token.trim() === '') {
            continue;
        }
        
        // Normalize value for display (0-100%)
        const normalizedValue = Math.abs(value) / maxAbsValue * 100;
        const isPositive = value >= 0;
        
        rows.push(`
            <div class="token-row">
                <div class="token-text">${escapeHtml(token)}</div>
                <div class="shap-bar">
                    <div class="shap-bar-fill ${isPositive ? 'positive' : 'negative'}" 
                         style="width: ${normalizedValue}%"></div>
                </div>
                <div class="shap-value">${value >= 0 ? '+' : ''}${value.toFixed(4)}</div>
            </div>
        `);
    }
    
    return rows.length > 0 ? rows.join('') : '<p style="color: #999; font-style: italic;">No valid tokens to display</p>';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

console.log("Ethical Eye: Results page script loaded!");

