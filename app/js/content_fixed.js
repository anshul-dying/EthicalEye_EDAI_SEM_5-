const endpoint = "http://127.0.0.1:5000/analyze";
const descriptions = {
    "Sneaking": "Coerces users to act in ways that they would not normally act by obscuring information.",
    "Urgency": "Places deadlines on things to make them appear more desirable",
    "Misdirection": "Aims to deceptively incline a user towards one choice over the other.",
    "Social Proof": "Gives the perception that a given action or product has been approved by other people.",
    "Scarcity": "Tries to increase the value of something by making it appear to be limited in availability.",
    "Obstruction": "Tries to make an action more difficult so that a user is less likely to do that action.",
    "Forced Action": "Forces a user to complete extra, unrelated tasks to do something that should be simple.",
    "Hidden Costs": "Conceals or downplays additional fees and charges to mislead users.",
    "Not Dark Pattern": "Normal, non-manipulative content that doesn't use deceptive design."
};

// Function to extract text segments from DOM elements
function segments(element) {
    const textElements = [];

    function traverse(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            const text = node.textContent.trim();
            if (text.length > 0) {
                // Find the parent element that contains this text
                let parent = node.parentElement;
                if (parent && !textElements.includes(parent)) {
                    textElements.push(parent);
                }
            }
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            // Skip script and style elements
            if (node.tagName !== 'SCRIPT' && node.tagName !== 'STYLE') {
                for (let child of node.childNodes) {
                    traverse(child);
                }
            }
        }
    }

    traverse(element);
    return textElements;
}

// Global variables to store detected patterns
let detectedPatterns = [];
let currentPatternIndex = -1;
let allElements = [];

// Make scrape function globally accessible
window.scrape = function scrape() {
    console.log("🔍 Ethical Eye: scrape() function called!");

    if (document.getElementById("insite_count")) {
        console.log("🔍 Ethical Eye: Analysis already in progress, skipping...");
        return;
    }

    console.log("🔍 Ethical Eye: Starting text extraction...");
    allElements = segments(document.body);
    console.log("🔍 Ethical Eye: Found", allElements.length, "text elements");

    let filtered_elements = [];

    for (let i = 0; i < allElements.length; i++) {
        let text = allElements[i].innerText.trim().replace(/\t/g, " ");
        if (text.length == 0) {
            continue;
        }

        // Filter out common UI elements and instructions
        const lowerText = text.toLowerCase();
        const shouldSkip =
            lowerText.includes('click') && lowerText.includes('button') ||
            lowerText.includes('test') && lowerText.includes('page') ||
            lowerText.includes('expected') && lowerText.includes('results') ||
            lowerText.includes('step') && lowerText.includes('instructions') ||
            lowerText.includes('console') && lowerText.includes('message') ||
            lowerText.includes('devtools') ||
            lowerText.includes('f12') ||
            lowerText.includes('reload') && lowerText.includes('extension') ||
            lowerText.includes('chrome://extensions') ||
            lowerText.includes('should') && lowerText.includes('highlighted') ||
            lowerText.includes('welcome') && lowerText.includes('website') ||
            text.length < 10; // Skip very short text

        if (shouldSkip) {
            continue;
        }

        filtered_elements.push(text);
    }

    // Prepare segments for API
    const apiSegments = filtered_elements.map((text, index) => ({
        text: text,
        element_id: `segment_${index}`,
        position: { x: 0, y: 0 } // Simplified position
    }));

    console.log("🔍 Ethical Eye: Starting analysis...");
    console.log("🔍 Sending", apiSegments.length, "segments to API");
    console.log("🔍 API endpoint:", endpoint);
    console.log("🔍 Request payload:", JSON.stringify({
        segments: apiSegments,
        confidence_threshold: 0.5
    }));

    fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            segments: apiSegments,
            confidence_threshold: 0.5  // Increased from 0.15 to 0.5 (50%)
        }),
    })
        .then((resp) => {
            console.log("🔍 API Response status:", resp.status);
            return resp.json();
        })
        .then((data) => {
            console.log("🔍 API Response data:", data);

            // Clear previous patterns
            detectedPatterns = [];
            currentPatternIndex = -1;

            let element_index = 0;
            let filtered_index = 0;

            for (let i = 0; i < allElements.length; i++) {
                let text = allElements[i].innerText.trim().replace(/\t/g, " ");
                if (text.length == 0) {
                    continue;
                }

                const result = data.results[filtered_index];
                if (result && result.is_dark_pattern) {
                    detectedPatterns.push({
                        element: allElements[i],
                        category: result.category,
                        confidence: result.confidence,
                        explanation: result.explanation,
                        text: text
                    });
                }
                filtered_index++;
            }

            console.log("🔍 Ethical Eye: Found", detectedPatterns.length, "dark patterns");

            // Store patterns globally for navigation
            window.ethicalEyePatterns = detectedPatterns;
            window.ethicalEyeCurrentIndex = -1;

            // Send count to popup
            let g = document.createElement("div");
            g.id = "insite_count";
            g.value = detectedPatterns.length;
            g.style.opacity = 0;
            g.style.position = "fixed";
            document.body.appendChild(g);
            sendDarkPatterns(detectedPatterns.length);

            // Send patterns data to popup
            sendPatternsToPopup(detectedPatterns);
        })
        .catch((error) => {
            console.error("❌ Ethical Eye API Error:", error);
            console.error("❌ Make sure the API server is running: python api/ethical_eye_api.py");
            alert("❌ Ethical Eye Error: " + error.message + "\n\nMake sure the API server is running!");
        });
}

function highlight(element, type, confidence, explanation) {
    element.classList.add("insite-highlight");

    let body = document.createElement("span");
    body.classList.add("insite-highlight-body");

    /* header */
    let header = document.createElement("div");
    header.classList.add("modal-header");
    let headerText = document.createElement("h1");
    headerText.style.border = "1px solid #FF0000";
    headerText.style.borderRadius = "10px";
    headerText.style.backgroundColor = "#A28CEC";
    headerText.style.width = "fit-content";
    headerText.innerHTML = type + " Pattern";
    header.appendChild(headerText);
    body.appendChild(header);

    let content = document.createElement("div");
    content.classList.add("modal-content");
    content.style.border = "1px solid #FF0000";
    content.style.borderRadius = "10px";
    content.style.backgroundColor = "#A28CEC";
    content.style.width = "fit-content";

    // Enhanced content with confidence and explanation
    let contentHTML = `<strong>${type} Pattern</strong><br>`;
    contentHTML += `Confidence: ${(confidence * 100).toFixed(1)}%<br><br>`;
    contentHTML += `<strong>Description:</strong><br>${descriptions[type]}<br><br>`;

    if (explanation) {
        contentHTML += `<strong>AI Explanation:</strong><br>${explanation}`;
    }

    content.innerHTML = contentHTML;
    body.appendChild(content);

    element.appendChild(body);
}

function sendDarkPatterns(number) {
    chrome.runtime.sendMessage({
        message: "update_current_count",
        count: number,
    });
}

function sendPatternsToPopup(patterns) {
    chrome.runtime.sendMessage({
        message: "patterns_detected",
        patterns: patterns.map(p => ({
            category: p.category,
            confidence: p.confidence,
            explanation: p.explanation,
            text: p.text.substring(0, 100) + (p.text.length > 100 ? "..." : "")
        }))
    });
}

// Make highlightPattern function globally accessible
window.highlightPattern = function highlightPattern(index) {
    // Clear all previous highlights
    clearAllHighlights();

    if (index >= 0 && index < detectedPatterns.length) {
        const pattern = detectedPatterns[index];
        highlight(pattern.element, pattern.category, pattern.confidence, pattern.explanation);

        // Scroll to the highlighted element
        pattern.element.scrollIntoView({ behavior: 'smooth', block: 'center' });

        currentPatternIndex = index;
        window.ethicalEyeCurrentIndex = index;

        console.log(`🔍 Ethical Eye: Highlighted pattern ${index + 1}/${detectedPatterns.length}: ${pattern.category}`);
    }
}

function clearAllHighlights() {
    // Remove all existing highlights
    const highlightedElements = document.querySelectorAll('.insite-highlight');
    highlightedElements.forEach(element => {
        element.classList.remove('insite-highlight');
        // Remove highlight body if it exists
        const highlightBody = element.querySelector('.insite-highlight-body');
        if (highlightBody) {
            highlightBody.remove();
        }
    });
}

// Make navigation functions globally accessible
window.nextPattern = function nextPattern() {
    if (detectedPatterns.length === 0) return;

    const nextIndex = (currentPatternIndex + 1) % detectedPatterns.length;
    highlightPattern(nextIndex);

    // Send update to popup
    chrome.runtime.sendMessage({
        message: "pattern_navigated",
        currentIndex: nextIndex,
        totalPatterns: detectedPatterns.length
    });
}

window.previousPattern = function previousPattern() {
    if (detectedPatterns.length === 0) return;

    const prevIndex = currentPatternIndex <= 0 ? detectedPatterns.length - 1 : currentPatternIndex - 1;
    highlightPattern(prevIndex);

    // Send update to popup
    chrome.runtime.sendMessage({
        message: "pattern_navigated",
        currentIndex: prevIndex,
        totalPatterns: detectedPatterns.length
    });
}

// Add debugging to confirm content script is loaded
console.log("🔍 Ethical Eye: Content script loaded successfully!");
console.log("🔍 Ethical Eye: Content script version 3.0 - FIXED");
console.log("🔍 Ethical Eye: API endpoint:", endpoint);

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    console.log("🔍 Ethical Eye: Received message:", request.message);

    if (request.message === "analyze_site") {
        console.log("🔍 Ethical Eye: Starting site analysis...");
        // Call scrape function immediately
        try {
            scrape();
            console.log("🔍 Ethical Eye: Scrape function called successfully");
        } catch (error) {
            console.error("❌ Ethical Eye: Error calling scrape function:", error);
        }
        sendResponse({ status: "analysis_started" });
    } else if (request.message === "popup_open") {
        console.log("🔍 Ethical Eye: Popup opened");
        let element = document.getElementById("insite_count");
        if (element) {
            sendDarkPatterns(element.value);
        }
        // Send current patterns if available
        if (detectedPatterns.length > 0) {
            sendPatternsToPopup(detectedPatterns);
        }
        sendResponse({ status: "popup_opened" });
    } else if (request.message === "next_pattern") {
        console.log("🔍 Ethical Eye: Next pattern requested");
        nextPattern();
        sendResponse({ status: "pattern_navigated", currentIndex: currentPatternIndex });
    } else if (request.message === "previous_pattern") {
        console.log("🔍 Ethical Eye: Previous pattern requested");
        previousPattern();
        sendResponse({ status: "pattern_navigated", currentIndex: currentPatternIndex });
    } else if (request.message === "highlight_pattern") {
        console.log("🔍 Ethical Eye: Highlight specific pattern:", request.index);
        highlightPattern(request.index);
        sendResponse({ status: "pattern_highlighted", currentIndex: currentPatternIndex });
    } else if (request.message === "clear_highlights") {
        console.log("🔍 Ethical Eye: Clear highlights requested");
        clearAllHighlights();
        currentPatternIndex = -1;
        sendResponse({ status: "highlights_cleared" });
    }

    return false; // Don't keep channel open to avoid port closing
});
