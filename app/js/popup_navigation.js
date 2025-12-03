// Navigation popup script for Ethical Eye
console.log("🔍 Ethical Eye: Navigation popup script loading...");

let detectedPatterns = [];
let currentPatternIndex = -1;

document.addEventListener('DOMContentLoaded', function () {
    console.log("🔍 Ethical Eye: DOM ready, setting up navigation popup...");

    // Get elements
    const analyzeButton = document.querySelector('.analyze-button');
    const prevButton = document.getElementById('prev-button');
    const nextButton = document.getElementById('next-button');
    const clearButton = document.getElementById('clear-button');
    const patternList = document.getElementById('pattern-list');
    const patternsContainer = document.getElementById('patterns-container');
    const patternCount = document.getElementById('pattern-count');
    const currentPatternInfo = document.getElementById('current-pattern-info');
    const flagButton = document.querySelector('.flag-button');
    const linkElement = document.querySelector('.link');

    // Setup analyze button
    if (analyzeButton) {
        analyzeButton.addEventListener('click', function () {
            console.log("🔍 Ethical Eye: ANALYZE SITE button clicked!");

            chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
                if (tabs && tabs[0]) {
                    const tab = tabs[0];
                    
                    // Check if it's a valid URL
                    if (tab.url.startsWith("chrome://") || tab.url.startsWith("edge://") || 
                        tab.url.startsWith("about:") || tab.url.startsWith("chrome-extension://") ||
                        tab.url.startsWith("moz-extension://")) {
                        alert("Cannot analyze system pages. Please navigate to a regular website.");
                        return;
                    }
                    
                    console.log("🔍 Ethical Eye: Sending analyze message to tab:", tab.id);

                    // First, ensure content script is loaded
                    chrome.scripting.executeScript({
                        target: { tabId: tab.id },
                        files: ['js/content_fixed.js']
                    }).then(() => {
                        console.log("✅ Ethical Eye: Content script ensured");
                        
                        // Wait a bit for script to initialize
                        setTimeout(() => {
                            // Try sending message
                            chrome.tabs.sendMessage(tab.id, { message: "analyze_site" }, function (response) {
                                if (chrome.runtime.lastError) {
                                    const errorMsg = chrome.runtime.lastError.message;
                                    console.log("ℹ️ Ethical Eye: Message send result:", errorMsg);
                                    
                                    // If message fails, try direct execution
                                    chrome.scripting.executeScript({
                                        target: { tabId: tab.id },
                                        function: function () {
                                            console.log("🔍 Ethical Eye: Direct execution - calling scrape...");
                                            if (typeof scrape === 'function') {
                                                scrape();
                                            } else {
                                                console.error("❌ Ethical Eye: scrape function not found");
                                            }
                                        }
                                    }).catch(error => {
                                        console.log("ℹ️ Ethical Eye: Direct execution error:", error.message);
                                        alert("Please refresh the page and try again.");
                                    });
                                } else {
                                    console.log("✅ Ethical Eye: Analysis started, response:", response);
                                }
                            });
                        }, 200);
                    }).catch(error => {
                        console.error("❌ Ethical Eye: Could not inject script:", error);
                        alert("Could not analyze this page. Please refresh and try again.");
                    });
                }
            });
        });
    }

    // Setup navigation buttons
    if (prevButton) {
        prevButton.addEventListener('click', function () {
            console.log("🔍 Ethical Eye: Previous pattern requested");
            chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
                if (tabs && tabs[0]) {
                    chrome.tabs.sendMessage(tabs[0].id, { message: "previous_pattern" }, function (response) {
                        if (chrome.runtime.lastError) {
                            const errorMsg = chrome.runtime.lastError.message;
                            // Only log if it's not a connection error (which is normal)
                            if (!errorMsg.includes("Could not establish connection") && 
                                !errorMsg.includes("Receiving end does not exist")) {
                                console.log("ℹ️ Ethical Eye: Navigation message:", errorMsg);
                            }
                        }
                    });
                }
            });
        });
    }

    if (nextButton) {
        nextButton.addEventListener('click', function () {
            console.log("🔍 Ethical Eye: Next pattern requested");
            chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
                if (tabs && tabs[0]) {
                    chrome.tabs.sendMessage(tabs[0].id, { message: "next_pattern" }, function (response) {
                        if (chrome.runtime.lastError) {
                            const errorMsg = chrome.runtime.lastError.message;
                            // Only log if it's not a connection error (which is normal)
                            if (!errorMsg.includes("Could not establish connection") && 
                                !errorMsg.includes("Receiving end does not exist")) {
                                console.log("ℹ️ Ethical Eye: Navigation message:", errorMsg);
                            }
                        }
                    });
                }
            });
        });
    }

    if (clearButton) {
        clearButton.addEventListener('click', function () {
            console.log("🔍 Ethical Eye: Clear highlights requested");
            chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
                if (tabs && tabs[0]) {
                    chrome.tabs.sendMessage(tabs[0].id, { message: "clear_highlights" }, function (response) {
                        if (chrome.runtime.lastError) {
                            const errorMsg = chrome.runtime.lastError.message;
                            // Only log if it's not a connection error (which is normal)
                            if (!errorMsg.includes("Could not establish connection") && 
                                !errorMsg.includes("Receiving end does not exist")) {
                                console.log("ℹ️ Ethical Eye: Clear message:", errorMsg);
                            }
                        }
                    });
                }
            });
            currentPatternIndex = -1;
            updateUI();
        });
    }

    // Setup view results button
    const viewResultsButton = document.getElementById('view-results-button');
    if (viewResultsButton) {
        viewResultsButton.addEventListener('click', function () {
            console.log("🔍 Ethical Eye: View full results requested");
            chrome.tabs.create({ url: chrome.runtime.getURL('results.html') });
        });
    }

    // Setup flag button
    if (flagButton) {
        flagButton.addEventListener('click', function () {
            chrome.tabs.create({ url: "https://darksurfer.streamlit.app/" });
        });
    }

    // Setup link
    if (linkElement) {
        linkElement.addEventListener('click', function (e) {
            e.preventDefault();
            chrome.tabs.create({ url: linkElement.href });
        });
    }

    // Send popup open message
    chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
        if (tabs && tabs[0]) {
            const tab = tabs[0];
            
            // Check if it's a valid URL for content scripts
            if (tab.url.startsWith("chrome://") || tab.url.startsWith("edge://") || 
                tab.url.startsWith("about:") || tab.url.startsWith("chrome-extension://") ||
                tab.url.startsWith("moz-extension://")) {
                console.log("ℹ️ Ethical Eye: Cannot run on system pages");
                const currentPatternInfo = document.getElementById('current-pattern-info');
                if (currentPatternInfo) {
                    currentPatternInfo.textContent = "Cannot analyze system pages";
                    currentPatternInfo.style.color = "#999";
                }
                return;
            }

            // Try to send message with better error handling
            try {
                chrome.tabs.sendMessage(tab.id, { message: "popup_open" }, function (response) {
                    if (chrome.runtime.lastError) {
                        const errorMsg = chrome.runtime.lastError.message;
                        console.log("ℹ️ Ethical Eye: Content script not ready:", errorMsg);
                        
                        // This is normal if content script hasn't loaded yet
                        // Don't show error for "Could not establish connection" - it's expected
                        if (!errorMsg.includes("Could not establish connection") && 
                            !errorMsg.includes("Receiving end does not exist")) {
                            console.error("❌ Ethical Eye: Unexpected error:", errorMsg);
                        }
                        
                        // Try to inject content script if it's not loaded
                        chrome.scripting.executeScript({
                            target: { tabId: tab.id },
                            files: ['js/content_fixed.js']
                        }).then(() => {
                            console.log("✅ Ethical Eye: Content script injected");
                            // Try sending message again after a short delay
                            setTimeout(() => {
                                chrome.tabs.sendMessage(tab.id, { message: "popup_open" }, function (response) {
                                    if (chrome.runtime.lastError) {
                                        console.log("ℹ️ Ethical Eye: Still waiting for content script");
                                    }
                                });
                            }, 100);
                        }).catch(err => {
                            console.log("ℹ️ Ethical Eye: Could not inject script:", err.message);
                        });
                    } else {
                        console.log("✅ Ethical Eye: Popup open message sent successfully");
                    }
                });
            } catch (error) {
                console.log("ℹ️ Ethical Eye: Error handling:", error.message);
            }
        }
    });
});

// Message listener for updates from content script
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    console.log("🔍 Ethical Eye: Popup received message:", request.message);

    if (request.message === "update_current_count") {
        const patternCount = document.getElementById('pattern-count');
        if (patternCount) {
            patternCount.textContent = request.count;
        }
    } else if (request.message === "patterns_detected") {
        console.log("🔍 Ethical Eye: Patterns detected:", request.patterns);
        detectedPatterns = request.patterns;
        currentPatternIndex = -1;
        updatePatternList();
        updateUI();
    } else if (request.message === "pattern_navigated") {
        currentPatternIndex = request.currentIndex;
        updateUI();
    }
});

function updatePatternList() {
    const patternList = document.getElementById('pattern-list');
    const patternsContainer = document.getElementById('patterns-container');
    const viewResultsButton = document.getElementById('view-results-button');

    if (!patternList || !patternsContainer) return;

    if (detectedPatterns.length === 0) {
        patternsContainer.style.display = 'none';
        if (viewResultsButton) viewResultsButton.style.display = 'none';
        return;
    }

    patternsContainer.style.display = 'block';
    if (viewResultsButton) viewResultsButton.style.display = 'block';
    patternList.innerHTML = '';

    detectedPatterns.forEach((pattern, index) => {
        const patternItem = document.createElement('div');
        patternItem.className = 'pattern-item';
        if (index === currentPatternIndex) {
            patternItem.classList.add('selected');
        }

        patternItem.innerHTML = `
      <div class="pattern-category">${pattern.category}</div>
      <div class="pattern-text">${pattern.text}</div>
      <div class="pattern-confidence">Confidence: ${(pattern.confidence * 100).toFixed(1)}%</div>
    `;

        patternItem.addEventListener('click', function () {
            console.log("🔍 Ethical Eye: Pattern clicked:", index);
            chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
                if (tabs && tabs[0]) {
                    chrome.tabs.sendMessage(tabs[0].id, {
                        message: "highlight_pattern",
                        index: index
                    }, function (response) {
                        if (chrome.runtime.lastError) {
                            console.error("❌ Ethical Eye: Error sending highlight message:", chrome.runtime.lastError);
                        }
                    });
                }
            });
        });

        patternList.appendChild(patternItem);
    });
}

function updateUI() {
    const prevButton = document.getElementById('prev-button');
    const nextButton = document.getElementById('next-button');
    const currentPatternInfo = document.getElementById('current-pattern-info');

    if (detectedPatterns.length === 0) {
        if (prevButton) prevButton.disabled = true;
        if (nextButton) nextButton.disabled = true;
        if (currentPatternInfo) currentPatternInfo.textContent = 'No patterns detected';
        return;
    }

    if (prevButton) prevButton.disabled = false;
    if (nextButton) nextButton.disabled = false;

    if (currentPatternIndex >= 0 && currentPatternIndex < detectedPatterns.length) {
        const pattern = detectedPatterns[currentPatternIndex];
        if (currentPatternInfo) {
            currentPatternInfo.textContent = `Pattern ${currentPatternIndex + 1}/${detectedPatterns.length}: ${pattern.category}`;
        }
    } else {
        if (currentPatternInfo) {
            currentPatternInfo.textContent = 'No pattern selected';
        }
    }

    // Update pattern list selection
    updatePatternList();
}

console.log("🔍 Ethical Eye: Navigation popup script loaded!");
