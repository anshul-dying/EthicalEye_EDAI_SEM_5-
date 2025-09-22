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
                    console.log("🔍 Ethical Eye: Sending analyze message to tab:", tabs[0].id);

                    // Send message with better error handling
                    chrome.tabs.sendMessage(tabs[0].id, { message: "analyze_site" }, function (response) {
                        if (chrome.runtime.lastError) {
                            console.error("❌ Ethical Eye: Error sending message:", chrome.runtime.lastError);
                            // Don't show alert for port closed errors - this is often normal
                            if (!chrome.runtime.lastError.message.includes("port closed")) {
                                alert("❌ Error: " + chrome.runtime.lastError.message + "\n\nTry refreshing the page and reloading the extension.");
                            }
                        } else {
                            console.log("✅ Ethical Eye: Analysis started, response:", response);
                        }
                    });

                    // Also try direct script execution as backup
                    chrome.scripting.executeScript({
                        target: { tabId: tabs[0].id },
                        function: function () {
                            console.log("🔍 Ethical Eye: Direct execution - calling scrape...");
                            if (typeof scrape === 'function') {
                                scrape();
                            } else {
                                console.error("❌ Ethical Eye: scrape function not found");
                            }
                        }
                    }).catch(error => {
                        console.log("ℹ️ Ethical Eye: Direct execution not available:", error.message);
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
                            console.error("❌ Ethical Eye: Error sending previous message:", chrome.runtime.lastError);
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
                            console.error("❌ Ethical Eye: Error sending next message:", chrome.runtime.lastError);
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
                            console.error("❌ Ethical Eye: Error sending clear message:", chrome.runtime.lastError);
                        }
                    });
                }
            });
            currentPatternIndex = -1;
            updateUI();
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
            chrome.tabs.sendMessage(tabs[0].id, { message: "popup_open" }, function (response) {
                if (chrome.runtime.lastError) {
                    console.error("❌ Ethical Eye: Error sending popup open message:", chrome.runtime.lastError);
                }
            });
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

    if (!patternList || !patternsContainer) return;

    if (detectedPatterns.length === 0) {
        patternsContainer.style.display = 'none';
        return;
    }

    patternsContainer.style.display = 'block';
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
