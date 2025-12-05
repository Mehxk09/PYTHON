// Enhanced gradient interaction
document.addEventListener('mousemove', (e) => {
    const gradientBg = document.querySelector('.gradient-bg');
    const x = e.clientX / window.innerWidth;
    const y = e.clientY / window.innerHeight;
    
    gradientBg.style.transform = `translate(${x * 20 - 10}px, ${y * 20 - 10}px)`;
});

// Add floating letters dynamically scattered all over screen
function createFloatingLetters() {
    const container = document.getElementById('floating-letters');
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    
    for (let i = 0; i < 20; i++) {
        const letter = document.createElement('div');
        letter.className = 'floating-letter';
        letter.textContent = letters[Math.floor(Math.random() * letters.length)];
        
        // Random horizontal position (0% to 100% of viewport width)
        letter.style.left = Math.random() * 100 + 'vw';
        
        // Random vertical position scattered all over screen (0% to 100% of viewport height)
        letter.style.top = Math.random() * 100 + 'vh';
        
        // Random size
        letter.style.fontSize = (Math.random() * 40 + 20) + 'px';
        
        // Very short delay
        letter.style.animationDelay = Math.random() * 3 + 's';
        
        container.appendChild(letter);
    }
}

// Custom cursor functionality - NON-BLOCKING
function initCustomCursor() {
    // Create cursor elements
    const cursorTrail = document.createElement('div');
    cursorTrail.className = 'cursor-trail';
    document.body.appendChild(cursorTrail);

    const cursorDot = document.createElement('div');
    cursorDot.className = 'cursor-dot';
    document.body.appendChild(cursorDot);

    const cursorRing = document.createElement('div');
    cursorRing.className = 'cursor-ring';
    document.body.appendChild(cursorRing);

    // Move cursor with mouse
    let lastX = 0, lastY = 0;
    document.addEventListener('mousemove', (e) => {
        const now = Date.now();
        
        cursorTrail.style.left = (e.clientX - 10) + 'px';
        cursorTrail.style.top = (e.clientY - 10) + 'px';
        
        cursorDot.style.left = (e.clientX - 3) + 'px';
        cursorDot.style.top = (e.clientY - 3) + 'px';
        
        cursorRing.style.left = (e.clientX - 20) + 'px';
        cursorRing.style.top = (e.clientY - 20) + 'px';
        
        lastX = e.clientX;
        lastY = e.clientY;
    });

    // Add click effect
    document.addEventListener('mousedown', () => {
        cursorTrail.style.transform = 'scale(0.8)';
        cursorRing.style.transform = 'scale(0.9)';
    });

    document.addEventListener('mouseup', () => {
        cursorTrail.style.transform = 'scale(1)';
        cursorRing.style.transform = 'scale(1)';
    });
}

// Initialize when page loads - NON-BLOCKING
document.addEventListener('DOMContentLoaded', function() {
    // Initialize visual effects only
    createFloatingLetters();
    
    // Initialize cursor with a small delay to not block other scripts
    setTimeout(initCustomCursor, 100);
});

// Fetch and update predictions from backend
function updatePredictions() {
    fetch('/get_prediction')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update current letter
                const currentLetterEl = document.getElementById('current-letter');
                if (currentLetterEl) {
                    currentLetterEl.textContent = data.current_letter || '—';
                }
                
                // Update current word
                const currentWordEl = document.getElementById('current-word');
                if (currentWordEl) {
                    currentWordEl.textContent = data.current_word || '—';
                }
                
                // Update predicted word (accumulated text)
                const predictedWordEl = document.getElementById('predicted-word');
                if (predictedWordEl) {
                    predictedWordEl.textContent = data.predicted_word || '';
                }
                
                // Update mode status
                updateModeDisplay(data.mode, data.has_letter_model, data.has_word_model);
            }
        })
        .catch(error => {
            console.error('Error fetching predictions:', error);
        });
}

function updateModeDisplay(mode, hasLetterModel, hasWordModel) {
    const modeStatus = document.getElementById('mode-status');
    const letterCard = document.getElementById('letter-card');
    const wordCard = document.getElementById('word-card');
    const letterModeBtn = document.getElementById('letter-mode-btn');
    const wordModeBtn = document.getElementById('word-mode-btn');
    
    if (modeStatus) {
        let statusText = `Modo: ${mode === 'words' ? 'Palavras' : 'Letras'}`;
        if (mode === 'words' && !hasWordModel) {
            statusText += ' (Modelo não disponível - Treine primeiro!)';
        } else if (mode === 'letters' && !hasLetterModel) {
            statusText += ' (Modelo não disponível)';
        }
        modeStatus.textContent = statusText;
    }
    
    // Show/hide cards based on mode
    if (letterCard) {
        letterCard.style.display = mode === 'letters' ? 'block' : 'none';
    }
    if (wordCard) {
        wordCard.style.display = mode === 'words' ? 'block' : 'none';
    }
    
    // Update button states
    if (letterModeBtn) {
        letterModeBtn.classList.toggle('active', mode === 'letters');
        letterModeBtn.disabled = !hasLetterModel;
        if (!hasLetterModel) {
            letterModeBtn.title = 'Modelo de letras não disponível';
        }
    }
    if (wordModeBtn) {
        wordModeBtn.classList.toggle('active', mode === 'words');
        wordModeBtn.disabled = !hasWordModel;
        if (!hasWordModel) {
            wordModeBtn.title = 'Modelo de palavras não disponível. Execute: python PAP/model/train_word_model.py';
        } else {
            wordModeBtn.title = '';
        }
    }
}

function setMode(mode) {
    fetch('/set_mode', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ mode: mode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log(`Mode switched to: ${mode}`);
            updatePredictions();
        } else {
            alert(data.error || 'Failed to switch mode');
        }
    })
    .catch(error => {
        console.error('Error switching mode:', error);
    });
}

// Poll for predictions every 100ms (10 times per second)
setInterval(updatePredictions, 100);

let cameraActive = false;

function setCameraPlaceholder(visible) {
    const placeholder = document.getElementById('camera-placeholder');
    if (!placeholder) return;
    placeholder.classList.toggle('hidden', !visible);
}

function updateCameraButton() { 
    const toggleBtn = document.getElementById('toggle-camera-btn');
    if (toggleBtn) {
        toggleBtn.textContent = cameraActive ? 'Stop Camera' : 'Start Camera';
    }
}

function startCamera() {
    if (cameraActive) return;
    const videoStream = document.getElementById('video-stream');
    if (!videoStream) return;
    const baseUrl = videoStream.dataset.streamUrl || '/video_feed';
    videoStream.src = `${baseUrl}?t=${Date.now()}`;
    cameraActive = true;
    updateCameraButton();
    setCameraPlaceholder(false);
}

function stopCamera() {
    if (!cameraActive) return;
    const videoStream = document.getElementById('video-stream');
    if (!videoStream) return;
    videoStream.src = '';
    cameraActive = false;
    updateCameraButton();
    setCameraPlaceholder(true);
}

// Button handlers
document.addEventListener('DOMContentLoaded', function() {
    updateCameraButton();
    setCameraPlaceholder(true);

    const toggleCameraBtn = document.getElementById('toggle-camera-btn');
    if (toggleCameraBtn) {
        toggleCameraBtn.addEventListener('click', function() {
            if (cameraActive) {
                stopCamera();
            } else {
                startCamera();
            }
        });
    }

    // Reset button
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            fetch('/reset_word', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updatePredictions(); // Update display immediately
                }
            })
            .catch(error => {
                console.error('Error resetting word:', error);
            });
        });
    }
    
    // Space button
    const spaceBtn = document.getElementById('space-btn');
    if (spaceBtn) {
        spaceBtn.addEventListener('click', function() {
            fetch('/add_space', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updatePredictions(); // Update display immediately
                }
            })
            .catch(error => {
                console.error('Error adding space:', error);
            });
        });
    }

    // Backspace button
    const backspaceBtn = document.getElementById('backspace-btn');
    if (backspaceBtn) {
        backspaceBtn.addEventListener('click', function() {
            fetch('/backspace', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updatePredictions(); // Update display immediately
                }
            })
            .catch(error => {
                console.error('Error doing backspace:', error);
            });
        });
    }
    
    // Mode toggle buttons
    const letterModeBtn = document.getElementById('letter-mode-btn');
    const wordModeBtn = document.getElementById('word-mode-btn');
    
    if (letterModeBtn) {
        letterModeBtn.addEventListener('click', function() {
            setMode('letters');
        });
    }
    
    if (wordModeBtn) {
        wordModeBtn.addEventListener('click', function() {
            setMode('words');
        });
    }
    
    // Initial update
    updatePredictions();
    
    // Check available modes on load
    fetch('/get_mode')
        .then(response => response.json())
        .then(data => {
            updateModeDisplay(data.mode, data.has_letter_model, data.has_word_model);
        })
        .catch(error => {
            console.error('Error fetching mode:', error);
        });
});