// Enhanced gradient interaction (throttled)
let gradientThrottle = false;
document.addEventListener('mousemove', (e) => {
    if (gradientThrottle) return;
    gradientThrottle = true;
    requestAnimationFrame(() => {
        const gradientBg = document.querySelector('.gradient-bg');
        if (gradientBg) {
            const x = e.clientX / window.innerWidth;
            const y = e.clientY / window.innerHeight;
            gradientBg.style.transform = `translate(${x * 20 - 10}px, ${y * 20 - 10}px)`;
        }
        gradientThrottle = false;
    });
});

// Add floating letters dynamically (reduced count for performance)
function createFloatingLetters() {
    const container = document.getElementById('floating-letters');
    if (!container) return;
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

    for (let i = 0; i < 12; i++) {
        const letter = document.createElement('div');
        letter.className = 'floating-letter';
        letter.textContent = letters[Math.floor(Math.random() * letters.length)];
        letter.style.left = Math.random() * 100 + 'vw';
        letter.style.top = Math.random() * 100 + 'vh';
        letter.style.fontSize = (Math.random() * 35 + 18) + 'px';
        letter.style.animationDelay = Math.random() * 3 + 's';
        container.appendChild(letter);
    }
}

// Custom cursor functionality (lightweight)
function initCustomCursor() {
    const cursorTrail = document.createElement('div');
    cursorTrail.className = 'cursor-trail';
    document.body.appendChild(cursorTrail);

    const cursorDot = document.createElement('div');
    cursorDot.className = 'cursor-dot';
    document.body.appendChild(cursorDot);

    const cursorRing = document.createElement('div');
    cursorRing.className = 'cursor-ring';
    document.body.appendChild(cursorRing);

    let cursorRAF = false;
    let mx = 0, my = 0;

    document.addEventListener('mousemove', (e) => {
        mx = e.clientX;
        my = e.clientY;
        if (!cursorRAF) {
            cursorRAF = true;
            requestAnimationFrame(() => {
                cursorTrail.style.left = (mx - 10) + 'px';
                cursorTrail.style.top = (my - 10) + 'px';
                cursorDot.style.left = (mx - 3) + 'px';
                cursorDot.style.top = (my - 3) + 'px';
                cursorRing.style.left = (mx - 20) + 'px';
                cursorRing.style.top = (my - 20) + 'px';
                cursorRAF = false;
            });
        }
    });

    document.addEventListener('mousedown', () => {
        cursorTrail.style.transform = 'scale(0.8)';
        cursorRing.style.transform = 'scale(0.9)';
    });

    document.addEventListener('mouseup', () => {
        cursorTrail.style.transform = 'scale(1)';
        cursorRing.style.transform = 'scale(1)';
    });

    // Only hide native cursor after custom one is ready
    document.body.classList.add('custom-cursor-active');
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function () {
    createFloatingLetters();
    setTimeout(initCustomCursor, 150);
});

// -------- Prediction polling --------
let predictionInterval = null;

function updatePredictions() {
    fetch('/get_prediction')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const currentLetterEl = document.getElementById('current-letter');
                if (currentLetterEl) {
                    currentLetterEl.textContent = data.current_letter || '\u2014';
                }

                const currentWordEl = document.getElementById('current-word');
                if (currentWordEl) {
                    currentWordEl.textContent = data.current_word || '\u2014';
                }

                const predictedWordEl = document.getElementById('predicted-word');
                if (predictedWordEl) {
                    predictedWordEl.textContent = data.predicted_word || '';
                }

                // Update confidence bar
                updateConfidenceBar(data.confidence || 0);

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
            statusText += ' (modelo não disponível)';
        } else if (mode === 'letters' && !hasLetterModel) {
            statusText += ' (modelo não disponível)';
        }
        modeStatus.textContent = statusText;
    }

    if (letterCard) {
        letterCard.style.display = mode === 'letters' ? 'block' : 'none';
    }
    if (wordCard) {
        wordCard.style.display = mode === 'words' ? 'block' : 'none';
    }

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
            wordModeBtn.title = 'Modelo de palavras não disponível. Treine primeiro.';
        } else {
            wordModeBtn.title = 'Mudar para modo palavras';
        }
    }
}

function setMode(mode) {
    fetch('/set_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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

// Start polling only when camera is active (saves resources)
function startPolling() {
    if (predictionInterval) return;
    predictionInterval = setInterval(updatePredictions, 350);
}

function stopPolling() {
    if (predictionInterval) {
        clearInterval(predictionInterval);
        predictionInterval = null;
    }
}

// -------- Camera controls --------
let cameraActive = false;

function setCameraPlaceholder(visible) {
    const placeholder = document.getElementById('camera-placeholder');
    if (!placeholder) return;
    placeholder.classList.toggle('hidden', !visible);
}

function updateCameraButton() {
    const toggleBtn = document.getElementById('toggle-camera-btn');
    if (toggleBtn) {
        toggleBtn.textContent = cameraActive ? 'Parar Câmara' : 'Iniciar Câmara';
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
    startPolling();
    // Add glowing border
    const frame = document.querySelector('.camera-frame');
    if (frame) frame.classList.add('camera-active');
}

function stopCamera() {
    if (!cameraActive) return;
    const videoStream = document.getElementById('video-stream');
    if (!videoStream) return;
    videoStream.src = '';
    cameraActive = false;
    updateCameraButton();
    setCameraPlaceholder(true);
    stopPolling();
    // Remove glowing border
    const frame = document.querySelector('.camera-frame');
    if (frame) frame.classList.remove('camera-active');
}

// -------- Button handlers --------
document.addEventListener('DOMContentLoaded', function () {
    updateCameraButton();
    setCameraPlaceholder(true);

    const toggleCameraBtn = document.getElementById('toggle-camera-btn');
    if (toggleCameraBtn) {
        toggleCameraBtn.addEventListener('click', function () {
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
        resetBtn.addEventListener('click', function () {
            fetch('/reset_word', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
                .then(r => r.json())
                .then(data => { if (data.success) updatePredictions(); })
                .catch(e => console.error('Error resetting word:', e));
        });
    }

    // Space button
    const spaceBtn = document.getElementById('space-btn');
    if (spaceBtn) {
        spaceBtn.addEventListener('click', function () {
            fetch('/add_space', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
                .then(r => r.json())
                .then(data => { if (data.success) updatePredictions(); })
                .catch(e => console.error('Error adding space:', e));
        });
    }

    // Backspace button
    const backspaceBtn = document.getElementById('backspace-btn');
    if (backspaceBtn) {
        backspaceBtn.addEventListener('click', function () {
            fetch('/backspace', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
                .then(r => r.json())
                .then(data => { if (data.success) updatePredictions(); })
                .catch(e => console.error('Error doing backspace:', e));
        });
    }

    // Mode toggle buttons
    const letterModeBtn = document.getElementById('letter-mode-btn');
    const wordModeBtn = document.getElementById('word-mode-btn');

    if (letterModeBtn) {
        letterModeBtn.addEventListener('click', function () {
            setMode('letters');
        });
    }

    if (wordModeBtn) {
        wordModeBtn.addEventListener('click', function () {
            setMode('words');
        });
    }

    // Initial mode check
    fetch('/get_mode')
        .then(r => r.json())
        .then(data => {
            updateModeDisplay(data.mode, data.has_letter_model, data.has_word_model);
        })
        .catch(e => console.error('Error fetching mode:', e));

    // One initial prediction update
    updatePredictions();

    // 4. Button ripple effect on all buttons
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('click', function (e) {
            const ripple = document.createElement('span');
            ripple.className = 'btn-ripple';
            const rect = btn.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
            ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
            btn.appendChild(ripple);
            setTimeout(() => ripple.remove(), 600);
        });
    });
});

// 7. Update confidence bar
function updateConfidenceBar(confidence) {
    const bar = document.getElementById('confidence-bar');
    const label = document.getElementById('confidence-label');
    if (!bar || !label) return;

    const pct = Math.round(confidence * 100);
    bar.style.width = pct + '%';
    label.textContent = pct + '%';

    // Change color based on confidence level
    if (pct >= 70) {
        bar.style.background = 'linear-gradient(90deg, #2d8a4e, #153d13)';
    } else if (pct >= 40) {
        bar.style.background = 'linear-gradient(90deg, #8a8a2d, #5a6b1a)';
    } else {
        bar.style.background = 'linear-gradient(90deg, #8a4a2d, #6b2a1a)';
    }
}
