function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Deactivate all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Activate selected tab and button
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.currentTarget.classList.add('active');
}

// File input handling
const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('file-name');
const dropZone = document.getElementById('drop-zone');

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        fileNameDisplay.textContent = e.target.files[0].name;
    }
});

// Drag and drop support
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#3b82f6';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '';

    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        fileNameDisplay.textContent = e.dataTransfer.files[0].name;
    }
});

async function analyzeText() {
    const text = document.getElementById('text-input').value;
    if (!text.trim()) {
        alert("Please enter some text first.");
        return;
    }

    if (text.length < 50) {
        if (!confirm("This text is very short (less than 50 chars). The results might be unreliable. Do you want to continue?")) {
            return;
        }
    }

    const formData = new FormData();
    formData.append('text', text);

    await sendRequest('/detect', formData);
}

async function analyzeFile() {
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select a file first.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    await sendRequest('/detect/file', formData);
}

async function sendRequest(endpoint, formData) {
    const resultSection = document.getElementById('result-section');
    const analyzeBtns = document.querySelectorAll('.analyze-btn');

    // Loading state
    analyzeBtns.forEach(btn => {
        btn.textContent = "Analyzing...";
        btn.disabled = true;
    });

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            alert("Error: " + (data.detail || "Something went wrong"));
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Failed to connect to the server.");
    } finally {
        // Reset state
        analyzeBtns.forEach(btn => {
            btn.textContent = endpoint.includes('file') ? "Analyze File" : "Analyze Text";
            btn.disabled = false;
        });
    }
}

function displayResults(data) {
    const resultSection = document.getElementById('result-section');
    const aiScoreElem = document.getElementById('ai-score');
    const verdictElem = document.getElementById('verdict');
    const confidenceElem = document.getElementById('confidence');
    const ring = document.getElementById('ai-score-ring');
    const scoreContainer = document.querySelector('.result-card');

    resultSection.style.display = 'block';

    // Update values
    const aiScore = data.ai_score;
    aiScoreElem.textContent = aiScore.toFixed(1);
    // Verdict and confidence are now set dynamically below based on score thresholds
    // verdictElem.textContent = data.prediction;
    // confidenceElem.textContent = data.confidence + '%';

    // Update visuals
    const offset = 100 - aiScore;
    ring.style.strokeDasharray = `${aiScore}, 100`;

    // Remove old classes
    scoreContainer.classList.remove('human-safe', 'ai-risk', 'mixed-risk', 'too-short');

    if (data.prediction === "Not Enough Text") {
        scoreContainer.classList.add('too-short');
        verdictElem.textContent = "Text Too Short";
        verdictElem.style.color = "var(--secondary-color)";
        confidenceElem.textContent = data.message;
        aiScoreElem.textContent = "0";
        ring.style.strokeDasharray = `0, 100`;
    } else if (aiScore > 60) {
        scoreContainer.classList.add('ai-risk');
        verdictElem.textContent = "AI Generated";
        verdictElem.style.color = "var(--danger-color)";
        confidenceElem.textContent = data.ai_score.toFixed(2) + '% (AI)';
    } else if (aiScore < 40) {
        scoreContainer.classList.add('human-safe');
        verdictElem.textContent = "Human Written";
        verdictElem.style.color = "var(--success-color)";
        confidenceElem.textContent = data.human_score.toFixed(2) + '% (Human)';
    } else {
        scoreContainer.classList.add('mixed-risk');
        verdictElem.textContent = "Uncertain / Mixed";
        verdictElem.style.color = "#f59e0b";
        confidenceElem.textContent = "Low Confidence";
    }

    // Show analysis notes if any
    const detailsContainer = document.querySelector('.details');
    // Remove existing notes if any
    const existingNotes = detailsContainer.querySelector('.analysis-notes');
    if (existingNotes) existingNotes.remove();

    if (data.analysis_notes && data.analysis_notes.length > 0) {
        const notesDiv = document.createElement('div');
        notesDiv.className = 'analysis-notes';
        notesDiv.style.marginTop = '1rem';
        notesDiv.style.textAlign = 'left';
        notesDiv.style.fontSize = '0.8rem';

        const title = document.createElement('div');
        title.textContent = "💡 Analysis Insights:";
        title.style.fontWeight = "600";
        notesDiv.appendChild(title);

        data.analysis_notes.forEach(note => {
            const p = document.createElement('p');
            p.textContent = `• ${note}`;
            p.style.margin = "0.2rem 0";
            p.style.color = "var(--secondary-color)";
            notesDiv.appendChild(p);
        });

        detailsContainer.appendChild(notesDiv);
    }


    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

async function checkModelStatus() {
    const badge = document.getElementById('model-badge');
    const statusText = document.getElementById('model-status');

    try {
        const response = await fetch('/model-info');
        const data = await response.json();

        statusText.textContent = `⚡ Active: ${data.model_type}`;
        badge.classList.add('active');
        badge.title = `Loaded from: ${data.model_name}`;
    } catch (e) {
        statusText.textContent = "⚠️ Model Status Unknown";
        console.error("Could not fetch model info", e);
    }
}

// Check model status on load
document.addEventListener('DOMContentLoaded', checkModelStatus);

