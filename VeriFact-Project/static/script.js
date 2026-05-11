document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('fact-check-form');
    const claimInput = document.getElementById('claim-input');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('span');
    const btnLoader = document.getElementById('btn-loader');
    
    const resultsContainer = document.getElementById('results-container');
    const errorToast = document.getElementById('error-toast');
    
    const icons = {
        "VERIFIED": "✅",
        "FALSE": "❌",
        "PARTIALLY TRUE": "⚠️",
        "INCONCLUSIVE": "🔍"
    };

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const claim = claimInput.value.trim();
        if (!claim) return;
        
        // UI State: Loading
        btnText.textContent = 'Verifying...';
        btnLoader.classList.remove('hidden');
        submitBtn.disabled = true;
        resultsContainer.classList.add('hidden');
        hideError();
        
        try {
            const response = await fetch('/api/check', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ claim })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'An error occurred while verifying the claim.');
            }
            
            renderResults(data);
            
        } catch (err) {
            showError(err.message);
        } finally {
            // UI State: Restored
            btnText.textContent = 'Check';
            btnLoader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });
    
    function renderResults(data) {
        const verdict = data.verdict || 'INCONCLUSIVE';
        
        // Update Banner
        const banner = document.getElementById('verdict-banner');
        banner.className = 'verdict-banner'; // reset classes
        
        if (verdict === 'VERIFIED') banner.classList.add('verdict-verified');
        else if (verdict === 'FALSE') banner.classList.add('verdict-false');
        else if (verdict === 'PARTIALLY TRUE') banner.classList.add('verdict-partial');
        else banner.classList.add('verdict-inconclusive');
        
        document.getElementById('verdict-icon').textContent = icons[verdict] || "🔍";
        document.getElementById('verdict-title').textContent = verdict;
        
        // Update Text Fields
        document.getElementById('verdict-summary').textContent = data.summary || '';
        
        const reasoningCard = document.getElementById('reasoning-card');
        if (data.reasoning) {
            document.getElementById('verdict-reasoning').textContent = data.reasoning;
            reasoningCard.classList.remove('hidden');
        } else {
            reasoningCard.classList.add('hidden');
        }
        
        // Update Key Findings
        const findingsList = document.getElementById('key-findings-list');
        findingsList.innerHTML = '';
        if (data.key_findings && data.key_findings.length > 0) {
            data.key_findings.forEach(finding => {
                const li = document.createElement('li');
                li.textContent = finding;
                findingsList.appendChild(li);
            });
        } else {
            findingsList.innerHTML = '<li style="color: var(--text-secondary)">No key findings available.</li>';
        }
        
        // Update Caveats
        const caveatsCard = document.getElementById('caveats-card');
        if (data.caveats) {
            document.getElementById('verdict-caveats').textContent = data.caveats;
            caveatsCard.classList.remove('hidden');
        } else {
            caveatsCard.classList.add('hidden');
        }
        
        // Update Sources
        const sourcesList = document.getElementById('sources-list');
        sourcesList.innerHTML = '';
        if (data.cited_sources && data.cited_sources.length > 0) {
            data.cited_sources.forEach(source => {
                const li = document.createElement('li');
                
                const titleNode = document.createElement('div');
                const titleLink = document.createElement('a');
                titleLink.href = source.url;
                titleLink.target = '_blank';
                titleLink.rel = 'noopener noreferrer';
                titleLink.textContent = source.title || 'Unknown Source';
                titleNode.appendChild(titleLink);
                
                const urlNode = document.createElement('span');
                urlNode.textContent = source.url;
                
                li.appendChild(titleNode);
                li.appendChild(urlNode);
                sourcesList.appendChild(li);
            });
        } else {
            sourcesList.innerHTML = '<li style="color: var(--text-secondary)">No sources cited.</li>';
        }
        
        // Update Meta
        document.getElementById('meta-time').textContent = new Date(data.checked_at).toLocaleString();
        document.getElementById('meta-elapsed').textContent = data.elapsed_sec || '?';
        document.getElementById('meta-searched').textContent = data.sources_searched || '?';
        
        // Show container
        resultsContainer.classList.remove('hidden');
    }
    
    function showError(msg) {
        errorToast.textContent = msg;
        errorToast.classList.remove('hidden');
        setTimeout(() => {
            errorToast.classList.add('hidden');
        }, 5000);
    }
    
    function hideError() {
        errorToast.classList.add('hidden');
    }
});
