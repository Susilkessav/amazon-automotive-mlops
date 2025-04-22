function formatCodeBlocks(text) {
    // Regular expression to match code blocks (text between triple backticks)
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    
    return text.replace(codeBlockRegex, (match, language, code) => {
        return `<div class="code-block-wrapper">
                    <div class="code-block">
                        <div class="code-header">
                            ${language ? `<span class="code-language">${language}</span>` : ''}
                        </div>
                        <pre><code>${escapeHtml(code.trim())}</code></pre>
                    </div>
                    <button class="copy-button" onclick="copyCode(this)">Copy</button>
                </div>`;
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function copyCode(button) {
    const codeBlock = button.previousElementSibling.querySelector('code');
    const text = codeBlock.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        const originalText = button.textContent;
        button.textContent = 'Copied!';
        setTimeout(() => {
            button.textContent = originalText;
        }, 2000);
    });
}

// Auto-resize textarea as user types
const textarea = document.getElementById('user-input');
textarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    // Limit max height
    if (this.scrollHeight > 150) {
        this.style.overflowY = 'auto';
        this.style.height = '150px';
    } else {
        this.style.overflowY = 'hidden';
    }
});

// Function to format links in text
function formatLinks(text) {
    // URL regex pattern
    const urlPattern = /(https?:\/\/[^\s]+)/g;
    return text.replace(urlPattern, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
}

// Function to append messages to the chat
function appendMessage(message, isUser = false, sources = null, embeddingModel = null) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    headerDiv.innerHTML = `<span>${isUser ? 'You' : 'Assistant'}</span>`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Format links in the message
    if (!isUser) {
        message = formatLinks(message);
        contentDiv.innerHTML = message;
        
        // Add embedding model info if available
        if (embeddingModel) {
            const embeddingInfo = document.createElement('div');
            embeddingInfo.className = 'embedding-info';
            embeddingInfo.innerHTML = `<span class="embedding-model">Embedding model: ${embeddingModel}</span>`;
            contentDiv.appendChild(embeddingInfo);
        }
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const sourcesToggle = document.createElement('button');
            sourcesToggle.className = 'sources-toggle';
            sourcesToggle.textContent = 'Show sources';
            contentDiv.appendChild(sourcesToggle);
            
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources-container';
            sourcesDiv.style.display = 'none';
            
            sources.forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                sourceItem.innerHTML = `
                    <div class="source-header">
                        <span class="source-title">${index + 1}</span>
                        <span class="source-similarity">Match</span>
                    </div>
                    <div class="source-details">
                        <p class="source-snippet">${source.snippet}</p>
                    </div>
                `;
                sourcesDiv.appendChild(sourceItem);
            });
            
            contentDiv.appendChild(sourcesDiv);
            
            // Toggle sources visibility
            sourcesToggle.addEventListener('click', function() {
                const isHidden = sourcesDiv.style.display === 'none';
                sourcesDiv.style.display = isHidden ? 'block' : 'none';
                this.textContent = isHidden ? 'Hide sources' : 'Show sources';
            });
        }
    } else {
        contentDiv.textContent = message;
    }
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to the bottom with smooth animation
    messagesContainer.scrollTo({
        top: messagesContainer.scrollHeight,
        behavior: 'smooth'
    });
}

// Function to show the loader
function showLoader(message = 'Processing...') {
    const loader = document.getElementById('loader');
    const status = loader.querySelector('.processing-status');
    
    status.textContent = message;
    loader.style.display = 'flex';
    
    // Add random thinking messages for longer waits
    const thinkingMessages = [
        'Searching through product information...',
        'Finding the most relevant details...',
        'Analyzing automotive products...',
        'Comparing product specifications...',
        'Retrieving product recommendations...'
    ];
    
    // Change message every 3 seconds
    let messageIndex = 0;
    window.loaderInterval = setInterval(() => {
        messageIndex = (messageIndex + 1) % thinkingMessages.length;
        status.textContent = thinkingMessages[messageIndex];
    }, 3000);
}

// Function to hide the loader
function hideLoader() {
    const loader = document.getElementById('loader');
    loader.style.display = 'none';
    
    // Clear the interval if it exists
    if (window.loaderInterval) {
        clearInterval(window.loaderInterval);
    }
}

function updateProcessingStatus(message) {
    document.getElementById('processing-details').textContent = message;
}

// Function to send a message to the server
async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    if (message) {
        appendMessage(message, true);
        input.value = '';
        input.style.height = 'auto'; // Reset height
        
        // Focus back on textarea
        input.focus();
        
        showLoader('Finding the best information for you...');
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message }),
            });
            
            const data = await response.json();
            
            if (data.error) {
                appendMessage(`I'm sorry, I encountered an error: ${data.error}`);
            } else {
                appendMessage(data.response, false, data.sources, data.embedding_model);
            }
        } catch (error) {
            appendMessage('I apologize, but I was unable to process your request. Please try again later.');
        } finally {
            hideLoader();
        }
    }
}

async function uploadFiles() {
    const fileInput = document.getElementById('pdf-upload');
    const files = fileInput.files;
    
    if (files.length === 0) return;
    
    showLoader('Starting file upload...');
    
    for (let file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            updateProcessingStatus(`Processing ${file.name}...`);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });
            
            const data = await response.json();
            
            if (data.error) {
                appendMessage(`Error processing ${file.name}: ${data.error}`);
            } else {
                appendMessage(`File ${file.name}: ${data.message}`);
            }
            
        } catch (error) {
            appendMessage(`Error uploading ${file.name}: ${error.message}`);
        }
    }
    
    hideLoader();
    fileInput.value = '';
}

// Add event listener for Enter key
document.getElementById('user-input').addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

// Focus on input when page loads
window.onload = function() {
    document.getElementById('user-input').focus();
};

async function processDataset() {
    const datasetPath = document.getElementById('dataset-path').value.trim();
    const textColumn = document.getElementById('text-column').value.trim();
    const idColumn = document.getElementById('id-column').value.trim();
    
    if (!datasetPath || !textColumn) {
        appendMessage('Error: Dataset path and text column are required', false);
        return;
    }
    
    showLoader('Processing dataset...');
    
    try {
        const response = await fetch('/process_dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                dataset_path: datasetPath,
                text_column: textColumn,
                id_column: idColumn || null
            }),
        });
        
        const data = await response.json();
        
        if (data.error) {
            appendMessage(`Error processing dataset: ${data.error}`, false);
        } else {
            appendMessage(`Dataset processed: ${data.message}`, false);
        }
    } catch (error) {
        appendMessage(`Error: ${error.message}`, false);
    } finally {
        hideLoader();
    }
}
