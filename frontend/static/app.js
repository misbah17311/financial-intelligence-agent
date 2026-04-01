// Financial Intelligence Agent — frontend logic

const API_BASE = '';  // same origin
let isLoading = false;

// ----- initialization -----

document.addEventListener('DOMContentLoaded', async () => {
    await loadExamples();
    await checkHealth();
    document.getElementById('query-input').focus();
});


async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        document.getElementById('llm-info').textContent = `LLM: ${data.llm}`;
    } catch {
        document.getElementById('llm-info').textContent = 'LLM: connecting...';
    }
}


async function loadExamples() {
    try {
        const res = await fetch(`${API_BASE}/api/examples`);
        const data = await res.json();

        const exContainer = document.getElementById('example-queries');
        const allExamples = [...data.structured, ...data.unstructured];
        allExamples.forEach(q => {
            const btn = document.createElement('button');
            btn.className = 'example-btn';
            btn.textContent = q;
            btn.onclick = () => sendWithText(q);
            exContainer.appendChild(btn);
        });

        const grContainer = document.getElementById('guardrail-queries');
        data.guardrail_tests.forEach(q => {
            const btn = document.createElement('button');
            btn.className = 'example-btn guardrail-btn';
            btn.textContent = q;
            btn.onclick = () => sendWithText(q);
            grContainer.appendChild(btn);
        });
    } catch (e) {
        console.error('Failed to load examples:', e);
    }
}


// ----- sending queries -----

function sendWithText(text) {
    document.getElementById('query-input').value = text;
    sendQuery();
}

function sendFromCard(el) {
    const text = el.querySelector('.card-text').textContent;
    sendWithText(text);
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
}

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}


async function sendQuery() {
    const input = document.getElementById('query-input');
    const question = input.value.trim();
    if (!question || isLoading) return;

    isLoading = true;
    document.getElementById('send-btn').disabled = true;

    // hide welcome, show messages
    const welcome = document.getElementById('welcome-screen');
    if (welcome) welcome.style.display = 'none';

    // add user message
    addMessage('user', question);
    input.value = '';
    input.style.height = 'auto';

    // show thinking indicator
    const thinkingId = showThinking();

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        const data = await res.json();
        removeThinking(thinkingId);

        if (data.blocked) {
            addBlockedMessage(data);
        } else {
            addAssistantMessage(data);
        }
    } catch (err) {
        removeThinking(thinkingId);
        addMessage('assistant', `Connection error: ${err.message}. Is the server running?`);
    } finally {
        isLoading = false;
        document.getElementById('send-btn').disabled = false;
        input.focus();
    }
}


// ----- message rendering -----

function addMessage(role, content) {
    const container = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'message';

    const avatar = role === 'user' ? '👤' : '🤖';
    const label = role === 'user' ? 'You' : 'FinAgent';

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar ${role}">${avatar}</div>
            <span class="message-label">${label}</span>
        </div>
        <div class="message-body">${role === 'user' ? escapeHtml(content) : content}</div>
    `;

    container.appendChild(div);
    scrollToBottom();
}


function addAssistantMessage(data) {
    const container = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'message';

    const confClass = (data.confidence || 'medium').toLowerCase();
    const formattedAnswer = formatMarkdown(data.answer);

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar assistant">🤖</div>
            <span class="message-label">FinAgent</span>
            <span class="confidence-badge ${confClass}">${data.confidence || 'N/A'}</span>
            <span class="message-meta">${data.latency_seconds}s</span>
        </div>
        <div class="message-body">${formattedAnswer}</div>
        ${renderGuardrails(data.guardrails)}
    `;

    container.appendChild(div);
    scrollToBottom();
}


function addBlockedMessage(data) {
    const container = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'message';

    const guardrailName = (data.blocked_by || 'unknown').replace(/_/g, ' ');

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar assistant">🤖</div>
            <span class="message-label">FinAgent</span>
            <span class="message-meta">${data.latency_seconds}s</span>
        </div>
        <div class="message-body">
            <div class="blocked-message">
                <div class="blocked-header">🛡️ Query Blocked</div>
                <div class="blocked-reason">${escapeHtml(data.block_message)}</div>
                <span class="blocked-guardrail-tag">${guardrailName}</span>
            </div>
        </div>
        ${renderGuardrails(data.guardrails)}
    `;

    container.appendChild(div);
    scrollToBottom();
}


function renderGuardrails(guardrails) {
    if (!guardrails || guardrails.length === 0) return '';

    const pills = guardrails.map(g => {
        const cls = g.passed ? 'pass' : 'fail';
        const icon = g.passed ? '✓' : '✗';
        const name = g.name.replace(/_/g, ' ');
        return `<span class="guardrail-pill ${cls}"><span class="guardrail-dot"></span>${name} ${icon}</span>`;
    }).join('');

    return `<div class="guardrails-bar">${pills}</div>`;
}


function showThinking() {
    const container = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'message';
    div.id = 'thinking-' + Date.now();

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar assistant">🤖</div>
            <span class="message-label">FinAgent</span>
        </div>
        <div class="thinking">
            <div class="thinking-dots"><span></span><span></span><span></span></div>
            <span>Analyzing your question...</span>
        </div>
    `;

    container.appendChild(div);
    scrollToBottom();
    return div.id;
}


function removeThinking(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}


// ----- utilities -----

function scrollToBottom() {
    const container = document.getElementById('chat-container');
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}


function clearChat() {
    document.getElementById('messages').innerHTML = '';
    const welcome = document.getElementById('welcome-screen');
    if (welcome) welcome.style.display = 'flex';
}


function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


function formatMarkdown(text) {
    if (!text) return '';

    // escape HTML first
    let html = escapeHtml(text);

    // headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');

    // bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // bullet lists — convert lines starting with - to <li>
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    // wrap consecutive <li> in <ul>
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // numbered lists
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

    // paragraphs (double newline)
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';

    // single newlines inside paragraphs
    html = html.replace(/\n/g, '<br>');

    // clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');

    return html;
}
