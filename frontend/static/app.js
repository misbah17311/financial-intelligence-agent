/* ===========================================================
   FinAgent — Production UI Logic
   =========================================================== */

const API_BASE = '';
let isLoading = false;

/* ---- init ---- */
document.addEventListener('DOMContentLoaded', async () => {
    await Promise.all([loadExamples(), checkHealth()]);
    document.getElementById('query-input').focus();
});

async function checkHealth() {
    const el = document.getElementById('llm-info');
    const dot = document.querySelector('.status-dot');
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const d = await res.json();
        if (d.ready) {
            el.textContent = `Online · ${d.llm}`;
            dot.style.background = 'var(--success)';
        } else {
            el.textContent = `Warming up · ${d.llm}`;
            dot.style.background = 'var(--warning)';
            dot.style.boxShadow = '0 0 6px rgba(251,191,36,.4)';
            // Poll until ready
            setTimeout(checkHealth, 3000);
        }
    } catch {
        el.textContent = 'Connecting...';
        dot.style.background = 'var(--danger)';
        dot.style.boxShadow = '0 0 6px rgba(248,113,113,.4)';
        setTimeout(checkHealth, 3000);
    }
}

async function loadExamples() {
    try {
        const res = await fetch(`${API_BASE}/api/examples`);
        const data = await res.json();

        const exContainer = document.getElementById('example-queries');
        [...data.structured, ...data.unstructured].forEach(q => {
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

/* ---- sending ---- */
function sendWithText(text) {
    document.getElementById('query-input').value = text;
    sendQuery();
}

function sendFromCard(el) {
    const text = el.dataset.query || el.querySelector('.prompt-card-desc')?.textContent;
    if (text) sendWithText(text);
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
}

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

async function sendQuery() {
    const input = document.getElementById('query-input');
    const question = input.value.trim();
    if (!question || isLoading) return;

    isLoading = true;
    document.getElementById('send-btn').disabled = true;

    const welcome = document.getElementById('welcome-screen');
    if (welcome) welcome.style.display = 'none';

    addMessage('user', question);
    input.value = '';
    input.style.height = 'auto';

    const thinkingId = showThinking();
    const startTime = performance.now();

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
            await addAssistantMessage(data);
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

/* ---- message rendering ---- */
function addMessage(role, content) {
    const container = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = `message message--${role}`;

    const avatarContent = role === 'user'
        ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
        : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>';
    const label = role === 'user' ? 'You' : 'FinAgent';

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar ${role}">${avatarContent}</div>
            <span class="message-label">${label}</span>
        </div>
        <div class="message-body">${role === 'user' ? escapeHtml(content) : formatMarkdown(content)}</div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

async function addAssistantMessage(data) {
    const container = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'message message--assistant';

    const confClass = (data.confidence || 'medium').toLowerCase();
    const latency = data.latency_seconds ? `${data.latency_seconds}s` : '';

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar assistant"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
            <span class="message-label">FinAgent</span>
            <span class="confidence-badge ${confClass}">${data.confidence || 'N/A'}</span>
            <span class="message-meta">${latency}</span>
        </div>
        <div class="message-body" id="typewriter-target-${Date.now()}"></div>
        ${renderSourcePills(data)}
        ${renderSqlBlock(data.sql_queries)}
        ${renderGuardrails(data.guardrails)}
        ${renderPipelineTrace(data)}
    `;
    container.appendChild(div);

    const bodyEl = div.querySelector('.message-body');
    await typewriterRender(bodyEl, data.answer);

    // try to render a chart if the answer contains tabular data
    tryRenderChart(div, data.answer);

    scrollToBottom();
}

function addBlockedMessage(data) {
    const container = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'message message--assistant';

    const guardrailName = (data.blocked_by || 'unknown').replace(/_/g, ' ');
    const latency = data.latency_seconds ? `${data.latency_seconds}s` : '';

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar assistant"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
            <span class="message-label">FinAgent</span>
            <span class="message-meta">${latency}</span>
        </div>
        <div class="message-body">
            <div class="blocked-message">
                <div class="blocked-header">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                    Query Blocked
                </div>
                <div class="blocked-reason">${escapeHtml(data.block_message)}</div>
                <span class="blocked-guardrail-tag">${guardrailName}</span>
            </div>
        </div>
        ${renderGuardrails(data.guardrails)}
    `;
    container.appendChild(div);
    scrollToBottom();
}

/* ---- typewriter effect ---- */
async function typewriterRender(el, text) {
    if (!text) return;
    const html = formatMarkdown(text);
    // For short responses, just set directly
    if (text.length < 100) {
        el.innerHTML = html;
        return;
    }
    // For longer ones, reveal in chunks for a streaming feel
    const words = text.split(' ');
    let current = '';
    const chunkSize = 3;
    for (let i = 0; i < words.length; i += chunkSize) {
        current += (i > 0 ? ' ' : '') + words.slice(i, i + chunkSize).join(' ');
        el.innerHTML = formatMarkdown(current);
        scrollToBottom();
        await sleep(18);
    }
    el.innerHTML = html; // ensure final output is clean
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/* ---- guardrails ---- */
function renderGuardrails(guardrails) {
    if (!guardrails || guardrails.length === 0) return '';

    const pills = guardrails.map(g => {
        const cls = g.passed ? 'pass' : 'fail';
        const icon = g.passed ? '✓' : '✗';
        const name = g.name.replace(/_/g, ' ');
        return `<span class="guardrail-pill ${cls}">${icon} ${name}</span>`;
    }).join('');

    const id = 'gr-' + Date.now();
    return `
        <div class="guardrails-bar">
            <button class="guardrails-toggle" onclick="toggleGuardrails('${id}')">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                ${guardrails.length} checks passed
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
            </button>
            <div class="guardrails-detail" id="${id}">${pills}</div>
        </div>
    `;
}

function toggleGuardrails(id) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('open');
}

/* ---- thinking indicator ---- */
const thinkingStages = ['Planning', 'Retrieving', 'Analyzing', 'Checking'];

function showThinking() {
    const container = document.getElementById('messages');
    const div = document.createElement('div');
    const id = 'thinking-' + Date.now();
    div.className = 'message message--assistant';
    div.id = id;

    div.innerHTML = `
        <div class="message-header">
            <div class="message-avatar assistant"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
            <span class="message-label">FinAgent</span>
        </div>
        <div class="thinking-indicator">
            <div class="thinking-spinner"></div>
            <span class="thinking-text">Analyzing your question...</span>
        </div>
        <div class="thinking-steps" id="steps-${id}"></div>
    `;
    container.appendChild(div);
    scrollToBottom();

    // Animate thinking stages
    let stageIdx = 0;
    const stepsEl = div.querySelector(`#steps-${id}`);
    const textEl = div.querySelector('.thinking-text');
    const stageMessages = [
        'Understanding your query...',
        'Searching knowledge base...',
        'Analyzing relevant data...',
        'Verifying response quality...'
    ];

    const interval = setInterval(() => {
        if (stageIdx < thinkingStages.length) {
            // Add step pill
            const pill = document.createElement('span');
            pill.className = 'thinking-step active';
            pill.textContent = thinkingStages[stageIdx];
            pill.style.animationDelay = `${stageIdx * 0.05}s`;
            stepsEl.appendChild(pill);

            // Update text
            textEl.textContent = stageMessages[stageIdx] || 'Processing...';

            // Mark previous as completed
            const prev = stepsEl.querySelectorAll('.thinking-step');
            if (prev.length > 1) prev[prev.length - 2].classList.remove('active');

            stageIdx++;
            scrollToBottom();
        }
    }, 2500);

    div._interval = interval;
    return id;
}

function removeThinking(id) {
    const el = document.getElementById(id);
    if (el) {
        clearInterval(el._interval);
        el.remove();
    }
}

/* ---- SQL syntax display ---- */
function renderSqlBlock(queries) {
    if (!queries || queries.length === 0) return '';
    const id = 'sql-' + Date.now();
    const blocks = queries.map(q => {
        const highlighted = (typeof hljs !== 'undefined')
            ? hljs.highlight(q, { language: 'sql' }).value
            : escapeHtml(q);
        return `<pre><code class="hljs language-sql">${highlighted}</code></pre>`;
    }).join('');

    return `
        <div class="sql-block" style="margin-left:40px;">
            <div class="sql-block-header" onclick="toggleSqlBlock('${id}')">
                <span class="sql-block-label">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                    SQL Quer${queries.length > 1 ? 'ies' : 'y'} Used
                </span>
                <svg class="sql-block-toggle" id="toggle-${id}" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
            <div class="sql-block-body" id="${id}">${blocks}</div>
        </div>
    `;
}

function toggleSqlBlock(id) {
    const body = document.getElementById(id);
    const toggle = document.getElementById('toggle-' + id);
    if (body) body.classList.toggle('open');
    if (toggle) toggle.classList.toggle('open');
}

/* ---- source pills ---- */
function renderSourcePills(data) {
    const sources = data.sources_used || [];
    if (sources.length === 0) return '';

    const pills = sources.map(s => {
        if (s.includes('SQL') || s.includes('Database')) {
            return `<span class="source-pill source-pill--db"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg> ${escapeHtml(s)}</span>`;
        }
        if (s.includes('News') || s.includes('Article')) {
            return `<span class="source-pill source-pill--news"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg> ${escapeHtml(s)}</span>`;
        }
        return `<span class="source-pill"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg> ${escapeHtml(s)}</span>`;
    }).join('');

    return `<div class="response-meta">${pills}</div>`;
}

/* ---- pipeline trace ---- */
function renderPipelineTrace(data) {
    const plan = data.plan;
    if (!plan) return '';
    const id = 'trace-' + Date.now();

    // parse plan steps
    let steps = [];
    try {
        const match = plan.match(/\[.*\]/s);
        if (match) {
            const parsed = JSON.parse(match[0]);
            steps = parsed.map(s => ({
                tool: s.tool || 'unknown',
                input: typeof s.input === 'string' ? s.input.substring(0, 80) : JSON.stringify(s.input).substring(0, 80)
            }));
        }
    } catch {}

    if (steps.length === 0) return '';

    const rows = steps.map(s => `
        <div class="pipeline-step-row">
            <div class="pipeline-step-icon done">✓</div>
            <span class="pipeline-step-name">${escapeHtml(s.tool)}</span>
            <span style="color:var(--text-tertiary);font-size:12px;font-family:'JetBrains Mono',monospace; overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(s.input)}</span>
        </div>
    `).join('');

    return `
        <div class="pipeline-trace">
            <button class="pipeline-trace-toggle" onclick="togglePipeline('${id}')">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                ${steps.length} pipeline steps
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
            </button>
            <div class="pipeline-steps" id="${id}">${rows}</div>
        </div>
    `;
}

function togglePipeline(id) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('open');
}

/* ---- chart rendering ---- */
const CHART_COLORS = ['#818cf8', '#a78bfa', '#c4b5fd', '#6366f1', '#4f46e5', '#7c3aed'];

// Words that are never valid chart labels
const LABEL_BLACKLIST = /^(here|the|this|it|they|market|revenue|total|source|amount|income|sector|in|is|was|has|had|at|of|a|an|or|and|for|to|by|on|with|from|no|not|its|their|our|so|but|up|if|all|as|summary|finding|overview|note|data|result|approximately|about|around|comparison|missing|unfortunately|quarterly|annual|net|average|highest|lowest|per|which|that)$/i;

function tryRenderChart(msgDiv, answer) {
    if (typeof Chart === 'undefined') return;

    const lines = answer.split('\n');
    const dataPoints = [];

    // Strategy 1: Label and value on SAME line
    // e.g. "- **Apple**: $394.3 billion" or "1. Microsoft — $820 billion"
    for (const line of lines) {
        const listMatch = line.match(/^\s*(?:[-*•]|\d+[.)]\s)\s*(?:\*\*)?([A-Z][A-Za-z.& ']{1,25}?)(?:\*\*)?[\s:\-—]+.*?\$\s?([\d,.]+)\s*(billion|million|B|M|bn|mn|trillion|T)\b/i);
        if (listMatch) {
            const label = cleanLabel(listMatch[1]);
            if (!label) continue;
            const value = parseValue(listMatch[2], listMatch[3]);
            if (value > 0) dataPoints.push({ label, value });
        }
    }

    // Strategy 2: Label on one line, value on NEXT line(s)
    // e.g. "• **Tesla**\n  - Market Cap: $856.04 billion"
    if (dataPoints.length < 2) {
        dataPoints.length = 0; // reset
        let currentLabel = null;
        for (const line of lines) {
            // Check if line is a bold entity name (standalone label)
            const labelMatch = line.match(/^\s*(?:[-*•]|\d+[.)]\s)\s*\*\*([A-Z][A-Za-z.& ']{1,25}?)\*\*\s*:?\s*$/);
            if (labelMatch) {
                currentLabel = cleanLabel(labelMatch[1]);
                continue;
            }
            // Check if line has a dollar value and we have a pending label
            if (currentLabel) {
                const valMatch = line.match(/\$\s?([\d,.]+)\s*(billion|million|B|M|bn|mn|trillion|T)\b/i);
                if (valMatch) {
                    const value = parseValue(valMatch[1], valMatch[2]);
                    if (value > 0) dataPoints.push({ label: currentLabel, value });
                    currentLabel = null;
                }
            }
            // Reset label if we hit an empty line without finding value
            if (line.trim() === '' && currentLabel) {
                // keep currentLabel — value might be on next non-empty line
            }
            // Reset if we hit another bullet without finding value
            if (/^\s*(?:[-*•]|\d+[.)])/.test(line) && !line.match(/\*\*/) && !line.match(/\$/)) {
                // sub-bullet without value, keep looking
            }
        }
    }

    // Deduplicate by label (case-insensitive), keep first
    const seen = new Set();
    const unique = dataPoints.filter(d => {
        const key = d.label.toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });

    // Need 2-10 distinct entities
    if (unique.length < 2 || unique.length > 10) return;

    const chartContainer = document.createElement('div');
    chartContainer.className = 'chart-container';
    chartContainer.style.marginLeft = '40px';
    chartContainer.innerHTML = `<canvas></canvas><div class="chart-label">Auto-generated from response data (values in $B)</div>`;

    const bodyEl = msgDiv.querySelector('.message-body');
    bodyEl.after(chartContainer);

    const canvas = chartContainer.querySelector('canvas');
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: unique.map(d => d.label),
            datasets: [{
                label: 'Value ($B)',
                data: unique.map(d => d.value),
                backgroundColor: unique.map((_, i) => CHART_COLORS[i % CHART_COLORS.length] + '90'),
                borderColor: unique.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
                borderWidth: 1.5,
                borderRadius: 6,
                barPercentage: 0.6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2,
            layout: { padding: { bottom: 8 } },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#18181c',
                    titleColor: '#f4f4f5',
                    bodyColor: '#a1a1aa',
                    borderColor: 'rgba(255,255,255,.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 10,
                    callbacks: { label: ctx => `$${ctx.parsed.y.toFixed(1)}B` }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#a1a1aa', font: { size: 12, family: 'Inter', weight: '500' }, maxRotation: 0, minRotation: 0 },
                    border: { display: false }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,.05)', drawTicks: false },
                    ticks: { color: '#71717a', font: { size: 11, family: 'Inter' }, padding: 8, callback: v => `$${v}B` },
                    border: { display: false }
                }
            }
        }
    });
}

function cleanLabel(raw) {
    let label = raw.trim().replace(/\s+(leads?|with|has|had|is|was|at|of|the|follows?|next|comes?|for|and|or)$/i, '').trim();
    if (label.length < 2 || LABEL_BLACKLIST.test(label)) return null;
    return label;
}

function parseValue(numStr, unit) {
    let value = parseFloat(numStr.replace(/,/g, ''));
    const u = (unit || '').toLowerCase();
    if (['billion', 'b', 'bn'].includes(u)) { /* billions */ }
    else if (['million', 'm', 'mn'].includes(u)) value /= 1000;
    else if (['trillion', 't'].includes(u)) value *= 1000;
    return isNaN(value) ? 0 : value;
}

/* ---- utilities ---- */
function scrollToBottom() {
    const container = document.getElementById('chat-scroll');
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

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
    document.getElementById('sidebar-overlay').classList.toggle('open');
}

function formatMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    // headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');

    // bold & italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // bullet lists
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // numbered lists
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

    // tables (simple pipe tables)
    html = html.replace(/^(\|.+\|)\n(\|[-| :]+\|)\n((?:\|.+\|\n?)+)/gm, (match, header, sep, body) => {
        const ths = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
        const rows = body.trim().split('\n').map(row => {
            const tds = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
            return `<tr>${tds}</tr>`;
        }).join('');
        return `<table><thead><tr>${ths}</tr></thead><tbody>${rows}</tbody></table>`;
    });

    // paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';
    html = html.replace(/\n/g, '<br>');
    html = html.replace(/<p>\s*<\/p>/g, '');

    return html;
}
