// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const removeFileBtn = document.getElementById('remove-file-btn');
const processBtn = document.getElementById('process-btn');

const intensitySlider = document.getElementById('intensity-slider');
const intensityVal = document.getElementById('intensity-val');
const gammaSlider = document.getElementById('gamma-slider');
const gammaVal = document.getElementById('gamma-val');
const deltaSlider = document.getElementById('delta-slider');
const deltaVal = document.getElementById('delta-val');

const blankState = document.getElementById('blank-state');
const loadingState = document.getElementById('loading-state');
const loadingStatus = document.getElementById('loading-status');
const progressBar = document.getElementById('progress-bar');
const outputWorkspace = document.getElementById('output-workspace');

const imgBefore = document.getElementById('img-before');
const imgAfter = document.getElementById('img-after');
const txtBefore = document.getElementById('txt-before');
const txtAfter = document.getElementById('txt-after');
const audioBefore = document.getElementById('audio-before');
const audioAfter = document.getElementById('audio-after');

const imageSliderContainer = document.getElementById('image-slider-container');
const textComparison = document.getElementById('text-comparison');
const audioComparison = document.getElementById('audio-comparison');

const sliderHandle = document.getElementById('slider-handle');
const afterLayer = document.getElementById('after-layer');

const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

const downloadCardBtn = document.getElementById('download-card-btn');

// State Variables
let activeFile = null;
let activeFileType = null;
let rawFileBase64 = null;
let processedFileBlob = null;
let latestAnalysis = null;

// Event Listeners for Upload
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        setFile(e.dataTransfer.files[0]);
    }
});

removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetFile();
});

// Parameter Sliders
intensitySlider.addEventListener('input', () => {
    intensityVal.textContent = `${intensitySlider.value}%`;
});

gammaSlider.addEventListener('input', () => {
    gammaVal.textContent = `${gammaSlider.value}%`;
});

deltaSlider.addEventListener('input', () => {
    deltaVal.textContent = `${deltaSlider.value}%`;
});

// Tabs switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        const tabId = btn.getAttribute('data-tab');
        tabContents.forEach(content => {
            content.style.display = 'none';
        });
        document.getElementById(`tab-${tabId}`).style.display = 'flex';
    });
});

function setFile(file) {
    activeFile = file;
    fileName.textContent = file.name;
    fileInfo.style.display = 'flex';
    dropZone.style.display = 'none';
    processBtn.disabled = false;
    
    const ext = file.name.split('.').pop().toLowerCase();
    activeFileType = ext;
    
    // Toggle image parameter sliders visibility
    const isImage = ['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(ext);
    document.querySelectorAll('.image-only').forEach(el => {
        el.style.display = isImage ? 'block' : 'none';
    });

    // Reader for previewing raw file
    const reader = new FileReader();
    if (['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(ext)) {
        reader.onload = (e) => { rawFileBase64 = e.target.result; };
        reader.readAsDataURL(file);
    } else if (['txt', 'md', 'json'].includes(ext)) {
        reader.onload = (e) => { rawFileBase64 = e.target.result; };
        reader.readAsText(file);
    } else if (['wav'].includes(ext)) {
        reader.onload = (e) => { rawFileBase64 = e.target.result; };
        reader.readAsDataURL(file);
    }
}

function resetFile() {
    activeFile = null;
    activeFileType = null;
    rawFileBase64 = null;
    processedFileBlob = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    dropZone.style.display = 'block';
    processBtn.disabled = true;
    
    blankState.style.display = 'flex';
    loadingState.style.display = 'none';
    outputWorkspace.style.display = 'none';
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        setFile(e.target.files[0]);
    }
}

// Align Process Trigger
processBtn.addEventListener('click', async () => {
    if (!activeFile) return;

    // Transition UI
    blankState.style.display = 'none';
    outputWorkspace.style.display = 'none';
    loadingState.style.display = 'flex';
    progressBar.style.width = '0%';
    loadingStatus.textContent = "Connecting to WebSocket comparison gateway...";

    // 1. Establish WebSocket for Progress Monitoring
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/ws/compare`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        ws.send(JSON.stringify({ action: "start" }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        progressBar.style.width = `${msg.progress}%`;
        loadingStatus.textContent = msg.status;
        
        if (msg.progress === 100) {
            ws.close();
        }
    };

    // 2. Perform Detection and Humanization Parallel API calls
    try {
        const formData = new FormData();
        formData.append("file", activeFile);
        formData.append("intensity", (intensitySlider.value / 100.0).toString());
        formData.append("gamma", (gammaSlider.value / 100.0).toString());
        formData.append("delta", (deltaSlider.value / 100.0).toString());

        // Perform Parallel Fetch Requests
        const [detectRes, humanizeRes] = await Promise.all([
            fetch('/api/detect', { method: 'POST', body: formData }),
            fetch('/api/humanize', { method: 'POST', body: formData })
        ]);

        if (!detectRes.ok || !humanizeRes.ok) {
            throw new Error("Alignment failed at API level.");
        }

        const analysis = await detectRes.json();
        latestAnalysis = analysis;
        const processedBlob = await humanizeRes.blob();
        processedFileBlob = processedBlob;

        // Render Results
        renderWorkspace(analysis, processedBlob);
    } catch (err) {
        loadingStatus.textContent = `Error: ${err.message}`;
        progressBar.style.backgroundColor = 'var(--danger)';
    }
});

function renderWorkspace(analysis, processedBlob) {
    loadingState.style.display = 'none';
    outputWorkspace.style.display = 'flex';

    // Update Analytics Tab
    const scores = analysis.metadata;
    
    // Donut chart logic
    const donutScore = Math.round(scores.ai_score * 100);
    document.getElementById('meta-ai-score').textContent = `${donutScore}%`;
    const donutEl = document.querySelector('.ai-score-donut');
    donutEl.style.background = `conic-gradient(var(--danger) ${donutScore}%, rgba(255,255,255,0.05) 0)`;

    // Fill metrics list progress bars
    const fields = ['plastic', 'emotion', 'symmetry', 'lighting', 'texture', 'repetition'];
    fields.forEach(field => {
        const val = Math.round(scores[`${field}_score`] * 100);
        document.getElementById(`meta-${field}-val`).textContent = `${val}%`;
        document.getElementById(`meta-${field}-fill`).style.width = `${val}%`;
    });

    // Hide all containers initially
    imageSliderContainer.style.display = 'none';
    textComparison.style.display = 'none';
    audioComparison.style.display = 'none';

    // Show appropriate comparison container
    const isImage = ['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(activeFileType);
    const isText = ['txt', 'md', 'json'].includes(activeFileType);
    const isAudio = ['wav'].includes(activeFileType);

    if (isImage) {
        imageSliderContainer.style.display = 'block';
        imgBefore.src = rawFileBase64;
        imgAfter.src = URL.createObjectURL(processedBlob);
        
        // Reset slider coordinates
        afterLayer.style.width = '50%';
        sliderHandle.style.left = '50%';
        setupSliderEvents();
    } else if (isText) {
        textComparison.style.display = 'grid';
        txtBefore.textContent = rawFileBase64;
        
        const reader = new FileReader();
        reader.onload = (e) => { txtAfter.textContent = e.target.result; };
        reader.readAsText(processedBlob);
    } else if (isAudio) {
        audioComparison.style.display = 'flex';
        audioBefore.src = rawFileBase64;
        audioAfter.src = URL.createObjectURL(processedBlob);
    }
}

// Side-by-Side Slider Dragging Logic
let isDragging = false;

function setupSliderEvents() {
    sliderHandle.addEventListener('mousedown', () => { isDragging = true; });
    window.addEventListener('mouseup', () => { isDragging = false; });
    window.addEventListener('mousemove', dragSlider);

    // Touch events
    sliderHandle.addEventListener('touchstart', () => { isDragging = true; });
    window.addEventListener('touchend', () => { isDragging = false; });
    window.addEventListener('touchmove', dragSlider);
}

function dragSlider(e) {
    if (!isDragging) return;

    const containerRect = imageSliderContainer.getBoundingClientRect();
    let clientX = e.clientX;
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
    }

    let offsetX = clientX - containerRect.left;
    offsetX = Math.max(0, Math.min(offsetX, containerRect.width));
    
    const percentage = (offsetX / containerRect.width) * 100;
    
    afterLayer.style.width = `${percentage}%`;
    sliderHandle.style.left = `${percentage}%`;
}

// Client-Side Share Card Generation via HTML Canvas
downloadCardBtn.addEventListener('click', () => {
    if (!latestAnalysis) return;

    const canvas = document.createElement('canvas');
    canvas.width = 1000;
    canvas.height = 750;
    const ctx = canvas.getContext('2d');

    // 1. Draw Background
    ctx.fillStyle = '#131316';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 2. Draw Header
    ctx.fillStyle = '#1a1a20';
    ctx.fillRect(0, 0, canvas.width, 80);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 22px Outfit, sans-serif';
    ctx.fillText('QUIRKY COMPARE: PORTRAIT RESTORATION', 30, 48);

    // 3. Draw Section Dividers & Badges
    ctx.fillStyle = '#8e8e9f';
    ctx.font = 'bold 13px Outfit, sans-serif';
    ctx.fillText('BEFORE (Plastic Base)', 50, 125);
    ctx.fillStyle = '#10b981';
    ctx.fillText('AFTER (Pore Restored)', 530, 125);

    // 4. Draw Before & After Content
    const isImage = ['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(activeFileType);
    if (isImage) {
        const imageB = new Image();
        const imageA = new Image();
        imageB.src = rawFileBase64;
        imageA.src = URL.createObjectURL(processedFileBlob);

        imageB.onload = () => {
            ctx.drawImage(imageB, 50, 150, 420, 420);
            imageA.onload = () => {
                ctx.drawImage(imageA, 530, 150, 420, 420);
                drawMetricsAndDownload();
            };
        };
    } else {
        // Text/Other Fallback rendering
        ctx.fillStyle = '#1d1d24';
        ctx.strokeStyle = '#2e2e38';
        ctx.lineWidth = 1;
        ctx.fillRect(50, 150, 420, 420);
        ctx.strokeRect(50, 150, 420, 420);
        ctx.fillRect(530, 150, 420, 420);
        ctx.strokeRect(530, 150, 420, 420);

        ctx.fillStyle = '#d1d1e0';
        ctx.font = '14px Inter, sans-serif';
        
        const bText = txtBefore.textContent.slice(0, 200) + "...";
        const aText = txtAfter.textContent.slice(0, 200) + "...";
        
        wrapText(ctx, bText, 70, 180, 380, 24);
        wrapText(ctx, aText, 550, 180, 380, 24);
        drawMetricsAndDownload();
    }

    function drawMetricsAndDownload() {
        // Draw Metrics Block
        ctx.fillStyle = '#16161c';
        ctx.fillRect(0, 590, canvas.width, 100);

        ctx.font = 'bold 15px Outfit, sans-serif';
        ctx.fillStyle = '#ef4444';
        ctx.fillText(`AI SCORE: ${latestAnalysis.metadata.ai_score}`, 50, 625);
        ctx.fillText(`PLASTIC SCORE: ${latestAnalysis.metadata.plastic_score}`, 50, 655);

        ctx.fillStyle = '#10b981';
        ctx.fillText(`AI SCORE: ${latestAnalysis.metadata.ai_score * 0.25}`, 530, 625);
        ctx.fillText(`PLASTIC SCORE: ${latestAnalysis.metadata.plastic_score * 0.15}`, 530, 655);

        // Draw Footer
        ctx.fillStyle = '#0a0a0c';
        ctx.fillRect(0, 690, canvas.width, 60);

        ctx.fillStyle = '#8e8e9f';
        ctx.font = '12px Inter, sans-serif';
        ctx.fillText('Powered by Quirky', 30, 725);
        ctx.fillStyle = '#6366f1';
        ctx.font = 'bold 12px Outfit, sans-serif';
        ctx.fillText('by MITPO', 140, 725);

        // Download Trigger
        const link = document.createElement('a');
        link.download = `quirky_compare_${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    }
});

function wrapText(context, text, x, y, maxWidth, lineHeight) {
    const words = text.split(' ');
    let line = '';

    for (let n = 0; n < words.length; n++) {
        let testLine = line + words[n] + ' ';
        let metrics = context.measureText(testLine);
        let testWidth = metrics.width;
        if (testWidth > maxWidth && n > 0) {
            context.fillText(line, x, y);
            line = words[n] + ' ';
            y += lineHeight;
        } else {
            line = testLine;
        }
    }
    context.fillText(line, x, y);
}
