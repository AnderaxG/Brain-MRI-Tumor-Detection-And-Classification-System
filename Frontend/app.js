// static/js/app.js
(() => {
  // DOM references
  const dropArea = document.getElementById('dropArea');
  const fileInput = document.getElementById('fileInput');
  const previewImg = document.getElementById('previewImg');
  const placeholder = document.getElementById('placeholderText');
  const clearBtn = document.getElementById('clearBtn');
  const predictBtn = document.getElementById('predictBtn');
  const resultsBox = document.getElementById('resultsBox');
  const barsContainer = document.getElementById('barsContainer');
  const predictedLabel = document.getElementById('predictedLabel');
  const predictedConfidence = document.getElementById('predictedConfidence');
  const viewJsonBtn = document.getElementById('viewJsonBtn');
  const jsonBox = document.getElementById('jsonBox');

  let currentFile = null;
  const MAX_BYTES = 10 * 1024 * 1024; // 10MB

  // helper: show/hide
  const show = (el) => { if (!el) return; el.classList.remove('hidden'); el.style.display = 'block'; };
  const hide = (el) => { if (!el) return; el.classList.add('hidden'); el.style.display = 'none'; };

  // enable/disable predict
  const updatePredictState = () => {
    predictBtn.disabled = !currentFile;
    predictBtn.style.opacity = predictBtn.disabled ? '0.6' : '1';
  };

  // reset UI state related to file
  const clearFile = () => {
    currentFile = null;
    previewImg.src = '';
    previewImg.style.display = 'none';
    clearBtn.style.display = 'none';
    placeholder.style.display = 'block';
    hide(resultsBox);
    hide(jsonBox);
    barsContainer.innerHTML = '';
    predictedLabel.textContent = '';
    predictedConfidence.textContent = '';
    updatePredictState();
  };

  // load file into preview
  const previewSelectedFile = (file) => {
    if (!file) return clearFile();
    // size check
    if (file.size > MAX_BYTES) {
      alert('File is too large. Max size ~10MB.');
      return clearFile();
    }
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
      previewImg.style.display = 'block';
      placeholder.style.display = 'none';
      clearBtn.style.display = 'inline-block';
      currentFile = file;
      updatePredictState();
    };
    reader.readAsDataURL(file);
  };

  // drop handlers
  function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
  ['dragenter','dragover','dragleave','drop'].forEach(ev => {
    dropArea.addEventListener(ev, preventDefaults, false);
  });
  ['dragenter','dragover'].forEach(ev => {
    dropArea.addEventListener(ev, () => dropArea.classList.add('dragover'), false);
  });
  ['dragleave','drop'].forEach(ev => {
    dropArea.addEventListener(ev, () => dropArea.classList.remove('dragover'), false);
  });
  dropArea.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (!dt) return;
    const files = dt.files;
    if (!files || files.length === 0) return;
    const f = files[0];
    if (!f.type.startsWith('image/')) { alert('Please upload an image file (jpg/png).'); return; }
    previewSelectedFile(f);
  });

  // choose file button
  fileInput.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    if (!f.type.startsWith('image/')) { alert('Only image files allowed (jpg/png).'); fileInput.value = ''; return; }
    previewSelectedFile(f);
  });

  // clear
  clearBtn.addEventListener('click', (e) => {
    e.preventDefault();
    fileInput.value = '';
    clearFile();
  });

  // keyboard accessibility: hitting Enter on dropArea opens file chooser
  dropArea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      fileInput.click();
      e.preventDefault();
    }
  });

  // view JSON toggle
  let jsonVisible = false;
  viewJsonBtn.addEventListener('click', (e) => {
    e.preventDefault();
    jsonVisible = !jsonVisible;
    if (jsonVisible) {
      show(jsonBox);
      jsonBox.setAttribute('aria-hidden', 'false');
      viewJsonBtn.textContent = 'Hide JSON';
    } else {
      hide(jsonBox);
      jsonBox.setAttribute('aria-hidden', 'true');
      viewJsonBtn.textContent = 'View JSON';
    }
  });

  // helper to build bar row
  const makeBarRow = (label, pct, idx, isTop) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'result-row';

    const lab = document.createElement('div');
    lab.className = 'result-label';
    lab.textContent = label;

    const progress = document.createElement('div');
    progress.className = 'progress';
    const inner = document.createElement('i');
    // all bars sky-blue by default; top prediction will get 'pct-main' (white)
    inner.className = isTop ? 'pct-main' : 'pct-sky';
    inner.style.width = '0%'; // animate in
    progress.appendChild(inner);

    const pctEl = document.createElement('div');
    pctEl.className = 'result-pct';
    pctEl.textContent = (pct * 100).toFixed(1) + '%';

    wrapper.appendChild(lab);
    wrapper.appendChild(progress);
    wrapper.appendChild(pctEl);

    // animate width
    setTimeout(() => { inner.style.width = (pct * 100) + '%'; }, 60);
    return wrapper;
  };

  // render results JSON
  const renderResults = (data) => {
    if (!data) return;
    predictedLabel.textContent = data.predicted_label || 'Unknown';
    predictedConfidence.textContent = data.probability ? ('Confidence: ' + (data.probability * 100).toFixed(1) + '%') : '';

    const probs = data.all_probs || {};
    const entries = Object.entries(probs);
    entries.sort((a,b) => b[1] - a[1]);

    // clear old
    barsContainer.innerHTML = '';

    // fill bars: mark the top index as white
    entries.forEach((pair, idx) => {
      const label = pair[0];
      const pct = Number(pair[1]) || 0;
      const isTop = (idx === 0);
      const row = makeBarRow(label, pct, idx, isTop);
      barsContainer.appendChild(row);
    });

    // fill JSON box
    jsonBox.textContent = JSON.stringify(data, null, 2);
    if (jsonVisible) show(jsonBox);

    show(resultsBox);
  };

  // send file to server
  const doPredict = async () => {
    if (!currentFile) return alert('Choose an image first.');
    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting...';

    try {
      const fd = new FormData();
      fd.append('file', currentFile, currentFile.name);

      const resp = await fetch('/predict', {
        method: 'POST',
        body: fd
      });

      if (!resp.ok) {
        const txt = await resp.text().catch(()=>null);
        throw new Error('Server error ' + resp.status + (txt ? ': ' + txt : ''));
      }
      const j = await resp.json();
      renderResults(j);
    } catch (err) {
      console.error('Predict error:', err);
      alert('Prediction failed: ' + (err.message || err));
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = 'Predict';
    }
  };

  // handle click predict
  predictBtn.addEventListener('click', (e) => {
    e.preventDefault();
    doPredict();
  });

  // initialize
  clearFile();

  // expose a programmatic function (optional)
  window._mriFrontend = {
    previewSelectedFile,
    clearFile,
  };
})();